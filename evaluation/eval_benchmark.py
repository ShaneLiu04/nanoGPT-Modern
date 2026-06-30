"""Standardized benchmark evaluation for pretrained checkpoints.

Supports:
* Perplexity on a local tokenized val set (always available).
* Zero-shot downstream tasks via ``lm-eval`` if installed and a HuggingFace
  format checkpoint is available (see ``--hf_dir``).

Examples
--------
::

    # Local perplexity only
    python evaluation/eval_benchmark.py \
        --checkpoint out/pretrain/best_ckpt.pt \
        --data_dir data/openwebtext --split val

    # With downstream tasks via a pre-exported HF checkpoint
    python export_to_hf.py --checkpoint out/pretrain/best_ckpt.pt --out_dir hf/nanogpt-modern
    python evaluation/eval_benchmark.py \
        --checkpoint out/pretrain/best_ckpt.pt \
        --hf_dir hf/nanogpt-modern \
        --data_dir data/openwebtext --split val \
        --tasks hellaswag,lambada_openai
"""

import argparse
import math
import json
import tempfile
import shutil

import torch
import torch.nn.functional as F


from data.openwebtext import get_dataloader
from training.trainer_base import load_model_from_checkpoint
from utils.config import parse_args_with_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint"
    )
    parser.add_argument(
        "--hf_dir",
        type=str,
        default=None,
        help="HuggingFace-format directory for lm-eval (auto-exported from "
        "--checkpoint if omitted and the checkpoint is ModernGPT)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/openwebtext",
        help="Directory containing {split}.bin / {split}.idx",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument(
        "--num_batches",
        type=int,
        default=None,
        help="Max batches to evaluate (default: all)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated lm-eval tasks (requires lm-eval package)",
    )
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    return parse_args_with_config(parser)


@torch.no_grad()
def evaluate_ppl(model, dataloader, device, num_batches=None):
    """Compute perplexity on a tokenized language-modeling dataset."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = 0
    for batch in dataloader:
        if num_batches is not None and n >= num_batches:
            break
        x, y = batch
        x, y = x.to(device), y.to(device)
        logits, loss, _ = model(x, targets=y)
        if loss is not None:
            # loss returned by model is mean over tokens; recover total.
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        else:
            # Fallback: compute cross-entropy manually.
            B, T = y.shape
            logits = logits.view(B * T, -1)
            targets = y.view(B * T)
            ce = F.cross_entropy(logits, targets, reduction="sum")
            total_loss += ce.item()
            total_tokens += targets.numel()
        n += 1

    if total_tokens == 0:
        return float("inf")
    avg_nll = total_loss / total_tokens
    ppl = math.exp(avg_nll)
    return ppl


def _auto_export_hf(checkpoint_path, device):
    """Export a ModernGPT checkpoint to a temporary HF directory for lm-eval.

    Returns the temporary directory path, or None if the checkpoint is not a
    ModernGPT checkpoint or the export fails.
    """
    try:
        from model.hf_model import NanoGPTModernConfig, NanoGPTModernForCausalLM
        from model.modern_gpt import ModernGPTConfig
    except Exception as e:
        print(f"[WARNING] Cannot import HF model wrappers: {e}")
        return None

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[WARNING] Failed to load checkpoint for HF export: {e}")
        return None

    raw_config = ckpt.get("config", {})
    if isinstance(raw_config, dict):
        model_type = raw_config.get("model_type", "modern")
    else:
        model_type = getattr(raw_config, "model_type", "modern")

    if model_type != "modern":
        print(
            f"[INFO] Auto HF export only supports ModernGPT; checkpoint is '{model_type}'."
        )
        return None

    tmp_dir = tempfile.mkdtemp(prefix="nanogpt_hf_")
    try:
        if isinstance(raw_config, dict):
            nano_config = ModernGPTConfig.from_dict(raw_config)
        else:
            nano_config = raw_config
        hf_config = NanoGPTModernConfig.from_nanogpt_config(nano_config)
        wrapper = NanoGPTModernForCausalLM(hf_config)
        wrapper.model.load_state_dict(ckpt["model"])
        wrapper.save_pretrained(tmp_dir, safe_serialization=True)
        return tmp_dir
    except Exception as e:
        print(f"[WARNING] Auto HF export failed: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None


def run_lm_eval(hf_dir, tasks, device):
    """Run lm-evaluation-harness on a HuggingFace-format checkpoint."""
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[WARNING] lm-eval not installed. Skipping downstream tasks.")
        return {}

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    if not task_list:
        return {}

    print(f"[INFO] Running lm-eval tasks: {tasks}")
    try:
        results = evaluator.simple_evaluate(
            model=HFLM(pretrained=hf_dir),
            tasks=task_list,
            device=device,
            batch_size="auto",
        )
        return results.get("results", {})
    except Exception as e:
        print(f"[WARNING] lm-eval failed: {e}")
        return {}


def main():
    args = get_args()
    device = args.device

    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    # Use the model's native block size so we never feed longer sequences
    # than the checkpoint was trained with.
    block_size = getattr(model.config, "block_size", args.block_size)
    dataloader = get_dataloader(
        args.data_dir,
        args.split,
        args.batch_size,
        block_size,
        num_workers=4,
        use_packing=False,
    )

    ppl = evaluate_ppl(model, dataloader, device, args.num_batches)
    print(f"{args.split} perplexity: {ppl:.4f}")

    results = {"perplexity": ppl}
    tmp_hf_dir = None
    if args.tasks:
        hf_dir = args.hf_dir
        if hf_dir is None:
            tmp_hf_dir = _auto_export_hf(args.checkpoint, device)
            hf_dir = tmp_hf_dir

        if hf_dir is not None:
            lm_results = run_lm_eval(hf_dir, args.tasks, device)
            if lm_results:
                results["lm_eval"] = lm_results
        else:
            print(
                "[INFO] Skipping lm-eval: provide --hf_dir with a HuggingFace-format "
                "checkpoint, or use a ModernGPT checkpoint so it can be auto-exported."
            )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.output_json}")

    if tmp_hf_dir is not None:
        shutil.rmtree(tmp_hf_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
