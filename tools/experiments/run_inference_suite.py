"""E-B: decode throughput and memory for the KV-cache family.

Loads the modern-full checkpoint from the architecture matrix and times
greedy decode. A separate small draft model is trained only when
``--draft_iters`` is positive, then used for the speculative-decoding sweep.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from training.trainer_base import load_model_from_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--prompt_lengths", type=int, nargs="+", default=[128, 512, 896])
    parser.add_argument("--new_tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--draft_iters", type=int, default=1200)
    parser.add_argument("--draft_out", default="out/ablations/draft")
    parser.add_argument("--data_dir", default="data/openwebtext")
    return parser.parse_args()


def apply_variant(model, variant):
    model.config.use_paged_kv_cache = variant == "paged"
    model.config.kv_cache_dtype = "int8" if variant == "int8" else "bf16"


def time_generate(model, prompt_len, new_tokens, use_cache, repeats):
    device = next(model.parameters()).device
    prompt = torch.randint(0, 1000, (1, prompt_len), device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.generate(prompt, max_new_tokens=4, temperature=0.0, top_k=1, use_cache=use_cache)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model.generate(
            prompt,
            max_new_tokens=new_tokens,
            temperature=0.0,
            top_k=1,
            use_cache=use_cache,
        )
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)
    elapsed = sorted(samples)[len(samples) // 2]
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    return {
        "tok_per_s": new_tokens / max(elapsed, 1e-6),
        "seconds": elapsed,
        "peak_mb": peak_mb,
    }


def train_draft(args):
    out = Path(args.draft_out)
    ckpt = out / "best_ckpt.pt"
    if ckpt.is_file():
        return ckpt
    latest = out / "latest_ckpt.pt"
    if latest.is_file():
        return latest
    cmd = [
        sys.executable,
        "-m",
        "training.train_pretrain",
        "--config",
        "config/pretrain_cloud.yaml",
        "--model",
        "modern",
        "--n_layer",
        "4",
        "--n_head",
        "4",
        "--n_embd",
        "256",
        "--n_kv_head",
        "2",
        "--data_dir",
        args.data_dir,
        "--out_dir",
        str(out),
        "--exp_id",
        "draft",
        "--max_iters",
        str(args.draft_iters),
        "--warmup_iters",
        str(max(20, args.draft_iters // 10)),
        "--lr_decay_iters",
        str(args.draft_iters),
        "--eval_interval",
        str(max(100, args.draft_iters // 2)),
        "--eval_iters",
        "4",
        "--batch_size",
        "32",
    ]
    print("draft", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if ckpt.is_file():
        return ckpt
    return latest


def speculative_rows(target, draft, new_tokens):
    import model.modern_gpt as gpt

    device = next(target.parameters()).device
    prompt = torch.randint(0, 1000, (1, 128), device=device)
    rows = []
    original = gpt._verify
    for draft_len in (2, 4, 8):
        stats = {"accepted": 0, "proposed": 0}

        def wrapped(draft_tokens, target_logits, draft_probs, temperature, _stats=stats):
            accepted, n_accepted = original(draft_tokens, target_logits, draft_probs, temperature)
            _stats["accepted"] += int(n_accepted)
            _stats["proposed"] += int(draft_tokens.shape[1])
            return accepted, n_accepted

        gpt._verify = wrapped
        target.generate(
            prompt,
            max_new_tokens=8,
            temperature=0.0,
            top_k=1,
            use_cache=True,
            draft_model=draft,
            draft_tokens=draft_len,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        target.generate(
            prompt,
            max_new_tokens=new_tokens,
            temperature=0.0,
            top_k=1,
            use_cache=True,
            draft_model=draft,
            draft_tokens=draft_len,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        gpt._verify = original
        proposed = max(stats["proposed"], 1)
        rows.append(
            {
                "variant": f"speculative-{draft_len}",
                "prompt_len": 128,
                "draft_len": draft_len,
                "tok_per_s": new_tokens / max(elapsed, 1e-6),
                "accept_rate": stats["accepted"] / proposed,
                "peak_mb": torch.cuda.max_memory_allocated() / (1024**2),
            }
        )
        print(rows[-1], flush=True)
    return rows


def continuous_rows(model):
    from inference.continuous_batching import ContinuousBatchScheduler, Request, RequestQueue

    device = next(model.parameters()).device
    rows = []
    n_req = 16
    new_tokens = 32
    for concurrency in (1, 4, 16):
        queue = RequestQueue(maxsize=n_req + 2)
        for i in range(n_req):
            ids = torch.randint(0, 1000, (32,)).tolist()
            queue.enqueue(
                Request(
                    prompt_ids=ids,
                    max_tokens=new_tokens,
                    temperature=0.0,
                    top_k=1,
                    request_id=str(i),
                )
            )
        scheduler = ContinuousBatchScheduler(queue, model, device=device, batch_size=concurrency)
        torch.cuda.synchronize()
        start = time.perf_counter()
        scheduler.run()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "variant": f"continuous-b{concurrency}",
                "prompt_len": 32,
                "concurrency": concurrency,
                "tok_per_s": (n_req * new_tokens) / max(elapsed, 1e-6),
                "seconds": elapsed,
                "requests": n_req,
            }
        )
        print(rows[-1], flush=True)
    return rows


def main():
    args = get_args()
    device = "cuda"
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    rows = []
    for variant, use_cache in (
        ("no-cache", False),
        ("append", True),
        ("paged", True),
        ("int8", True),
    ):
        apply_variant(model, variant)
        for prompt_len in args.prompt_lengths:
            try:
                measured = time_generate(
                    model, prompt_len, args.new_tokens, use_cache, args.repeats
                )
            except Exception as exc:
                measured = {"error": str(exc)}
            row = {"variant": variant, "prompt_len": prompt_len, **measured}
            rows.append(row)
            print(row, flush=True)

    apply_variant(model, "append")
    try:
        eager = time_generate(model, 128, args.new_tokens, True, args.repeats)
        compiled = time_generate(model, 128, args.new_tokens, True, 1)
        # A dedicated compiled call. generate(compile=True) compiles internally.
        prompt = torch.randint(0, 1000, (1, 128), device=device)
        model.generate(
            prompt, max_new_tokens=4, temperature=0.0, top_k=1, use_cache=True, compile=True
        )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model.generate(
            prompt,
            max_new_tokens=args.new_tokens,
            temperature=0.0,
            top_k=1,
            use_cache=True,
            compile=True,
        )
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
        rows.append({"variant": "eager", "prompt_len": 128, **eager})
        rows.append(
            {
                "variant": "compile",
                "prompt_len": 128,
                "tok_per_s": args.new_tokens / max(elapsed, 1e-6),
                "seconds": elapsed,
                "eager_tok_per_s": eager["tok_per_s"],
            }
        )
        print(rows[-1], flush=True)
        del compiled
    except Exception as exc:
        rows.append({"variant": "compile", "error": str(exc)})
        print("compile failed", exc, flush=True)

    try:
        rows.extend(continuous_rows(model))
    except Exception as exc:
        rows.append({"variant": "continuous", "error": str(exc)})
        print("continuous failed", exc, flush=True)

    if args.draft_iters > 0:
        try:
            draft_path = train_draft(args)
            draft, _ = load_model_from_checkpoint(str(draft_path), device=device)
            draft.eval()
            rows.extend(speculative_rows(model, draft, args.new_tokens))
        except Exception as exc:
            rows.append({"variant": "speculative", "error": str(exc)})
            print("speculative failed", exc, flush=True)

    payload = {"checkpoint": args.checkpoint, "new_tokens": args.new_tokens, "rows": rows}
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
