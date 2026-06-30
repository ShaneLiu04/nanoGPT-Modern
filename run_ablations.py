"""Run a configurable ablation matrix and summarize results.

Examples
--------
Quick inference ablation (small model, few tokens)::

    python run_ablations.py --mode inference \
        --checkpoint out/pretrain/best_ckpt.pt \
        --max_new_tokens 100 --num_samples 10

Training ablation on a tiny model (for CI/smoke)::

    python run_ablations.py --mode train \
        --data_dir data/openwebtext_test \
        --max_iters 100 --n_layer 2 --n_head 2 --n_embd 64
"""

import os
import sys
import json
import subprocess
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="inference", choices=["inference", "train"]
    )
    parser.add_argument("--checkpoint", type=str, default="out/pretrain/best_ckpt.pt")
    parser.add_argument("--data_dir", type=str, default="data/openwebtext_test")
    parser.add_argument("--out_json", type=str, default="out/ablations.json")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument(
        "--prompt", type=str, default="The future of artificial intelligence is"
    )
    parser.add_argument("--max_iters", type=int, default=500)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def run(cmd):
    print(f"\n[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def parse_inference_output(stdout):
    """Parse JSON output from inference/generate.py."""
    lines = stdout.strip().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def inference_ablations(args):
    results = []
    for model in ["baseline", "modern"]:
        for use_cache in [False, True]:
            # Skip baseline cache since BaselineGPT may not support cache API.
            if model == "baseline" and use_cache:
                continue
            cmd = [
                sys.executable,
                "inference/generate.py",
                "--checkpoint",
                args.checkpoint,
                "--model",
                model,
                "--prompt",
                args.prompt,
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--num_samples",
                str(args.num_samples),
                "--device",
                args.device,
                "--use_cache",
                str(use_cache),
                "--output_json",
                f"out/ablation_{model}_cache{use_cache}.json",
            ]
            stdout = run(cmd)
            metrics = parse_inference_output(stdout)
            results.append(
                {
                    "model": model,
                    "use_cache": use_cache,
                    **metrics,
                }
            )
    return results


def train_ablations(args):
    results = []
    for model in ["baseline", "modern"]:
        out_dir = f"out/ablation_train_{model}"
        cmd = [
            sys.executable,
            "training/train_pretrain.py",
            "--data_dir",
            args.data_dir,
            "--out_dir",
            out_dir,
            "--model",
            model,
            "--n_layer",
            str(args.n_layer),
            "--n_head",
            str(args.n_head),
            "--n_embd",
            str(args.n_embd),
            "--block_size",
            str(args.block_size),
            "--batch_size",
            str(args.batch_size),
            "--max_iters",
            str(args.max_iters),
            "--eval_interval",
            str(max(1, args.max_iters // 5)),
            "--eval_iters",
            "10",
            "--log_interval",
            "50",
            "--device",
            args.device,
            "--num_workers",
            "0",
        ]
        stdout = run(cmd)
        # Extract final val loss from stdout.
        final_val = None
        for line in stdout.splitlines():
            if "val loss" in line.lower():
                try:
                    final_val = float(
                        line.split("val loss")[-1].strip().rstrip(",").split()[0]
                    )
                except Exception:
                    pass
        results.append(
            {
                "model": model,
                "final_val_loss": final_val,
                "out_dir": out_dir,
            }
        )
    return results


def main():
    args = get_args()
    os.makedirs("out", exist_ok=True)

    if args.mode == "inference":
        results = inference_ablations(args)
    else:
        results = train_ablations(args)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print markdown table.
    print("\n## Ablation Results")
    if args.mode == "inference":
        print("| model | use_cache | prefill_ms | decode_ms | total_ms | tok/s |")
        print("|-------|-----------|------------|-----------|----------|-------|")
        for r in results:
            print(
                f"| {r.get('model')} | {r.get('use_cache')} | "
                f"{r.get('prefill_ms', 'N/A')} | {r.get('decode_ms', 'N/A')} | "
                f"{r.get('total_ms', 'N/A')} | {r.get('decode_tok_s', 'N/A')} |"
            )
    else:
        print("| model | final_val_loss |")
        print("|-------|----------------|")
        for r in results:
            print(f"| {r.get('model')} | {r.get('final_val_loss', 'N/A')} |")

    print(f"\n[Ablation] Results saved to {args.out_json}")


if __name__ == "__main__":
    main()
