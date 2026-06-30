"""Inference profiler helper script for nanoGPT-Modern.

Usage::

    python profile_inference.py \
        --checkpoint out/best_ckpt.pt \
        --prompt "The future of artificial intelligence is" \
        --max_new_tokens 200 \
        --num_samples 10 \
        --output_dir profiles/inference

Runs a short generation benchmark under ``torch.profiler`` and exports a Chrome
trace + memory summary + per-token latency report.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer_base import load_model_from_checkpoint, set_seed
from utils.profiler import Profiler, InferenceProfiler


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--model", type=str, default="modern", choices=["baseline", "modern"]
    )
    p.add_argument(
        "--prompt", type=str, default="The future of artificial intelligence is"
    )
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--output_dir", type=str, default="profiles/inference")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--use_cache", action="store_true", default=True)
    return p.parse_args()


def main():
    args = get_args()
    set_seed(1337)
    device = args.device

    model, ckpt = load_model_from_checkpoint(
        args.checkpoint, device=device, model_type=args.model
    )
    if args.compile and device == "cuda":
        model = torch.compile(
            model, mode="reduce-overhead", fullgraph=False, dynamic=True
        )
    model.to(device).eval()

    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception:
        tokenizer = None

    if tokenizer:
        prompt_ids = torch.tensor(
            [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
        )
    else:
        prompt_ids = torch.tensor(
            [[ord(c) for c in args.prompt]], dtype=torch.long, device=device
        )

    print(
        f"Model: {args.model} | params={sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"Prompt tokens: {prompt_ids.shape[1]}")

    # Full profiler for the first sample.
    prof = Profiler(
        output_dir=args.output_dir,
        chrome_trace=True,
        memory_summary=True,
        with_stack=False,
    )
    with prof:
        with torch.no_grad():
            _ = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                use_cache=args.use_cache,
            )

    trace_path = prof.export_chrome_trace("inference_trace.json")
    mem_path = prof.export_memory_summary("inference_memory.txt")
    stats_path = prof.export_stats("inference_stats.json")
    print(f"Chrome trace: {trace_path}")
    print(f"Memory summary: {mem_path}")
    print(f"Stats: {stats_path}")

    # Lightweight per-token profiler for all samples.
    inf_prof = InferenceProfiler(output_dir=args.output_dir, device=device)
    with inf_prof:
        for run in range(args.num_samples):
            with torch.no_grad():
                out = model.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    use_cache=args.use_cache,
                )
            # Record one step per token (approximate).
            for _ in range(args.max_new_tokens):
                inf_prof.step()
            print(
                f"Run {run + 1}: generated {out.shape[1] - prompt_ids.shape[1]} tokens"
            )

    summary = inf_prof.summary()
    print(f"Median token latency: {summary.get('median_ms', 0):.2f} ms")
    print(f"Throughput: {summary.get('tok_per_sec', 0):.2f} tok/s")
    print(f"Peak memory: {summary.get('peak_mem_mb', 0):.1f} MB")
    inf_prof.export("inference_profile.json")


if __name__ == "__main__":
    main()
