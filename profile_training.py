"""Training profiler helper script for nanoGPT-Modern.

Usage::

    python profile_training.py \
        --checkpoint out/best_ckpt.pt \
        --steps 50 \
        --output_dir profiles/train

Runs a short training loop (pretrain or SFT) under ``torch.profiler`` and
exports a Chrome trace + memory summary.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer_base import load_model_from_checkpoint, set_seed
from utils.profiler import Profiler
from torch.utils.data import DataLoader


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--model", type=str, default="modern", choices=["baseline", "modern"]
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--output_dir", type=str, default="profiles/train")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def build_dummy_data(batch_size: int, block_size: int, steps: int, device: str):
    """Synthetic dataset for quick profiling."""
    x = torch.randint(0, 50257, (batch_size * steps, block_size + 1), device=device)
    return x[:, :-1], x[:, 1:]


def main():
    args = get_args()
    set_seed(1337)
    device = args.device

    model, ckpt = load_model_from_checkpoint(
        args.checkpoint, device=device, model_type=args.model
    )
    if args.compile and device == "cuda":
        model = torch.compile(model)

    model.to(device).train()

    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type="cuda" if device.startswith("cuda") else "cpu",
    )

    inputs, targets = build_dummy_data(
        args.batch_size, args.block_size, args.steps, device
    )
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    prof = Profiler(
        output_dir=args.output_dir,
        chrome_trace=True,
        memory_summary=True,
        with_stack=True,
    )
    with prof:
        for step, (x, y) in enumerate(loader):
            if step >= args.steps:
                break
            with torch.cuda.amp.autocast(
                device_type="cuda",
                dtype=(
                    torch.bfloat16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            ):
                logits, loss, _ = model(x, targets=y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print(f"step {step}: loss={loss.item():.4f}")

    trace_path = prof.export_chrome_trace("train_trace.json")
    mem_path = prof.export_memory_summary("train_memory.txt")
    stats_path = prof.export_stats("train_stats.json")
    print(
        f"Profiling complete.\n  Chrome trace: {trace_path}\n  Memory summary: {mem_path}\n  Stats: {stats_path}"
    )


if __name__ == "__main__":
    main()
