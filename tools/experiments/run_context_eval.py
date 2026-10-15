"""E-A-8: RoPE length-extension comparison (naive / ntk / pi / yarn) PPL.

The checkpoint was trained at ``--train_block`` (default 1024).  For each
evaluation length, the extension factor is ``max(1.0, length / train_block)``
and each method re-tunes the loaded model's RoPE tables in place via
``rope_retune.retune_rope`` before measuring perplexity over fixed val-bin
windows (same window semantics as run_ntk_eval.py / E-A-7).

Usage (repo root):
    python tools/experiments/run_context_eval.py \
        --checkpoint out/pretrain/ckpt_50m.pt --output_json docs/results/context_eval.json \
        --lengths 512 1024 2048 4096 --methods naive ntk pi yarn
    # low-compute tier: --quick  (lengths=[512, 1024], num_batches=4)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.experiments.rope_retune import LEGAL_METHODS, retune_rope  # noqa: E402
from training.trainer_base import load_model_from_checkpoint  # noqa: E402


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--val_bin", default="data/openwebtext/val.bin")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--methods", nargs="+", default=list(LEGAL_METHODS), choices=LEGAL_METHODS)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train_block", type=int, default=1024)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="low-compute tier: lengths=[512, 1024], num_batches=4",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.lengths = [512, 1024]
        args.num_batches = 4
    return args


def windows(data, length, batch_size, num_batches, seed):
    """Yield fixed token windows. The same seed gives the same windows.

    Same semantics as run_ntk_eval.py (E-A-7): [batch, length+1] uint16
    windows sampled with a seeded generator over the val memmap.
    """
    rng = np.random.default_rng(seed)
    usable = len(data) - length - 1
    if usable <= 0:
        raise RuntimeError(f"val set shorter than context {length}")
    for _ in range(num_batches):
        starts = rng.integers(0, usable, size=batch_size)
        batch = np.stack([data[s : s + length + 1] for s in starts])
        yield batch


@torch.no_grad()
def perplexity(model, data, length, batch_size, num_batches, seed, device):
    """Window-averaged NLL/PPL over the val memmap (E-A-7 semantics)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in windows(data, length, batch_size, num_batches, seed):
        tokens = torch.from_numpy(batch.astype(np.int64)).to(device)
        x = tokens[:, :-1].contiguous()
        y = tokens[:, 1:].contiguous()
        logits, loss, _ = model(x, targets=y)
        if loss is None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            )
        total_loss += float(loss.item()) * y.numel()
        total_tokens += y.numel()
    nll = total_loss / max(total_tokens, 1)
    return {"nll": nll, "ppl": math.exp(nll), "tokens": total_tokens}


def _run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    max_len = max(args.lengths)
    rows = []
    for length in args.lengths:
        factor = max(1.0, length / float(args.train_block))
        for method in args.methods:
            retune_rope(model, method, factor, max_len, args.train_block)
            stats = perplexity(
                model,
                data,
                length,
                args.batch_size,
                args.num_batches,
                args.seed + length,
                device,
            )
            temperature = 1.0 + 0.1 * math.log(factor) if method == "yarn" else 1.0
            row = {
                "length": length,
                "method": method,
                "factor": factor,
                **stats,
                "temperature": temperature,
            }
            rows.append(row)
            print(
                f"L={length} {method} factor={factor:.2f} "
                f"ppl={stats['ppl']:.3f} nll={stats['nll']:.4f}",
                flush=True,
            )
    payload = {
        "checkpoint": args.checkpoint,
        "train_block": args.train_block,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "rows": rows,
    }
    tmp = args.output_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, args.output_json)
    print("wrote", args.output_json)


def main(argv: list[str] | None = None) -> int:
    args = get_args(argv)
    if not args.checkpoint or not os.path.isfile(args.checkpoint):
        print(f"[error] checkpoint missing or not found: {args.checkpoint!r}", file=sys.stderr)
        return 1
    if not args.output_json:
        print("[error] --output_json is required", file=sys.stderr)
        return 1
    if not os.path.isfile(args.val_bin):
        print(f"[error] val_bin not found: {args.val_bin}", file=sys.stderr)
        return 1
    try:
        _run(args)
    except Exception as exc:  # noqa: BLE001 - tool entry: report, exit 1, no partial JSON
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
