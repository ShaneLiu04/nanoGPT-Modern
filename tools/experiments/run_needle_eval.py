"""E-A-9: Needle-in-Haystack depth x context grid experiment entry.

Wraps ``evaluation/needle_in_haystack.py::NeedleEvaluator`` with the repo CLI
conventions (atomic JSON write, exit 0/1/2 semantics, --quick low-compute
tier).  Optionally re-tunes the loaded model's RoPE first (naive/ntk/pi/yarn
via ``rope_retune.retune_rope``) so the same grid can be run at extended
contexts.

Tokenizer is tiktoken "gpt2" (50257 vocab, matching this repo's checkpoints).
Experiment inputs (checkpoint / tokenizer) fail loudly with exit 1 -- no
silent degradation (this is an experiment entry, not a probe tool).

Usage (repo root):
    python tools/experiments/run_needle_eval.py \
        --checkpoint out/pretrain/ckpt_50m.pt --output_json docs/results/needle_eval.json
    # low-compute tier: --quick  ([512,1024] x [0.0,0.5,1.0] x 1 trial)
    # extended context study: --rope_method yarn --rope_factor 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from evaluation.needle_in_haystack import NeedleEvaluator  # noqa: E402
from tools.experiments.rope_retune import retune_rope  # noqa: E402
from training.trainer_base import load_model_from_checkpoint  # noqa: E402


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--max_context_length", type=int, default=4096)
    parser.add_argument(
        "--rope_method",
        default="none",
        choices=["none", "ntk", "pi", "yarn"],
        help="retune RoPE before evaluating (none = keep as trained)",
    )
    parser.add_argument("--rope_factor", type=float, default=1.0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="low-compute tier: [512,1024] x [0.0,0.5,1.0] x 1 trial",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.context_lengths = [512, 1024]
        args.depths = [0.0, 0.5, 1.0]
        args.num_trials = 1
    return args


def _run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    train_block = int(model.config.block_size)

    if args.rope_method != "none":
        retune_rope(model, args.rope_method, args.rope_factor, args.max_context_length, train_block)

    # Lazy import: keeps the guard testable and avoids a hard dependency at
    # collection time (the entry reports a clear error when tiktoken is absent).
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    evaluator = NeedleEvaluator(model, tokenizer, max_context_length=args.max_context_length)
    result = evaluator.evaluate(
        depths=args.depths,
        context_lengths=args.context_lengths,
        num_trials=args.num_trials,
        max_new_tokens=args.max_new_tokens,
    )
    payload = {
        "checkpoint": args.checkpoint,
        "rope_method": args.rope_method,
        "rope_factor": args.rope_factor,
        "grid_params": {
            "context_lengths": args.context_lengths,
            "depths": args.depths,
            "num_trials": args.num_trials,
            "max_new_tokens": args.max_new_tokens,
            "max_context_length": args.max_context_length,
        },
        "summary": result["summary"],
        "scores": result["scores"],
        "raw": result["raw"],
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
    try:
        _run(args)
    except Exception as exc:  # noqa: BLE001 - tool entry: report, exit 1, no partial JSON
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
