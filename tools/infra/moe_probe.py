"""E-C-6: pure-CPU MoE routing/capacity probe (capacity sweep, load stats, skew).

Three experiments over the real ``model.modern_gpt.SwiGLU`` MoE forward
(top-1 routing + capacity drop, Switch-Transformer semantics):

1. capacity_sweep  -- capacity factor in {0.5, 1.0, 1.25}: dropped-token rate
   via the mirror formula (same semantics as the forward: counts=bincount,
   capacity=int(cf*N/E), stable-sort keeps the first ``capacity`` per expert)
   plus a real-forward cross-check that every dropped token's output row is
   exactly zero.
2. load_distribution -- per-expert token fractions, load entropy, max/min
   ratio of an unbiased router.
3. skew_injection -- router.weight[:, 0] scaled by k in {1, 2, 4} at cf=1.0:
   overflow rate grows with k (load imbalance amplifies capacity overflow).

Source is synthetic by default; ``--checkpoint`` optionally loads a real
router weight (any ``*.router.weight`` entry with a compatible shape).
Guard semantics: missing/incompatible checkpoint -> skip that source, record
a warning, still exit 0 (this is a probe, not an experiment entry).

Exit codes: 0 success (incl. guard degradation) / 1 run error or interrupt
(atomic tmp+rename: no partial JSON, no .tmp leftover) / 2 invalid args.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.modern_gpt import ModernGPTConfig, SwiGLU  # noqa: E402

NUM_EXPERTS = 8
N_EMBD = 32
FULL_CAPACITY_FACTORS = [0.5, 1.0, 1.25]
QUICK_CAPACITY_FACTORS = [0.5, 1.0]
FULL_SKEW_KS = [1, 2, 4]
QUICK_SKEW_KS = [1, 2]
DEFAULT_OUTPUT_JSON = os.path.join("docs", "results", "moe_probe.json")


def dispatch_stats(x_flat, router_logits, capacity_factor, num_experts):
    """Mirror of the forward dispatch (model/modern_gpt.py L1064/L1070-1074).

    Returns the top-1 selection, per-expert counts, truncating capacity,
    dropped-token count/rate, and a keep mask in ORIGINAL token order (the
    forward's internal ``keep`` is in sorted order; scatter it back through
    ``sort_idx``).  Pure function: no state, no side effects.
    """
    n = x_flat.shape[0]
    probs = torch.softmax(router_logits, dim=-1)
    _, selected = torch.topk(probs, 1, dim=-1)
    selected = selected.squeeze(-1)
    counts = torch.bincount(selected, minlength=num_experts)
    capacity = int(capacity_factor * n / num_experts)
    dropped = int(torch.clamp(counts - capacity, min=0).sum())
    # Stable sort groups by expert preserving token order; slots < capacity
    # keep exactly the first `capacity` tokens per expert (forward semantics).
    sorted_experts, sort_idx = torch.sort(selected, stable=True)
    seg_starts = torch.cumsum(counts, 0) - counts
    slots = torch.arange(n) - seg_starts[sorted_experts]
    keep = slots < capacity
    keep_mask = torch.zeros(n, dtype=torch.bool)
    keep_mask[sort_idx] = keep
    return {
        "selected": selected,
        "counts": counts,
        "capacity": capacity,
        "dropped": dropped,
        "drop_rate": dropped / n,
        "keep_mask": keep_mask,
    }


def exp_capacity_sweep(moe: SwiGLU, x: torch.Tensor, factors: list[float]) -> dict:
    """Experiment 1: capacity factor sweep with forward cross-check."""
    x_flat = x.reshape(-1, x.shape[-1])
    router_logits = moe.router(x_flat)
    sweep = []
    for cf in factors:
        stats = dispatch_stats(x_flat, router_logits, cf, moe.num_experts)
        # Real-forward cross-check: dropped tokens' output rows are exactly
        # zero (out_flat is only assigned for kept tokens).  dropout=0 + eval.
        moe.moe_capacity_factor = cf
        out, _ = moe(x)
        out_flat = out.reshape(x_flat.shape)
        dropped_rows_zero = not bool(out_flat[~stats["keep_mask"]].any())
        sweep.append(
            {
                "cf": float(cf),
                "capacity": stats["capacity"],
                "dropped": stats["dropped"],
                "drop_rate": float(stats["drop_rate"]),
                "dropped_rows_zero": bool(dropped_rows_zero),
            }
        )
    return {"sweep": sweep}


def exp_load_distribution(moe: SwiGLU, x: torch.Tensor) -> dict:
    """Experiment 2: unbiased-router per-expert load stats."""
    x_flat = x.reshape(-1, x.shape[-1])
    router_logits = moe.router(x_flat)
    stats = dispatch_stats(x_flat, router_logits, 1.0, moe.num_experts)
    counts = stats["counts"].float()
    fractions = counts / counts.sum()
    nonzero = fractions[fractions > 0]
    entropy = float(-(nonzero * nonzero.log()).sum())
    min_f = float(fractions.min())
    max_min_ratio = float(fractions.max() / min_f) if min_f > 0 else float("inf")
    return {
        "fractions": [float(f) for f in fractions],
        "entropy": entropy,
        "max_min_ratio": max_min_ratio,
    }


def exp_skew_injection(moe: SwiGLU, x: torch.Tensor, skew_ks: list[int]) -> dict:
    """Experiment 3: bias the router toward expert 0 and watch cf=1.0 overflow."""
    x_flat = x.reshape(-1, x.shape[-1])
    base_weight = moe.router.weight.detach().clone()
    curve = []
    try:
        for k in skew_ks:
            with torch.no_grad():
                moe.router.weight.copy_(base_weight)
                moe.router.weight[:, 0] *= k
            router_logits = moe.router(x_flat)
            stats = dispatch_stats(x_flat, router_logits, 1.0, moe.num_experts)
            curve.append({"k": int(k), "overflow_rate": float(stats["drop_rate"])})
    finally:
        with torch.no_grad():
            moe.router.weight.copy_(base_weight)
    return {"curve": curve}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", dest="output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--quick", action="store_true", help="shrink grid: cf {0.5, 1.0}, T=64, skew k {1, 2}"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="optional checkpoint with a compatible router.weight"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> dict:
    factors = QUICK_CAPACITY_FACTORS if args.quick else FULL_CAPACITY_FACTORS
    skew_ks = QUICK_SKEW_KS if args.quick else FULL_SKEW_KS
    seq = 64 if args.quick else 128

    # RNG consumption order is part of the pinned-seed contract
    # (manual_seed -> SwiGLU init -> randn input; see tests/test_moe_probe.py).
    torch.manual_seed(args.seed)
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=N_EMBD,
        block_size=64,
        num_experts=NUM_EXPERTS,
        dropout=0.0,
    )
    moe = SwiGLU(cfg).eval()
    x = torch.randn(2, seq, N_EMBD)

    warnings: list[str] = []
    source = "synthetic"
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            warnings.append(f"skipped checkpoint source: file not found ({args.checkpoint})")
        else:
            try:
                ckpt = torch.load(args.checkpoint, map_location="cpu")
                sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else None
                router_w = None
                if isinstance(sd, dict):
                    keys = [k for k in sd if k.endswith("router.weight")]
                    if keys and tuple(sd[keys[0]].shape) == (NUM_EXPERTS, N_EMBD):
                        router_w = sd[keys[0]]
                if router_w is not None:
                    with torch.no_grad():
                        moe.router.weight.copy_(router_w)
                    source = "checkpoint"
                else:
                    warnings.append(
                        f"skipped checkpoint source: no compatible router.weight "
                        f"({args.checkpoint})"
                    )
            except Exception as exc:  # noqa: BLE001 - guard degradation
                warnings.append(f"skipped checkpoint source: {type(exc).__name__}: {exc}")

    experiments = [
        {
            "name": "capacity_sweep",
            "source": source,
            "metrics": exp_capacity_sweep(moe, x, factors),
        },
        {
            "name": "load_distribution",
            "source": source,
            "metrics": exp_load_distribution(moe, x),
        },
        {
            "name": "skew_injection",
            "source": source,
            "metrics": exp_skew_injection(moe, x, skew_ks),
        },
    ]
    return {
        "meta": {
            "tool": "moe_probe",
            "timestamp": datetime.now().isoformat(),
            "seed": int(args.seed),
            "torch_version": torch.__version__,
        },
        "config": {
            "E": NUM_EXPERTS,
            "T_total": int(x.numel() // N_EMBD),
            "capacity_factors": [float(f) for f in factors],
            "hidden": N_EMBD,
            "source": source,
        },
        "experiments": experiments,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = _run(args)
    except Exception as exc:  # noqa: BLE001 - tool entry: report, exit 1
        print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    tmp = args.output_json + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, args.output_json)
    print("wrote", args.output_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
