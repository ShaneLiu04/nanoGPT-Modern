"""Pure-CPU numerical precision probe (AR-v0.7.5 FR2, task T004).

Deep-dive companion for docs/ai-infra/numerical-precision.md (topic 8).
Runs five teaching experiments on dtype representation semantics:

1. dtype_limits          -- torch.finfo field-by-field recompute (ulp at 1.0)
2. rne_rounding          -- IEEE 754 round-to-nearest-even tie cases (fp16)
3. relative_error_grid   -- element-wise relative error of bf16 vs fp16 casts
4. accumulation_precision-- fp32 vs bf16 accumulators over a K-dim matmul
5. loss_scaling_simulation-- fp16 tiny-gradient underflow + scale rescue

CLI:  python tools/infra/precision_probe.py [--quick] [--output-json PATH]
Exit: 0 success / 1 interrupted (no partial JSON) / 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "results" / "precision_probe.json"
TRITON_REFS = REPO_ROOT / "docs" / "results" / "triton_kernels.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure-CPU numerical precision probe (numerical-precision.md companion)."
    )
    parser.add_argument("--quick", action="store_true", help="shrink sample sizes (<=30s)")
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT),
        help=f"output JSON path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args(argv)


def _tolerance_refs() -> dict:
    """Pin the bf16 atol/rtol baseline from triton_kernels.json (fallback 0.01)."""
    atol = rtol = 0.01
    try:
        refs = json.loads(TRITON_REFS.read_text(encoding="utf-8"))
        for row in refs.get("consistency", []):
            if row.get("dtype") == "bfloat16":
                atol = row["atol"]
                rtol = row["rtol"]
                break
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return {"bf16_atol": atol, "bf16_rtol": rtol, "source": "docs/results/triton_kernels.json"}


def exp_dtype_limits() -> dict:
    """torch.finfo recompute per dtype, plus ulp at 1.0 via nextafter."""
    dtypes: dict[str, dict] = {}
    for name, torch_dtype in (
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ):
        info = torch.finfo(torch_dtype)
        one = torch.tensor(1.0, dtype=torch_dtype)
        two = torch.tensor(2.0, dtype=torch_dtype)
        ulp_at_1 = (torch.nextafter(one, two) - one).item()
        dtypes[name] = {
            "max": info.max,
            "min": info.min,
            "eps": info.eps,
            "smallest_normal": info.smallest_normal,
            "ulp_at_1": ulp_at_1,
        }
    return {"name": "dtype_limits", "dtypes": dtypes}


def exp_rne_rounding() -> dict:
    """Constructed tie cases: fp16 binade [2048, 4096) has spacing 2, so odd
    integers in it sit exactly on rounding ties and must go to even."""
    cases = []
    hits = 0
    for value in (2049.0, 2051.0, 2053.0, 2055.0, 2057.0, 2059.0):
        rounded = float(torch.tensor(value, dtype=torch.float16))
        # Nearest even neighbours around the tie point.
        lower = value - 1.0
        upper = value + 1.0
        expect = lower if lower % 4 == 0 else upper
        cases.append({"value": value, "rounded": rounded, "expected_even": expect})
        hits += int(rounded == expect)
    return {"name": "rne_rounding", "cases": cases, "tie_to_even_hits": hits / len(cases)}


def exp_relative_error_grid(quick: bool) -> dict:
    """Element-wise relative error of casting fp32 samples to bf16 vs fp16.

    bf16 has 7 mantissa bits vs fp16's 10, so bf16 error must dominate.
    """
    n = 2_000 if quick else 100_000
    gen = torch.Generator().manual_seed(1337)
    x = torch.rand(n, generator=gen) * 9.0 + 1.0  # values in [1, 10): no fp16 range risk
    out: dict = {"name": "relative_error_grid"}
    for label, torch_dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        rounded = x.to(torch_dtype).to(torch.float32)
        rel = (rounded - x).abs() / x.abs()
        out[f"fp32_vs_{label}"] = {
            "max": rel.max().item(),
            "mean": rel.mean().item(),
            "p99": torch.quantile(rel, 0.99).item(),
        }
    return out


def _accum_rel_err(a: torch.Tensor, b: torch.Tensor, ref: torch.Tensor, chunk: int) -> float:
    acc = torch.zeros(a.shape[0], b.shape[1], dtype=torch.bfloat16)
    for start in range(0, a.shape[1], chunk):
        a_blk = a[:, start : start + chunk].to(torch.float32)
        b_blk = b[start : start + chunk, :].to(torch.float32)
        acc = (acc.to(torch.float32) + a_blk @ b_blk).to(torch.bfloat16)
    return ((acc.to(torch.float64) - ref).norm() / ref.norm()).item()


def exp_accumulation_precision(quick: bool) -> dict:
    """Same bf16-rounded inputs, fp32 accumulator vs bf16 accumulator.

    The fp32 path multiplies in fp32 with an fp32 running sum; the bf16 path
    re-rounds the running sum to bf16 after every chunk (the "big eats small"
    ulp trap from the manual, section 4.2).
    """
    k_dim = 1_024 if quick else 4_096
    m = n = 64 if quick else 128
    gen = torch.Generator().manual_seed(1337)
    a32 = torch.randn(m, k_dim, generator=gen)
    b32 = torch.randn(k_dim, n, generator=gen)
    # Fixed bf16-rounded inputs: both accumulation paths see identical data.
    a = a32.to(torch.bfloat16)
    b = b32.to(torch.bfloat16)

    ref = a.to(torch.float64) @ b.to(torch.float64)
    fp32_path = a.to(torch.float32) @ b.to(torch.float32)
    fp32_err = ((fp32_path.to(torch.float64) - ref).norm() / ref.norm()).item()
    bf16_err = _accum_rel_err(a, b, ref, chunk=32)

    return {
        "name": "accumulation_precision",
        "k_dim": k_dim,
        "chunk": 32,
        "fp32_accum_rel_err": fp32_err,
        "bf16_accum_rel_err": bf16_err,
    }


def exp_loss_scaling_simulation() -> dict:
    """GradScaler semantics in miniature (manual section 3.4):

    - tiny gradients (1e-10..1e-6) mostly flush to zero in fp16;
    - scaling by 2**16 lifts them into the representable range;
    - a few large gradients (10.0) overflow fp16 *when scaled*, so the
      unscale-then-check-inf step detects them (the infskip / backoff story).
    """
    gen = torch.Generator().manual_seed(1337)
    n = 2_000
    tiny = 10.0 ** (torch.rand(n, generator=gen) * (-4.0) + (-10.0))  # [1e-10, 1e-6]
    large = torch.full((8,), 10.0)
    g = torch.cat([tiny, large])

    scale = 65536.0  # 2**16, GradScaler's initial scale
    naive = g.to(torch.float16)
    naive_zero_fraction = (naive == 0).to(torch.float32).mean().item()

    scaled = (g * scale).to(torch.float16)
    recovered_nonzero = (scaled != 0).to(torch.float32).mean().item()

    unscaled = scaled.to(torch.float32) / scale
    within_tol = ((unscaled - g).abs() <= 0.05 * g.abs() + 1e-12).to(torch.float32).mean().item()
    scaled_recover_fraction = min(recovered_nonzero, within_tol)

    unscaled_overflow_detected = bool(torch.isinf(scaled).any().item())

    return {
        "name": "loss_scaling_simulation",
        "grad_scale": scale,
        "naive_fp16_zero_fraction": naive_zero_fraction,
        "scaled_recover_fraction": scaled_recover_fraction,
        "unscaled_overflow_detected": unscaled_overflow_detected,
        "n_gradients": int(g.numel()),
    }


def run_experiments(quick: bool) -> list[dict]:
    return [
        exp_dtype_limits(),
        exp_rne_rounding(),
        exp_relative_error_grid(quick),
        exp_accumulation_precision(quick),
        exp_loss_scaling_simulation(),
    ]


def _summary_md(meta: dict, experiments: list[dict]) -> str:
    lines = [f"# precision_probe (quick={meta['quick']}, device=cpu)", ""]
    lines.append("| experiment | key stat |")
    lines.append("| --- | --- |")
    for exp in experiments:
        if exp["name"] == "dtype_limits":
            stat = f"bf16 ulp@1 = {exp['dtypes']['bfloat16']['ulp_at_1']:.3e}"
        elif exp["name"] == "rne_rounding":
            stat = f"tie_to_even_hits = {exp['tie_to_even_hits']}"
        elif exp["name"] == "relative_error_grid":
            stat = f"bf16 max rel err = {exp['fp32_vs_bf16']['max']:.3e}"
        elif exp["name"] == "accumulation_precision":
            stat = (
                f"fp32 {exp['fp32_accum_rel_err']:.2e} vs bf16 "
                f"{exp['bf16_accum_rel_err']:.2e} (K={exp['k_dim']})"
            )
        else:
            stat = (
                f"naive zero {exp['naive_fp16_zero_fraction']:.0%} -> "
                f"scaled recover {exp['scaled_recover_fraction']:.0%}"
            )
        lines.append(f"| {exp['name']} | {stat} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = Path(args.output_json)

    if not output.parent.is_dir():
        print(
            f"error: --output-json parent directory does not exist: {output.parent}",
            file=sys.stderr,
        )
        return 2

    # Preflight the atomic-write target: an unwritable location (e.g. a
    # read-only parent dir on POSIX) is a usage error, not a mid-run
    # interrupt -- fail fast with exit 2 before any measurement starts.
    probe = output.with_suffix(output.suffix + ".tmp")
    try:
        probe.touch()
        probe.unlink()
    except OSError as exc:
        print(
            f"error: --output-json is not writable: {output} ({exc})",
            file=sys.stderr,
        )
        return 2

    try:
        experiments = run_experiments(args.quick)
        meta: dict = {
            "tool": "precision_probe",
            "device": "cpu",
            "torch": torch.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quick": bool(args.quick),
            "tolerance_refs": _tolerance_refs(),
        }
        payload = {"meta": meta, "experiments": experiments}
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, output)
    except Exception as exc:  # noqa: BLE001 -- single exit-1 funnel, no partial JSON
        print(f"error: measurement interrupted: {exc}", file=sys.stderr)
        return 1

    print(_summary_md(meta, experiments))
    return 0


if __name__ == "__main__":
    sys.exit(main())
