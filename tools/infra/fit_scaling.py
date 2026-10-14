"""Pure-CPU scaling-law fitter and budget calculator (AR-v0.7.5 FR4, task T005).

Deep-dive companion for docs/ai-infra/scaling-laws.md (topic 9). Uses ONLY
existing repo data (ablation-summary.json + budget.json); never retrains.

Key honesty constraint (design D3): the 7 repo ablation groups all trained on
the same budget, so D is constant and the Chinchilla joint term B/D^beta is
unidentifiable. The primary fit is therefore Kaplan-style L(N)=E+A/N^alpha;
the joint form is demonstrated on synthetic data only.

CLI:  python tools/infra/fit_scaling.py [--quick] [--output-json PATH]
                                          [--data PATH] [--budget PATH]
Exit: 0 success (incl. honest non-convergence) / 1 data source missing /
      2 usage error (argparse, --output-json parent missing).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "results" / "scaling_fit.json"
DEFAULT_DATA = REPO_ROOT / "docs" / "results" / "ablation-summary.json"
DEFAULT_BUDGET = REPO_ROOT / "docs" / "results" / "budget.json"

# Nominal repo model size (50M-class; modern-full measures 59.56M -- see
# scaling-laws.md section 6.2 for the two tokens/param readings).
REPO_PARAMS_M = 50
# Hoffmann et al. compute-optimal anchors (literature defaults, cf. section 3).
LIT_ALPHA_N, LIT_ALPHA_D, LIT_D_OVER_N = 0.34, 0.28, 20.0
# Repo baseline training throughput (ablation-summary baseline mean tokens/s,
# E-D measured; used to translate token budgets into wall-clock hours).
REPO_THROUGHPUT_TOK_PER_S = 179504.49


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaling-law fit + compute-optimal budget calculator "
        "(scaling-laws.md companion). Existing data only; never retrains."
    )
    parser.add_argument("--quick", action="store_true", help="trim extrapolation grid")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="ablation-summary JSON")
    parser.add_argument("--budget", default=str(DEFAULT_BUDGET), help="budget JSON")
    return parser.parse_args(argv)


def _load_groups(path: Path) -> list[tuple[float, float]]:
    """Usable (params_m, best_val_loss) points: completed groups with a loss."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    points: list[tuple[float, float]] = []
    for group in payload.get("groups", []):
        if group.get("status") != "completed":
            continue
        params_m = group.get("params_m")
        loss = group.get("best_val_loss")
        if params_m is None or loss is None:
            continue
        points.append((float(params_m), float(loss)))
    return points


def _kaplan_residuals(p: np.ndarray, n: np.ndarray, loss: np.ndarray) -> np.ndarray:
    e, log_a, alpha = p
    return e + np.exp(log_a) / n**alpha - loss


def fit_kaplan(points: list[tuple[float, float]]) -> dict:
    """Multi-start least_squares fit of L(N) = E + A/N^alpha (log-space A)."""
    params_m = [p for p, _ in points]
    val_loss = [v for _, v in points]
    n = np.asarray(params_m, dtype=float) * 1e6
    loss = np.asarray(val_loss, dtype=float)

    fitted: list[float | None] | None = None
    converged = False
    r_squared = 0.0
    result_E = result_A = result_alpha = None
    stds: tuple[float | None, float | None, float | None] = (None, None, None)
    note = ""
    degenerate: list[str] = []

    if len(points) < 3:
        degenerate.append(f"insufficient usable points (n={len(points)} < 3)")
    elif float(np.std(loss)) < 1e-12:
        degenerate.append("zero-variance target (all losses identical)")

    if not degenerate:
        best = None
        for alpha0 in (0.05, 0.1, 0.2, 0.34):
            for log_a0 in (2.0, 3.0, 5.0):
                try:
                    res = least_squares(
                        _kaplan_residuals,
                        x0=[float(loss.min()) * 0.95, log_a0, alpha0],
                        args=(n, loss),
                        bounds=([0.0, -10.0, 1e-4], [np.inf, 30.0, 2.0]),
                        max_nfev=20000,
                    )
                except (ValueError, FloatingPointError):
                    continue
                if best is None or res.cost < best.cost:
                    best = res
        if best is not None and best.success:
            converged = True
            result_E = float(best.x[0])
            result_A = float(np.exp(best.x[1]))
            result_alpha = float(best.x[2])
            ss_res = float(np.sum(best.fun**2))
            ss_tot = float(np.sum((loss - loss.mean()) ** 2))
            r_squared = 1.0 - ss_res / ss_tot
            fitted = [float(v) for v in (result_E + result_A / n**result_alpha)]
            dof = len(n) - 3
            try:
                cov = np.linalg.pinv(best.jac.T @ best.jac) * (ss_res / dof)
                stds = (
                    float(np.sqrt(cov[2, 2])),  # alpha std
                    float(result_A * np.sqrt(cov[1, 1])),  # A std (log-space mapped)
                    float(np.sqrt(cov[0, 0])),  # E std
                )
            except (np.linalg.LinAlgError, ValueError):
                stds = (None, None, None)
        else:
            degenerate.append("multi-start least_squares did not converge")
    else:
        note = "; ".join(degenerate)
        if not note:
            note = "degenerate fit input"

    if fitted is None:
        fitted = [None] * len(points)

    return {
        "converged": converged,
        "r_squared": r_squared,
        "E": result_E,
        "A": result_A,
        "alpha": result_alpha,
        "alpha_std": stds[0],
        "A_std": stds[1],
        "E_std": stds[2],
        "params_m": params_m,
        "val_loss": val_loss,
        "fitted": fitted,
        "degenerate_notes": degenerate or note,
    }


def chinchilla_synthetic_demo() -> dict:
    """Joint-form L(N,D)=E+A/N^aN+B/D^aD demonstrated on synthetic data.

    The repo's real data has constant D (same budget across groups), so the
    joint term is unidentifiable there; this demo validates the fitter on a
    controlled (N, D) grid instead (design D3, section 6.3 of the manual).
    """
    n_grid = np.logspace(7, 9, 5)
    d_grid = np.logspace(9, 11, 5)
    n_axis, d_axis = np.meshgrid(n_grid, d_grid, indexing="ij")
    nn: np.ndarray = n_axis.ravel()
    dd: np.ndarray = d_axis.ravel()
    true = {"E": 3.0, "A": 1e3, "alpha_n": 0.34, "B": 1e3, "alpha_d": 0.28}
    rng = np.random.default_rng(1337)
    ll = true["E"] + true["A"] / nn ** true["alpha_n"] + true["B"] / dd ** true["alpha_d"]
    ll = ll * (1 + rng.normal(0, 0.002, nn.size))

    def residuals(p: np.ndarray) -> np.ndarray:
        e, log_a, a_n, log_b, a_d = p
        return e + np.exp(log_a) / nn**a_n + np.exp(log_b) / dd**a_d - ll

    best = None
    for a_n0 in (0.2, 0.34, 0.5):
        for a_d0 in (0.2, 0.28, 0.4):
            try:
                res = least_squares(
                    residuals,
                    x0=[float(ll.min()) * 0.9, 4.0, a_n0, 4.0, a_d0],
                    bounds=([0.0, -10.0, 1e-4, -10.0, 1e-4], [np.inf, 30.0, 2.0, 30.0, 2.0]),
                    max_nfev=40000,
                )
            except (ValueError, FloatingPointError):
                continue
            if best is None or res.cost < best.cost:
                best = res
    if best is None:  # pragma: no cover - scipy failure path
        recovered = dict.fromkeys(true, None)
    else:
        recovered = {
            "E": float(best.x[0]),
            "A": float(np.exp(best.x[1])),
            "alpha_n": float(best.x[2]),
            "B": float(np.exp(best.x[3])),
            "alpha_d": float(best.x[4]),
        }
    return {
        "true_params": true,
        "recovered_params": recovered,
        "grid": {"n_points": int(nn.size), "noise": 0.002, "seed": 1337},
    }


def build_predictions(fit: dict, quick: bool) -> list[dict]:
    """Teaching-grade extrapolations with honest positive-width intervals."""
    scales = (10.0, 100.0) if quick else (10.0, 50.0, 100.0, 1000.0)
    base_n = REPO_PARAMS_M * 1e6

    alpha = fit["alpha"]
    valid = (
        fit["converged"]
        and fit["A"] is not None
        and alpha is not None
        and fit["A"] > 0
        and alpha > 0
        and math.isfinite(fit["A"])
        and math.isfinite(alpha)
    )
    losses = [v for v in fit["val_loss"] if v is not None]
    e_anchor = (
        fit["E"] if valid and fit["E"] is not None else (min(losses) * 0.95 if losses else 3.4)
    )
    if valid:
        a_lit = fit["A"]
    else:
        # Literature-anchored fallback (section 3): A chosen so the anchor
        # reproduces the observed loss scale at the repo's 50M size.
        alpha = LIT_ALPHA_N
        a_lit = ((np.mean(losses) if losses else 3.6) - e_anchor) * base_n**LIT_ALPHA_N

    predictions = []
    for scale in scales:
        n_star = base_n * scale
        d_star = LIT_D_OVER_N * n_star
        pred = float(e_anchor + float(a_lit) / n_star**alpha)
        predictions.append(
            {
                "name": f"{scale:g}x-params-chinchilla-20to1",
                "N": float(n_star),
                "D": float(d_star),
                "pred_loss": pred,
                "interval": [pred * 0.9, pred * 1.15],
                "method": "kaplan-fit" if valid else "literature-anchored-fallback",
            }
        )
    return predictions


def build_budget_calculator(budget: dict) -> dict:
    """Compute-optimal cases under the Hoffmann anchors (6ND, D/N=20)."""
    train_tokens = float(budget.get("train_tokens", 0)) or float(CORPUS_FALLBACK)
    tokens_per_iter = float(budget.get("tokens_per_iter", 32768))
    repo_flops = 6 * REPO_PARAMS_M * 1e6 * train_tokens

    cases = []
    for label, mult in (("repo-budget", 1.0), ("10x-budget", 10.0), ("100x-budget", 100.0)):
        flops = repo_flops * mult
        n_star = math.sqrt(flops / (6 * LIT_D_OVER_N))
        d_star = LIT_D_OVER_N * n_star
        iters = d_star / tokens_per_iter
        hours = d_star / REPO_THROUGHPUT_TOK_PER_S / 3600
        cases.append(
            {
                "name": label,
                "flops_budget": float(flops),
                "N_star": float(n_star),
                "D_star": float(d_star),
                "D_over_N": LIT_D_OVER_N,
                "train_iters_at_repo_throughput": float(iters),
                "train_hours_at_repo_throughput": float(hours),
                "tokens_vs_repo_corpus": float(d_star / train_tokens) if train_tokens else None,
            }
        )
    return {
        "cases": cases,
        "flops_per_token_formula": "6*N*D (per-token cost ~ 6N; cf. utils/infra_math.py)",
        "literature_defaults": {
            "alpha_n": LIT_ALPHA_N,
            "alpha_d": LIT_ALPHA_D,
            "d_over_n": LIT_D_OVER_N,
        },
        "throughput_anchor": {
            "tokens_per_sec": REPO_THROUGHPUT_TOK_PER_S,
            "source": "docs/results/ablation-summary.json baseline mean_tokens_per_sec",
        },
    }


CORPUS_FALLBACK = 1_244_464_437.0


def build_repo_budget_review(budget: dict) -> dict:
    train_tokens = int(budget.get("train_tokens", 0))
    max_iters = budget.get("max_iters")
    tokens_per_iter = budget.get("tokens_per_iter")
    consumed = (
        int(max_iters) * int(tokens_per_iter)
        if max_iters is not None and tokens_per_iter is not None
        else None
    )
    tokens_per_param = train_tokens / (REPO_PARAMS_M * 1e6) if train_tokens else None
    verdict = (
        "教学预算：语料配比（corpus 口径 tokens/param≈24.9）略偏数据侧；实际消耗不足一纪元，"
        "consumed 口径≈9.9 t/p 属欠训练——见 scaling-laws.md §6.2 两口径诚实解读"
        if tokens_per_param
        else "budget.json 缺 train_tokens，无法对照"
    )
    return {
        "train_tokens": train_tokens,
        "params_m": REPO_PARAMS_M,
        "tokens_per_param": tokens_per_param,
        "chinchilla_ref": LIT_D_OVER_N,
        "verdict": verdict,
        "bookkeeping": {
            "corpus_tokens": train_tokens,
            "consumed_tokens": consumed,
            "note": "corpus=一纪元全量；consumed=max_iters×tokens_per_iter 实际喂入；"
            "one_epoch_iters 对应 train split（≈95%）——scaling-laws.md §6.1 记账三件套",
        },
    }


def _summary_md(meta: dict, fit_quality: dict, review: dict) -> str:
    lines = [f"# fit_scaling (quick={meta['quick']}, device=cpu)", ""]
    lines.append(
        f"- fit: converged={fit_quality['converged']}, "
        f"R2={fit_quality['r_squared']:.4f}, n={fit_quality['n_points']}"
    )
    if fit_quality["converged"]:
        lines.append(
            f"- Kaplan params: alpha={fit_quality['kaplan_alpha']:.4f} "
            f"(+/-{fit_quality['param_uncertainty']['alpha_std']:.4f})"
        )
    tpp = review["tokens_per_param"]
    if tpp:
        lines.append(
            f"- repo tokens/param: {tpp:.1f} vs Chinchilla 20 "
            f"(consumed {review['bookkeeping']['consumed_tokens']:,} tokens)"
        )
    lines.append("- identifiability: D constant across groups; Kaplan-style primary fit")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    data_path = Path(args.data)
    budget_path = Path(args.budget)
    output = Path(args.output_json)

    for probe, label in ((data_path, "--data"), (budget_path, "--budget")):
        if not probe.is_file():
            print(f"error: {label} data source not found: {probe}", file=sys.stderr)
            return 1
    if not output.parent.is_dir():
        print(
            f"error: --output-json parent directory does not exist: {output.parent}",
            file=sys.stderr,
        )
        return 2

    try:
        points = _load_groups(data_path)
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        fit = fit_kaplan(points)
        demo = chinchilla_synthetic_demo()
        predictions = build_predictions(fit, args.quick)
        calculator = build_budget_calculator(budget)
        review = build_repo_budget_review(budget)

        limitations = [
            "7 点非受控扫参（架构消融混入 N 变化，无跨架构外推力）",
            "教学拟合而非 scaling study（样本点少，参数不确定度大）",
        ]
        if not fit["converged"]:
            limitations.append(f"拟合未收敛/退化: {fit['degenerate_notes']}")

        fit_quality = {
            "n_points": len(points),
            "method": "scipy.optimize.least_squares(multi-start, log-space A)",
            "identifiability_note": (
                "D 恒定（7 组同预算训练），B/D^beta 不可辨识，主拟合为 Kaplan 式 L(N)"
            ),
            "converged": fit["converged"],
            "r_squared": fit["r_squared"],
            "kaplan_alpha": fit["alpha"],
            "param_uncertainty": {
                "alpha_std": fit["alpha_std"],
                "A_std": fit["A_std"],
                "E_std": fit["E_std"],
                "method": "least_squares Jacobian covariance (pinv(J^T J) * s^2/dof)",
            },
            "limitations": limitations,
        }
        kaplan_fit = {
            "E": fit["E"],
            "A": fit["A"],
            "alpha": fit["alpha"],
            "r_squared": fit["r_squared"],
            "params_m": fit["params_m"],
            "val_loss": fit["val_loss"],
            "fitted": fit["fitted"],
        }
        meta = {
            "tool": "fit_scaling",
            "device": "cpu",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quick": bool(args.quick),
            "numpy": np.__version__,
            "data_sources": [str(data_path.resolve()), str(budget_path.resolve())],
        }
        payload = {
            "meta": meta,
            "fit_quality": fit_quality,
            "kaplan_fit": kaplan_fit,
            "chinchilla_synthetic_demo": demo,
            "predictions": predictions,
            "budget_calculator": calculator,
            "repo_budget_review": review,
        }
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, output)
    except Exception as exc:  # noqa: BLE001 -- single exit-1 funnel, no partial JSON
        print(f"error: fit interrupted: {exc}", file=sys.stderr)
        return 1

    print(_summary_md(meta, fit_quality, review))
    return 0


if __name__ == "__main__":
    sys.exit(main())
