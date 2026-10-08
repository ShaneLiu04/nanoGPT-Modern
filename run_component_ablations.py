"""Component ablation orchestrator (FR4 / AR-v0.4.0-experiments).

Runs the fixed 7-group component ablation matrix as subprocess trainings of
``training.train_pretrain`` (one out_dir per group), resuming around groups
whose T001 experiment manifest already reports ``status == "completed"``,
then aggregates per-group experiment summaries (val loss / throughput /
memory / wall time / parameter counts) into JSON + Markdown artifacts under
``{out_root}/experiments/``.

Usage::

    python run_component_ablations.py --stage all --out_root out

Groups can be subsetted (``--groups baseline,moe``) and training length
capped (``--max_iters 100``) for smoke runs; ``--fail_fast`` stops the
train stage at the first failing group. Exit code: 0 on success, 1 when
any group failed, 2 on invalid arguments.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from model.modern_gpt import ModernGPT, ModernGPTConfig
from utils.config import load_yaml_config
from utils.experiment import summarize_experiment

MANIFEST_FILENAME = "manifest.json"

# Fixed 7-group component ablation matrix (design.md §6.1 / srs FR4).
# Insertion order == execution order.
ABLATION_GROUPS: Dict[str, List[str]] = {
    "baseline": ["--model", "baseline"],
    "modern-full": ["--model", "modern", "--n_kv_head", "2"],
    "no-rope": [
        "--model", "modern", "--n_kv_head", "2",
        "--position_encoding", "learned",
    ],
    "no-swiglu": [
        "--model", "modern", "--n_kv_head", "2",
        "--ffn_variant", "gelu",
    ],
    "mha": ["--model", "modern", "--n_kv_head", "8"],
    "moe": ["--model", "modern", "--n_kv_head", "2", "--num_experts", "4"],
    "qk-norm": ["--model", "modern", "--n_kv_head", "2", "--qk_norm"],
}
GROUP_ORDER: List[str] = list(ABLATION_GROUPS.keys())

_INT_FLAG_KEYS = {"n_kv_head", "num_experts"}
_BOOL_FLAGS = {"qk_norm"}


def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed 7-group component ablation matrix."
    )
    parser.add_argument(
        "--stage",
        choices=["train", "summarize", "all"],
        default="all",
        help="train = only run trainings; summarize = only aggregate; "
        "all = train then summarize (default).",
    )
    parser.add_argument(
        "--groups",
        default="all",
        help="Comma-separated subset of groups, or 'all' (default). "
        "Execution always follows the fixed matrix order.",
    )
    parser.add_argument(
        "--config",
        default="config/pretrain.yaml",
        help="Base training config (group flags override it).",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=None,
        help="Cap training length (omitted from commands when None).",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop the train stage at the first failing group.",
    )
    parser.add_argument(
        "--out_root",
        default="out",
        help="Root directory for per-group out_dirs and summaries.",
    )
    return parser.parse_args(argv)


def select_groups(groups_arg: str) -> List[str]:
    """Expand ``--groups`` ('all' or comma list) into ordered group names."""
    if groups_arg in (None, "", "all"):
        return list(GROUP_ORDER)
    wanted = [g.strip() for g in groups_arg.split(",") if g.strip()]
    unknown = [g for g in wanted if g not in ABLATION_GROUPS]
    if unknown:
        raise ValueError(
            f"Unknown group(s) {unknown}; valid groups: {GROUP_ORDER}"
        )
    return [g for g in GROUP_ORDER if g in wanted]


def group_out_dir(out_root: str, name: str) -> str:
    """Training out_dir for a group: ``{out_root}/ablation-<name>``."""
    return f"{out_root}/ablation-{name}"


def group_exp_dir(out_root: str, name: str) -> str:
    """T001 experiment dir for a group's training run."""
    return f"{group_out_dir(out_root, name)}/experiments/ablation-{name}"


def is_group_completed(exp_dir: str) -> bool:
    """True when the group's manifest reports status == 'completed'."""
    manifest_path = Path(exp_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return manifest.get("status") == "completed"


def build_train_command(
    name: str,
    config: str,
    max_iters: Optional[int] = None,
    out_root: str = "out",
) -> List[str]:
    """Assemble the subprocess command for one group's training run."""
    cmd = [
        sys.executable, "-m", "training.train_pretrain",
        "--config", str(config),
        "--out_dir", group_out_dir(out_root, name),
        "--exp_id", f"ablation-{name}",
        *ABLATION_GROUPS[name],
    ]
    if max_iters is not None:
        cmd += ["--max_iters", str(max_iters)]
    return cmd


def run_train_stage(args: argparse.Namespace) -> Dict[str, Any]:
    """Train every selected, not-yet-completed group as a subprocess.

    Completed groups (manifest status == 'completed') are skipped for
    resume; a failing group is recorded and normally does not stop the
    others (``--fail_fast`` stops after the first failure).
    """
    completed: List[str] = []
    skipped: List[str] = []
    errors: List[Dict[str, Any]] = []
    fail_fast_triggered = False

    for name in select_groups(args.groups):
        exp_dir = group_exp_dir(args.out_root, name)
        if is_group_completed(exp_dir):
            print(f"[skip] {name}: already completed ({exp_dir})")
            skipped.append(name)
            continue
        cmd = build_train_command(
            name, args.config, args.max_iters, out_root=args.out_root
        )
        print(f"[train] {name}: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            completed.append(name)
        else:
            errors.append(
                {
                    "group": name,
                    "returncode": result.returncode,
                    "command": cmd,
                }
            )
            print(
                f"[fail] {name}: exit code {result.returncode}"
            )
            if args.fail_fast:
                fail_fast_triggered = True
                break

    return {
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
        "fail_fast_triggered": fail_fast_triggered,
    }


def _flags_to_overrides(flags: List[str]) -> Dict[str, Any]:
    """Turn ``--key value`` / ``--flag`` CLI tokens into a config overlay."""
    overrides: Dict[str, Any] = {}
    tokens = list(flags)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected ablation flag token: {token!r}")
        key = token[2:]
        if key in _BOOL_FLAGS:
            overrides[key] = True
            i += 1
        else:
            value = tokens[i + 1]
            overrides[key] = int(value) if key in _INT_FLAG_KEYS else value
            i += 2
    return overrides


def _build_baseline_model(cfg: Dict[str, Any]) -> BaselineGPT:
    config = BaselineGPTConfig(
        vocab_size=int(cfg.get("vocab_size", 50257)),
        block_size=int(cfg.get("block_size", 1024)),
        n_layer=int(cfg.get("n_layer", 12)),
        n_head=int(cfg.get("n_head", 8)),
        n_embd=int(cfg.get("n_embd", 512)),
        dropout=float(cfg.get("dropout", 0.0)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
    )
    return BaselineGPT(config)


def _build_modern_model(cfg: Dict[str, Any]) -> ModernGPT:
    config = ModernGPTConfig(
        vocab_size=int(cfg.get("vocab_size", 50257)),
        block_size=int(cfg.get("block_size", 1024)),
        n_layer=int(cfg.get("n_layer", 12)),
        n_head=int(cfg.get("n_head", 8)),
        n_embd=int(cfg.get("n_embd", 512)),
        n_kv_head=int(cfg.get("n_kv_head", 8)),
        dropout=float(cfg.get("dropout", 0.0)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        use_ring_attention=bool(cfg.get("use_ring_attention", False)),
        ring_block_size_q=int(cfg.get("ring_block_size_q", 64)),
        ring_block_size_kv=int(cfg.get("ring_block_size_kv", 64)),
        position_encoding=str(cfg.get("position_encoding", "rope")),
        ffn_variant=str(cfg.get("ffn_variant", "swiglu")),
        qk_norm=bool(cfg.get("qk_norm", False)),
        num_experts=int(cfg.get("num_experts", 1)),
    )
    return ModernGPT(config)


def count_group_params(name: str, config_path: str) -> Dict[str, int]:
    """Parameter counts (total / active) for one group's architecture.

    The group's model is instantiated from the YAML base config overlaid
    with the group flags. Totals include embeddings. For MoE groups the
    active count is the dense (``num_experts=1``) equivalent; for every
    other group active == total.
    """
    cfg: Dict[str, Any] = dict(load_yaml_config(config_path) or {})
    cfg.update(_flags_to_overrides(ABLATION_GROUPS[name]))

    model_type = str(cfg.get("model", "modern"))
    if model_type == "baseline":
        model = _build_baseline_model(cfg)
    else:
        model = _build_modern_model(cfg)
    total = int(model.get_num_params(non_embedding=False))

    active = total
    if model_type != "baseline" and int(cfg.get("num_experts", 1)) > 1:
        dense_cfg = dict(cfg)
        dense_cfg["num_experts"] = 1
        dense_model = _build_modern_model(dense_cfg)
        active = int(dense_model.get_num_params(non_embedding=False))

    return {"total_params": total, "active_params": active}


def _fmt(value: Any, spec: str = "g") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def write_markdown_summary(summary: Dict[str, Any]) -> str:
    """Render a summary dict as a Markdown table (pure function)."""
    lines = [
        "# Component Ablation Summary",
        "",
        f"* Config: `{summary.get('config')}`",
        f"* Out root: `{summary.get('out_root')}`",
        f"* Generated: {summary.get('generated_ts')}",
        "",
        "| Group | Status | Params (M) | Active (M) | Best Val Loss | "
        "Final Val Loss | Delta vs Baseline (%) | Tok/s | Peak Mem (GB) | "
        "Wall Time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary.get("groups", []):
        delta = row.get("delta_vs_baseline_pct")
        delta_text = "-" if delta is None else f"{delta:+.2f}"
        lines.append(
            "| {name} | {status} | {params} | {active} | {best} | {final} "
            "| {delta} | {tps} | {mem} | {wall} |".format(
                name=row.get("name"),
                status=_fmt(row.get("status")),
                params=_fmt(row.get("params_m"), ".3f"),
                active=_fmt(row.get("active_params_m"), ".3f"),
                best=_fmt(row.get("best_val_loss"), ".4f"),
                final=_fmt(row.get("final_val_loss"), ".4f"),
                delta=delta_text,
                tps=_fmt(row.get("mean_tokens_per_sec"), ".1f"),
                mem=_fmt(row.get("peak_mem_gb"), ".2f"),
                wall=_fmt(row.get("wall_time_s"), ".1f"),
            )
        )
    lines.append("")
    lines.append(
        "Note: for the moe group, Params (M) counts all experts while "
        "Active (M) is the dense equivalent (same config with a single "
        "expert)."
    )

    errors = summary.get("errors") or []
    if errors:
        lines.append("")
        lines.append("## Failed Groups (errors)")
        for err in errors:
            command = " ".join(str(c) for c in err.get("command", []))
            lines.append(
                f"* **{err.get('group')}** — exit code "
                f"`{err.get('returncode')}` — `{command}`"
            )

    return "\n".join(lines) + "\n"


def _write_summary_artifacts(summary: Dict[str, Any], out_root: str) -> None:
    exp_root = Path(out_root) / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)
    with open(exp_root / "ablation-summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(exp_root / "ablation-summary.md", "w", encoding="utf-8") as f:
        f.write(write_markdown_summary(summary))


def summarize_groups(
    group_names: List[str],
    config_path: str,
    out_root: str,
) -> Dict[str, Any]:
    """Aggregate per-group experiment summaries into one summary dict.

    Missing / incomplete experiment dirs degrade to None fields; parameter
    counts are config-derived and always computed. Writes JSON + Markdown
    artifacts under ``{out_root}/experiments/``.
    """
    experiment_summaries = {
        name: summarize_experiment(group_exp_dir(out_root, name))
        for name in group_names
    }
    baseline = experiment_summaries.get("baseline") or {}
    baseline_best = baseline.get("best_val_loss")

    rows: List[Dict[str, Any]] = []
    for name in group_names:
        exp = experiment_summaries[name]
        params = count_group_params(name, config_path)
        best = exp.get("best_val_loss")
        delta = None
        if (
            name != "baseline"
            and best is not None
            and baseline_best is not None
        ):
            delta = (best - baseline_best) / baseline_best * 100
        rows.append(
            {
                "name": name,
                "status": exp.get("status"),
                "params_m": round(params["total_params"] / 1e6, 3),
                "active_params_m": round(params["active_params"] / 1e6, 3),
                "best_val_loss": best,
                "final_val_loss": exp.get("final_val_loss"),
                "delta_vs_baseline_pct": delta,
                "mean_tokens_per_sec": exp.get("mean_tokens_per_sec"),
                "peak_mem_gb": exp.get("peak_mem_gb"),
                "wall_time_s": exp.get("wall_time_s"),
            }
        )

    summary: Dict[str, Any] = {
        "generated_ts": datetime.now().astimezone().isoformat(),
        "config": str(config_path),
        "out_root": str(out_root),
        "errors": [],
        "groups": rows,
    }
    _write_summary_artifacts(summary, out_root)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = get_args(argv)
    try:
        selected = select_groups(args.groups)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    train_result: Optional[Dict[str, Any]] = None
    if args.stage in ("train", "all"):
        train_result = run_train_stage(args)
        for err in train_result["errors"]:
            print(
                f"[fail] {err['group']}: exit code {err['returncode']}"
            )

    if args.stage in ("summarize", "all"):
        summary = summarize_groups(selected, args.config, args.out_root)
        if train_result is not None:
            summary["errors"] = list(train_result["errors"])
            _write_summary_artifacts(summary, args.out_root)

    errors = train_result["errors"] if train_result is not None else []
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
