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
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from model.modern_gpt import ModernGPT, ModernGPTConfig
from utils.config import load_yaml_config
from utils.experiment import summarize_experiment
from utils.monitoring import MetricsHeartbeatWatchdog
from utils.notify import notify_webhook

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
    resume_from: Optional[Any] = None,
) -> List[str]:
    """Assemble the subprocess command for one group's training run.

    ``resume_from`` (path or path-like string to a group checkpoint)
    appends ``--resume <path>`` so the training continues from the last
    in-group checkpoint.
    """
    cmd = [
        sys.executable, "-m", "training.train_pretrain",
        "--config", str(config),
        "--out_dir", group_out_dir(out_root, name),
        "--exp_id", f"ablation-{name}",
        *ABLATION_GROUPS[name],
    ]
    if max_iters is not None:
        cmd += ["--max_iters", str(max_iters)]
    if resume_from is not None:
        cmd += ["--resume", str(resume_from)]
    return cmd


# Architecture fields compared against a group's resolved config overlay
# when deciding whether a checkpoint belongs to the group (design §4.3.1).
_ARCH_FIELD_DEFAULTS: Dict[str, Any] = {
    "num_experts": 1,
    "position_encoding": "rope",
    "ffn_variant": "swiglu",
}


def find_group_checkpoint(
    out_root: str, name: str
) -> Optional[Tuple[Path, int, Dict[str, Any]]]:
    """Probe ``{out_root}/ablation-{name}/latest_ckpt.pt``.

    Returns ``(ckpt_path, iter_num, ckpt_config_dict)`` when the file
    exists and loads; returns ``None`` when it is missing or corrupt
    (``torch.load`` raises) — never raises to the caller.
    """
    ckpt_path = Path(group_out_dir(out_root, name)) / "latest_ckpt.pt"
    if not ckpt_path.is_file():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    iter_num = int(ckpt.get("iter_num", 0))
    raw_cfg = ckpt.get("config")
    ckpt_cfg = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
    return ckpt_path, iter_num, ckpt_cfg


def _arch_matches(ckpt_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> bool:
    """Compare architecture fields of a ckpt config vs a config overlay."""
    ckpt_model = str(ckpt_cfg.get("model_type", "modern"))
    flag_model = str(overrides.get("model", "modern"))
    if ckpt_model != flag_model:
        return False

    if ckpt_model != "baseline":
        # GQA heads: ckpt None -> n_head (or 8); overlay likewise.
        ckpt_nkv = ckpt_cfg.get("n_kv_head")
        if ckpt_nkv is None:
            ckpt_nkv = ckpt_cfg.get("n_head", 8)
        flag_nkv = overrides.get("n_kv_head")
        if flag_nkv is None:
            flag_nkv = overrides.get("n_head", 8)
        if int(ckpt_nkv) != int(flag_nkv):
            return False

    for key, default in _ARCH_FIELD_DEFAULTS.items():
        ckpt_val = ckpt_cfg.get(key, default)
        flag_val = overrides.get(key, default)
        if key == "num_experts":
            if int(ckpt_val) != int(flag_val):
                return False
        elif str(ckpt_val) != str(flag_val):
            return False
    return True


def ckpt_matches_group(ckpt_cfg: Dict[str, Any], flags: List[str]) -> bool:
    """True when the checkpoint's config matches the group's CLI flags.

    Compares ``model_type`` vs ``--model`` plus the architecture fields
    (``n_kv_head``, ``num_experts``, ``position_encoding``,
    ``ffn_variant``); fields absent on either side fall back to defaults.
    """
    return _arch_matches(dict(ckpt_cfg), _flags_to_overrides(list(flags)))


def _group_overrides(args: argparse.Namespace, name: str) -> Dict[str, Any]:
    """Merged YAML + group-flag overlay used for checkpoint matching."""
    merged: Dict[str, Any] = {}
    try:
        merged = dict(load_yaml_config(args.config) or {})
    except Exception:
        merged = {}
    merged.update(_flags_to_overrides(ABLATION_GROUPS[name]))
    return merged


def start_group_watchdog(
    out_root: str, name: str, poll_interval: float = 15.0
) -> Optional[MetricsHeartbeatWatchdog]:
    """Build (not yet started) a heartbeat watchdog for one group.

    The caller starts/stops it around the training subprocess; the
    watchdog alerts (stderr + webhook) on stalls.  Any setup failure
    degrades to ``None`` and never blocks the training run.
    """
    try:
        metrics_path = os.path.join(group_exp_dir(out_root, name), "metrics.jsonl")
        return MetricsHeartbeatWatchdog(
            metrics_path, name, poll_interval=poll_interval
        )
    except Exception as exc:
        print(f"[warn] watchdog setup failed for {name}: {exc}")
        return None


def notify_failures(
    errors: List[Dict[str, Any]], env: Optional[Dict[str, str]] = None
) -> bool:
    """Send a failure digest to the configured webhook (missing URL: skip)."""
    if not errors:
        return False
    lines = [f"[ablations] {len(errors)} group(s) failed:"]
    for err in errors:
        lines.append(f"- {err['group']} (exit {err['returncode']})")
    return notify_webhook("\n".join(lines), env=env)


def _run_with_watchdog(
    cmd: List[str], out_root: str, name: str
) -> subprocess.CompletedProcess:
    """Run one training subprocess under a heartbeat watchdog.

    The watchdog monitors the group's metrics.jsonl while the subprocess
    runs and alerts on stalls (never kills); it is stopped afterwards.
    """
    watchdog = start_group_watchdog(out_root, name)
    if watchdog is not None:
        try:
            watchdog.start()
        except Exception as exc:
            print(f"[warn] watchdog start failed for {name}: {exc}")
            watchdog = None
    try:
        return subprocess.run(cmd)
    finally:
        if watchdog is not None:
            watchdog.stop()


def run_train_stage(args: argparse.Namespace) -> Dict[str, Any]:
    """Train every selected, not-yet-completed group as a subprocess.

    Completed groups (manifest status == 'completed') are skipped for
    resume.  For each pending group the in-group ``latest_ckpt.pt`` is
    probed; an architecture-matching checkpoint injects ``--resume`` (and
    a failing resumed run falls back to one rerun without ``--resume``).
    A failing group is recorded and normally does not stop the others
    (``--fail_fast`` stops after the first failure).
    """
    completed: List[str] = []
    skipped: List[str] = []
    errors: List[Dict[str, Any]] = []
    resumed_from_iter: Dict[str, Optional[int]] = {}
    fail_fast_triggered = False

    for name in select_groups(args.groups):
        exp_dir = group_exp_dir(args.out_root, name)
        if is_group_completed(exp_dir):
            print(f"[skip] {name}: already completed ({exp_dir})")
            skipped.append(name)
            continue

        resume_from = None
        found = find_group_checkpoint(args.out_root, name)
        if found is not None:
            ckpt_path, iter_num, ckpt_cfg = found
            if _arch_matches(ckpt_cfg, _group_overrides(args, name)):
                resume_from = ckpt_path
                resumed_from_iter[name] = iter_num
                print(f"[resume] {name}: from iter {iter_num} ({ckpt_path})")
            else:
                print(
                    f"[warn] {name}: checkpoint architecture mismatch "
                    f"({ckpt_path}); starting a fresh run"
                )

        cmd = build_train_command(
            name,
            args.config,
            args.max_iters,
            out_root=args.out_root,
            resume_from=resume_from,
        )
        print(f"[train] {name}: {' '.join(cmd)}")
        result = _run_with_watchdog(cmd, args.out_root, name)
        if result.returncode == 0:
            completed.append(name)
        elif resume_from is not None:
            # A failed resumed run degrades to one fresh rerun.
            print(
                f"[warn] {name}: resumed run failed "
                f"(exit code {result.returncode}); rerunning without --resume"
            )
            cmd = build_train_command(
                name,
                args.config,
                args.max_iters,
                out_root=args.out_root,
            )
            print(f"[train] {name}: {' '.join(cmd)}")
            result = _run_with_watchdog(cmd, args.out_root, name)
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
        "resumed_from_iter": resumed_from_iter,
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
        "Wall Time (s) | Resumed From Iter |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    resumed_map = summary.get("resumed_from_iter") or {}
    for row in summary.get("groups", []):
        delta = row.get("delta_vs_baseline_pct")
        delta_text = "-" if delta is None else f"{delta:+.2f}"
        lines.append(
            "| {name} | {status} | {params} | {active} | {best} | {final} "
            "| {delta} | {tps} | {mem} | {wall} | {resumed} |".format(
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
                resumed=_fmt(resumed_map.get(row.get("name"))),
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
        if train_result["errors"]:
            notified = notify_failures(train_result["errors"])
            if notified:
                print("[notify] failure digest sent to webhook")

    if args.stage in ("summarize", "all"):
        summary = summarize_groups(selected, args.config, args.out_root)
        if train_result is not None:
            summary["errors"] = list(train_result["errors"])
            summary["resumed_from_iter"] = dict(
                train_result.get("resumed_from_iter", {})
            )
            _write_summary_artifacts(summary, args.out_root)

    errors = train_result["errors"] if train_result is not None else []
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
