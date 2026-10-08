"""Tests for FR4 (AR-v0.4.0-experiments / T005): run_component_ablations orchestration. Red phase: run_component_ablations.py does not exist yet.

Contract under test (repo-root ``run_component_ablations.py``, sibling of
``run_ablations.py``):

  * ``ABLATION_GROUPS`` / ``GROUP_ORDER``: the fixed 7-group component
    ablation matrix (insertion order == execution order).
  * ``get_args``: --stage {train,summarize,all} (default all), --groups
    comma-separated subset (default "all"), --config (default
    "config/pretrain.yaml"), --max_iters int (default None, flag omitted
    when None), --fail_fast store_true, --out_root (default "out").
  * ``group_out_dir`` / ``group_exp_dir``: out_root/ablation-<name> and
    <out_dir>/experiments/ablation-<name> (T001 BaseTrainer convention).
  * ``is_group_completed``: manifest.json exists with status=="completed";
    missing dir / non-completed status / corrupt JSON -> False.
  * ``build_train_command``: [sys.executable, "-m", "training.train_pretrain",
    "--config", cfg, "--out_dir", dir, "--exp_id", "ablation-<name>",
    *group_flags] + optional ["--max_iters", str(n)].
  * ``run_train_stage``: resume-skip completed groups, per-group failure
    isolation (record + continue), --fail_fast stops after first failure.
  * ``count_group_params``: instantiate the group's model from the YAML
    base config + group flags; total includes embeddings
    (non_embedding=False); moe active params == dense (num_experts=1)
    equivalent; other groups active == total.
  * ``summarize_groups`` / ``write_markdown_summary``: per-group rows with
    val-loss / throughput / memory / wall-time fields, delta vs baseline,
    JSON + Markdown artifacts under {out_root}/experiments/.
  * ``main``: orchestrates stages; returns 0 (no errors) or 1 (errors).

No real training is executed: subprocess.run is monkeypatched at the module
attribute and experiment dirs are faked on disk (manifest.json +
metrics.jsonl), following the style of tests/test_experiment.py.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_component_ablations as rca


# ---------------------------------------------------------------------------
# Expected fixed matrix (design.md §6.1 / srs FR4)
# ---------------------------------------------------------------------------

EXPECTED_ORDER = [
    "baseline",
    "modern-full",
    "no-rope",
    "no-swiglu",
    "mha",
    "moe",
    "qk-norm",
]

EXPECTED_GROUPS = {
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

ROW_KEYS = [
    "name",
    "status",
    "params_m",
    "active_params_m",
    "best_val_loss",
    "final_val_loss",
    "delta_vs_baseline_pct",
    "mean_tokens_per_sec",
    "peak_mem_gb",
    "wall_time_s",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tiny_config(tmp_path):
    """Write a tiny flat YAML config (all keys count_group_params needs).

    n_head=8 keeps every group valid (n_kv_head 2 and 8 both divide 8);
    n_layer=2 / n_embd=64 / block_size=32 keep instantiation fast on CPU.
    """
    lines = [
        "model: modern",
        "vocab_size: 50257",
        "block_size: 32",
        "n_layer: 2",
        "n_head: 8",
        "n_embd: 64",
        "n_kv_head: 2",
        "dropout: 0.0",
        "gradient_checkpointing: false",
        "use_ring_attention: false",
        "ring_block_size_q: 16",
        "ring_block_size_kv: 16",
        "position_encoding: rope",
        "ffn_variant: swiglu",
        "qk_norm: false",
        "num_experts: 1",
    ]
    path = Path(tmp_path) / "tiny_pretrain.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_manifest(exp_dir, status="completed", final_metrics=None):
    """Fake a T001 manifest.json inside an experiment dir."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"exp_id": exp_dir.name, "status": status}
    if final_metrics is not None:
        manifest["final_metrics"] = final_metrics
    (exp_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return exp_dir


def _write_metrics(exp_dir, events):
    """Fake a T001 metrics.jsonl inside an experiment dir."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(e) for e in events) + "\n"
    (exp_dir / "metrics.jsonl").write_text(body, encoding="utf-8")


def _fake_subprocess(monkeypatch, fail_groups=(), returncode=1):
    """Monkeypatch run_component_ablations.subprocess.run.

    Records every command; groups named in ``fail_groups`` get a nonzero
    returncode, all others 0. Returns the list of recorded commands.
    """
    calls = []

    def fake_run(cmd, *args, **kwargs):
        cmd = list(cmd)
        calls.append(cmd)
        exp_id = cmd[cmd.index("--exp_id") + 1]
        name = exp_id[len("ablation-"):]
        rc = returncode if name in fail_groups else 0
        return SimpleNamespace(returncode=rc)

    monkeypatch.setattr(rca.subprocess, "run", fake_run)
    return calls


def _called_group_names(calls):
    """Extract group names (execution order) from recorded commands."""
    names = []
    for cmd in calls:
        exp_id = cmd[cmd.index("--exp_id") + 1]
        names.append(exp_id[len("ablation-"):])
    return names


def _fake_all_groups_completed(out_root, best_by_group=None):
    """Create all 7 fake experiment dirs (manifest + metrics.jsonl)."""
    best_by_group = best_by_group or {}
    for i, name in enumerate(EXPECTED_ORDER):
        best = best_by_group.get(name, 3.0 - 0.1 * i)
        final = best + 0.2
        exp_dir = Path(rca.group_exp_dir(str(out_root), name))
        _write_manifest(
            exp_dir,
            status="completed",
            final_metrics={
                "best_val_loss": best,
                "final_val_loss": final,
                "wall_time_s": 60.0 + i,
            },
        )
        _write_metrics(
            exp_dir,
            [
                {"type": "train", "step": 10, "loss": 3.5,
                 "tokens_per_sec": 100.0, "mem_gb": 1.0},
                {"type": "train", "step": 20, "loss": 3.4,
                 "tokens_per_sec": 200.0, "mem_gb": 2.0},
                {"type": "eval", "step": 20, "val_loss": best},
            ],
        )


# ---------------------------------------------------------------------------
# 1. Fixed ablation matrix
# ---------------------------------------------------------------------------

def test_ablation_matrix_has_seven_groups_in_fixed_order():
    assert rca.GROUP_ORDER == EXPECTED_ORDER
    assert len(rca.ABLATION_GROUPS) == 7
    assert list(rca.ABLATION_GROUPS.keys()) == EXPECTED_ORDER
    for name, flags in EXPECTED_GROUPS.items():
        assert rca.ABLATION_GROUPS[name] == flags, (
            f"group {name!r} flags mismatch"
        )


def test_ablation_matrix_no_rope_and_variants_flags():
    # Spot-check the semantically important flag combos.
    assert "--position_encoding" in rca.ABLATION_GROUPS["no-rope"]
    assert rca.ABLATION_GROUPS["no-rope"][
        rca.ABLATION_GROUPS["no-rope"].index("--position_encoding") + 1
    ] == "learned"
    assert rca.ABLATION_GROUPS["no-swiglu"][
        rca.ABLATION_GROUPS["no-swiglu"].index("--ffn_variant") + 1
    ] == "gelu"
    assert rca.ABLATION_GROUPS["mha"][
        rca.ABLATION_GROUPS["mha"].index("--n_kv_head") + 1
    ] == "8"
    assert rca.ABLATION_GROUPS["moe"][
        rca.ABLATION_GROUPS["moe"].index("--num_experts") + 1
    ] == "4"
    assert "--qk_norm" in rca.ABLATION_GROUPS["qk-norm"]


# ---------------------------------------------------------------------------
# 2. Directory helpers
# ---------------------------------------------------------------------------

def test_group_out_dir_and_group_exp_dir_layout():
    assert rca.group_out_dir("out", "moe") == "out/ablation-moe"
    assert rca.group_exp_dir("out", "moe") == (
        "out/ablation-moe/experiments/ablation-moe"
    )
    out_root = "some/root"
    assert rca.group_out_dir(out_root, "no-rope") == (
        f"{out_root}/ablation-no-rope"
    )
    assert rca.group_exp_dir(out_root, "no-rope") == (
        f"{out_root}/ablation-no-rope/experiments/ablation-no-rope"
    )


# ---------------------------------------------------------------------------
# 3. get_args
# ---------------------------------------------------------------------------

def test_get_args_defaults():
    args = rca.get_args([])
    assert args.stage == "all"
    assert args.groups == "all"
    assert args.config == "config/pretrain.yaml"
    assert args.max_iters is None
    assert args.fail_fast is False
    assert args.out_root == "out"


def test_get_args_explicit_overrides(tmp_path):
    cfg = _write_tiny_config(tmp_path)
    args = rca.get_args([
        "--stage", "train",
        "--groups", "baseline,moe",
        "--config", str(cfg),
        "--max_iters", "50",
        "--fail_fast",
        "--out_root", str(tmp_path),
    ])
    assert args.stage == "train"
    assert args.groups == "baseline,moe"
    assert args.config == str(cfg)
    assert args.max_iters == 50
    assert args.fail_fast is True
    assert args.out_root == str(tmp_path)


# ---------------------------------------------------------------------------
# 4. is_group_completed
# ---------------------------------------------------------------------------

def test_is_group_completed_missing_dir_is_false(tmp_path):
    assert rca.is_group_completed(str(tmp_path / "does-not-exist")) is False


def test_is_group_completed_running_status_is_false(tmp_path):
    exp_dir = _write_manifest(tmp_path / "exp", status="running")
    assert rca.is_group_completed(str(exp_dir)) is False


def test_is_group_completed_missing_status_key_is_false(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)
    (exp_dir / "manifest.json").write_text(
        json.dumps({"exp_id": "exp"}), encoding="utf-8"
    )
    assert rca.is_group_completed(str(exp_dir)) is False


def test_is_group_completed_completed_status_is_true(tmp_path):
    exp_dir = _write_manifest(tmp_path / "exp", status="completed")
    assert rca.is_group_completed(str(exp_dir)) is True


def test_is_group_completed_corrupt_json_is_false(tmp_path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)
    (exp_dir / "manifest.json").write_text(
        "{ not valid json !!", encoding="utf-8"
    )
    assert rca.is_group_completed(str(exp_dir)) is False


# ---------------------------------------------------------------------------
# 5. build_train_command
# ---------------------------------------------------------------------------

def test_build_train_command_full_contract():
    expected = [
        sys.executable, "-m", "training.train_pretrain",
        "--config", "config/pretrain.yaml",
        "--out_dir", "out/ablation-qk-norm",
        "--exp_id", "ablation-qk-norm",
        "--model", "modern",
        "--n_kv_head", "2",
        "--qk_norm",
    ]
    assert rca.build_train_command("qk-norm", "config/pretrain.yaml") == (
        expected
    )


def test_build_train_command_omits_max_iters_when_none():
    cmd = rca.build_train_command("baseline", "cfg.yaml", None, out_root="o")
    assert "--max_iters" not in cmd


def test_build_train_command_appends_max_iters_when_set():
    cmd = rca.build_train_command("baseline", "cfg.yaml", 100, out_root="o")
    assert "--max_iters" in cmd
    assert cmd[cmd.index("--max_iters") + 1] == "100"
    # max_iters goes at the end of the command.
    assert cmd[-2:] == ["--max_iters", "100"]


def test_build_train_command_group_flags_and_dirs(tmp_path):
    out_root = str(tmp_path)
    cmd = rca.build_train_command(
        "no-swiglu", str(out_root) + "/c.yaml", 50, out_root=out_root
    )
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "training.train_pretrain"]
    assert cmd[cmd.index("--ffn_variant") + 1] == "gelu"
    out_val = cmd[cmd.index("--out_dir") + 1]
    assert Path(out_val) == Path(rca.group_out_dir(out_root, "no-swiglu"))
    assert cmd[cmd.index("--exp_id") + 1] == "ablation-no-swiglu"


# ---------------------------------------------------------------------------
# 6. run_train_stage: resume / subset / failure isolation / fail_fast
# ---------------------------------------------------------------------------

def test_run_train_stage_skips_completed_groups(tmp_path, monkeypatch):
    """FR4 core acceptance: completed groups are skipped, others trained."""
    cfg = _write_tiny_config(tmp_path)
    pre_completed = ["baseline", "moe"]
    for name in pre_completed:
        _write_manifest(
            Path(rca.group_exp_dir(str(tmp_path), name)), status="completed"
        )

    args = rca.get_args([
        "--stage", "train",
        "--out_root", str(tmp_path),
        "--config", str(cfg),
        "--max_iters", "50",
    ])
    calls = _fake_subprocess(monkeypatch)
    result = rca.run_train_stage(args)

    assert sorted(result["skipped"]) == sorted(pre_completed)
    assert len(calls) == 5
    assert set(result["completed"]) == set(EXPECTED_ORDER) - set(pre_completed)
    assert result["errors"] == []
    assert result["fail_fast_triggered"] is False

    # Execution order follows GROUP_ORDER minus the skipped groups.
    assert _called_group_names(calls) == [
        n for n in EXPECTED_ORDER if n not in pre_completed
    ]

    # No command targets a skipped group's out_dir.
    skipped_dirs = {
        Path(rca.group_out_dir(str(tmp_path), n)) for n in pre_completed
    }
    for cmd in calls:
        out_val = Path(cmd[cmd.index("--out_dir") + 1])
        assert out_val not in skipped_dirs

    # max_iters is forwarded to every launched training command.
    for cmd in calls:
        assert cmd[cmd.index("--max_iters") + 1] == "50"


def test_run_train_stage_groups_subset_in_group_order(tmp_path, monkeypatch):
    cfg = _write_tiny_config(tmp_path)
    args = rca.get_args([
        "--stage", "train",
        "--out_root", str(tmp_path),
        "--config", str(cfg),
        # deliberately out-of-order subset: execution must follow GROUP_ORDER
        "--groups", "moe,baseline",
    ])
    calls = _fake_subprocess(monkeypatch)
    result = rca.run_train_stage(args)

    assert _called_group_names(calls) == ["baseline", "moe"]
    assert result["completed"] == ["baseline", "moe"]
    assert result["skipped"] == []
    assert result["errors"] == []


def test_run_train_stage_single_failure_is_isolated(tmp_path, monkeypatch):
    """A failing group is recorded but does not stop the others."""
    cfg = _write_tiny_config(tmp_path)
    args = rca.get_args([
        "--stage", "train",
        "--out_root", str(tmp_path),
        "--config", str(cfg),
    ])
    calls = _fake_subprocess(
        monkeypatch, fail_groups={"modern-full"}, returncode=1
    )
    result = rca.run_train_stage(args)

    assert len(result["errors"]) == 1
    err = result["errors"][0]
    assert err["group"] == "modern-full"
    assert err["returncode"] == 1
    assert err["command"] == rca.build_train_command(
        "modern-full", str(cfg), None, out_root=str(tmp_path)
    )

    # All 7 groups attempted; the 6 healthy ones still complete.
    assert len(calls) == 7
    assert set(result["completed"]) == set(EXPECTED_ORDER) - {"modern-full"}
    assert result["fail_fast_triggered"] is False


def test_run_train_stage_fail_fast_stops_after_first_failure(tmp_path, monkeypatch):
    cfg = _write_tiny_config(tmp_path)
    args = rca.get_args([
        "--stage", "train",
        "--out_root", str(tmp_path),
        "--config", str(cfg),
        "--fail_fast",
    ])
    calls = _fake_subprocess(
        monkeypatch, fail_groups={"modern-full"}, returncode=1
    )
    result = rca.run_train_stage(args)

    assert result["fail_fast_triggered"] is True
    # modern-full is 2nd in GROUP_ORDER: only baseline + modern-full run.
    assert _called_group_names(calls) == ["baseline", "modern-full"]
    assert len(result["errors"]) == 1
    assert result["errors"][0]["group"] == "modern-full"
    assert result["completed"] == ["baseline"]
    assert result["skipped"] == []


# ---------------------------------------------------------------------------
# 7. count_group_params
# ---------------------------------------------------------------------------

def test_count_group_params_matrix(tmp_path):
    cfg = _write_tiny_config(tmp_path)

    base = rca.count_group_params("baseline", str(cfg))
    modern = rca.count_group_params("modern-full", str(cfg))
    no_rope = rca.count_group_params("no-rope", str(cfg))
    moe = rca.count_group_params("moe", str(cfg))

    # Shape: ints, positive.
    for params in (base, modern, no_rope, moe):
        assert isinstance(params["total_params"], int)
        assert isinstance(params["active_params"], int)
        assert params["total_params"] > 0

    # baseline: BaselineGPT params, active == total.
    assert base["active_params"] == base["total_params"]

    # baseline and modern are different architectures -> different counts.
    assert base["total_params"] != modern["total_params"]

    # no-rope adds a learnable wpe of block_size * n_embd vs modern-full.
    assert no_rope["total_params"] - modern["total_params"] == 32 * 64
    assert no_rope["active_params"] == no_rope["total_params"]

    # moe: total (4 experts) > active; active matches dense modern-full
    # (same config with num_experts=1).
    assert moe["total_params"] > moe["active_params"]
    assert moe["active_params"] == modern["total_params"]


def test_count_group_params_qk_norm_group(tmp_path):
    cfg = _write_tiny_config(tmp_path)
    plain = rca.count_group_params("modern-full", str(cfg))
    qk = rca.count_group_params("qk-norm", str(cfg))
    # QK-Norm adds per-head norm weights but no extra big matrices; both
    # counts remain ints and the group instantiates without error.
    assert qk["total_params"] > 0
    assert qk["active_params"] == qk["total_params"]
    assert plain["total_params"] > 0


# ---------------------------------------------------------------------------
# 8. summarize_groups / write_markdown_summary
# ---------------------------------------------------------------------------

def test_summarize_groups_full_matrix(tmp_path):
    cfg = _write_tiny_config(tmp_path)
    best = {name: 3.0 - 0.1 * i for i, name in enumerate(EXPECTED_ORDER)}
    _fake_all_groups_completed(tmp_path, best_by_group=best)

    summary = rca.summarize_groups(EXPECTED_ORDER, str(cfg), str(tmp_path))

    assert summary["errors"] == []
    assert summary["config"] == str(cfg)
    assert summary["out_root"] == str(tmp_path)
    assert isinstance(summary["generated_ts"], str)

    assert [row["name"] for row in summary["groups"]] == EXPECTED_ORDER
    rows = {row["name"]: row for row in summary["groups"]}
    base_best = best["baseline"]

    for name, row in rows.items():
        for key in ROW_KEYS:
            assert key in row, f"row {name!r} missing key {key!r}"
        assert row["status"] == "completed"
        assert row["best_val_loss"] == pytest.approx(best[name])
        assert row["final_val_loss"] == pytest.approx(best[name] + 0.2)
        assert row["mean_tokens_per_sec"] == pytest.approx(150.0)
        assert row["peak_mem_gb"] == pytest.approx(2.0)
        assert row["wall_time_s"] == pytest.approx(
            60.0 + EXPECTED_ORDER.index(name)
        )
        assert row["params_m"] > 0
        assert row["active_params_m"] > 0

    # delta_vs_baseline_pct: None for baseline itself, exact pct otherwise.
    assert rows["baseline"]["delta_vs_baseline_pct"] is None
    for name in EXPECTED_ORDER[1:]:
        expected_delta = (best[name] - base_best) / base_best * 100
        assert rows[name]["delta_vs_baseline_pct"] == pytest.approx(
            expected_delta
        )

    # params_m matches count_group_params (total/1e6, rounded to 3).
    p_base = rca.count_group_params("baseline", str(cfg))
    assert rows["baseline"]["params_m"] == pytest.approx(
        round(p_base["total_params"] / 1e6, 3)
    )
    # baseline and dense modern: active == total.
    assert rows["modern-full"]["active_params_m"] == pytest.approx(
        rows["modern-full"]["params_m"]
    )
    # moe: active (dense equivalent) < total (4 experts).
    assert rows["moe"]["active_params_m"] < rows["moe"]["params_m"]

    # Artifacts written under {out_root}/experiments/.
    json_path = Path(tmp_path) / "experiments" / "ablation-summary.json"
    md_path = Path(tmp_path) / "experiments" / "ablation-summary.md"
    assert json_path.is_file()
    assert md_path.is_file()

    # JSON round-trips and matches the returned groups.
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["groups"] == summary["groups"]
    assert loaded["errors"] == []

    # Markdown: header row + one row per group.
    md = md_path.read_text(encoding="utf-8")
    table_lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    # header + separator + 7 data rows
    assert len(table_lines) >= 9
    for name in EXPECTED_ORDER:
        assert any(name in ln for ln in table_lines), (
            f"markdown missing row for group {name!r}"
        )


def test_summarize_groups_missing_dirs_degrade_to_none(tmp_path):
    cfg = _write_tiny_config(tmp_path)
    # Only baseline has an experiment dir; the other 6 are missing.
    best = 3.0
    exp_dir = Path(rca.group_exp_dir(str(tmp_path), "baseline"))
    _write_manifest(
        exp_dir,
        status="completed",
        final_metrics={
            "best_val_loss": best,
            "final_val_loss": best + 0.2,
            "wall_time_s": 60.0,
        },
    )
    _write_metrics(exp_dir, [
        {"type": "train", "step": 10, "tokens_per_sec": 100.0, "mem_gb": 1.0},
        {"type": "eval", "step": 10, "val_loss": best},
    ])

    summary = rca.summarize_groups(EXPECTED_ORDER, str(cfg), str(tmp_path))

    rows = {row["name"]: row for row in summary["groups"]}
    assert len(rows) == 7

    base_row = rows["baseline"]
    assert base_row["status"] == "completed"
    assert base_row["best_val_loss"] == pytest.approx(best)
    assert base_row["delta_vs_baseline_pct"] is None

    for name in EXPECTED_ORDER[1:]:
        row = rows[name]
        assert row["status"] is None
        assert row["best_val_loss"] is None
        assert row["final_val_loss"] is None
        assert row["delta_vs_baseline_pct"] is None
        assert row["mean_tokens_per_sec"] is None
        assert row["peak_mem_gb"] is None
        assert row["wall_time_s"] is None
        # params are config-derived and still computed for missing dirs.
        assert row["params_m"] > 0

    # Summary artifacts are still written.
    assert (Path(tmp_path) / "experiments" / "ablation-summary.json").is_file()
    assert (Path(tmp_path) / "experiments" / "ablation-summary.md").is_file()


def _make_summary(errors=None):
    return {
        "generated_ts": "2026-10-07T00:00:00+00:00",
        "config": "config/pretrain.yaml",
        "out_root": "out",
        "errors": list(errors or []),
        "groups": [
            {
                "name": "baseline",
                "status": "completed",
                "params_m": 3.4,
                "active_params_m": 3.4,
                "best_val_loss": 3.0,
                "final_val_loss": 3.2,
                "delta_vs_baseline_pct": None,
                "mean_tokens_per_sec": 150.0,
                "peak_mem_gb": 2.0,
                "wall_time_s": 60.0,
            },
            {
                "name": "moe",
                "status": "completed",
                "params_m": 4.0,
                "active_params_m": 3.4,
                "best_val_loss": 2.5,
                "final_val_loss": 2.7,
                "delta_vs_baseline_pct": -16.667,
                "mean_tokens_per_sec": 140.0,
                "peak_mem_gb": 2.5,
                "wall_time_s": 61.0,
            },
        ],
    }


def test_write_markdown_summary_is_pure_and_complete():
    md = rca.write_markdown_summary(_make_summary())
    assert isinstance(md, str)

    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    # header + separator + 2 data rows
    assert len(lines) >= 4
    assert any("baseline" in ln for ln in lines)
    assert any("moe" in ln for ln in lines)

    # Header names all required columns.
    low = md.lower()
    for token in ("best val loss", "final val loss", "tok/s", "wall time"):
        assert token in low, f"markdown header missing column {token!r}"

    # No errors section when there are no errors.
    assert "error" not in low


def test_write_markdown_summary_includes_errors_section():
    summary = _make_summary(
        errors=[{"group": "moe", "returncode": 1, "command": ["python", "-m"]}]
    )
    md = rca.write_markdown_summary(summary)
    low = md.lower()
    assert "error" in low
    assert "moe" in md


def test_write_markdown_summary_does_not_mutate_input():
    summary = _make_summary()
    snapshot = json.dumps(summary, sort_keys=True)
    rca.write_markdown_summary(summary)
    assert json.dumps(summary, sort_keys=True) == snapshot


# ---------------------------------------------------------------------------
# 9. main smoke tests
# ---------------------------------------------------------------------------

def test_main_stage_train_returns_zero(tmp_path, monkeypatch):
    cfg = _write_tiny_config(tmp_path)
    for name in ("baseline", "moe"):
        _write_manifest(
            Path(rca.group_exp_dir(str(tmp_path), name)), status="completed"
        )
    calls = _fake_subprocess(monkeypatch)

    rc = rca.main([
        "--stage", "train",
        "--out_root", str(tmp_path),
        "--config", str(cfg),
        "--max_iters", "50",
    ])

    assert rc == 0
    # 2 of 7 groups pre-completed -> only 5 trainings launched.
    assert len(calls) == 5


def test_main_stage_train_error_returns_one(tmp_path, monkeypatch):
    cfg = _write_tiny_config(tmp_path)
    _fake_subprocess(monkeypatch, fail_groups={"qk-norm"}, returncode=1)

    rc = rca.main([
        "--stage", "train",
        "--out_root", str(tmp_path),
        "--config", str(cfg),
    ])

    assert rc == 1


def test_main_stage_summarize_returns_zero_and_writes_artifacts(tmp_path):
    cfg = _write_tiny_config(tmp_path)
    _fake_all_groups_completed(tmp_path)

    rc = rca.main([
        "--stage", "summarize",
        "--out_root", str(tmp_path),
        "--config", str(cfg),
    ])

    assert rc == 0
    assert (Path(tmp_path) / "experiments" / "ablation-summary.json").is_file()
    assert (Path(tmp_path) / "experiments" / "ablation-summary.md").is_file()


def test_main_stage_all_orchestrates_train_then_summarize(tmp_path, monkeypatch):
    cfg = _write_tiny_config(tmp_path)
    # All groups already completed: train stage launches nothing,
    # summarize stage writes both artifacts, exit code 0.
    _fake_all_groups_completed(tmp_path)
    calls = _fake_subprocess(monkeypatch)

    rc = rca.main([
        "--stage", "all",
        "--out_root", str(tmp_path),
        "--config", str(cfg),
    ])

    assert rc == 0
    assert len(calls) == 0
    json_path = Path(tmp_path) / "experiments" / "ablation-summary.json"
    md_path = Path(tmp_path) / "experiments" / "ablation-summary.md"
    assert json_path.is_file()
    assert md_path.is_file()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["errors"] == []
    assert [row["name"] for row in loaded["groups"]] == EXPECTED_ORDER
