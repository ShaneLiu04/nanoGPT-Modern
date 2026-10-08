"""Tests for FR3a (AR-v0.5.0-hpc-readiness / T003): orchestrator auto --resume.

Contract under test (repo-root ``run_component_ablations.py``), on top of the
FR4 orchestration contract covered by tests/test_component_ablations.py:

  * ``find_group_checkpoint(out_root, name)``: probe
    ``{out_root}/ablation-{name}/latest_ckpt.pt``.  Returns
    ``(ckpt_path, iter_num, ckpt_config_dict)`` when the file exists and
    ``torch.load`` succeeds; returns ``None`` when the file is missing or
    corrupt (torch.load raises) — never raises to the caller.
  * ``ckpt_matches_group(ckpt_cfg, flags)``: compare the checkpoint's config
    dict with the group's CLI flags on the architecture fields
    (``model_type`` vs ``--model``, ``n_kv_head``, ``num_experts``,
    ``position_encoding``, ``ffn_variant``).  Missing fields fall back to
    defaults (n_kv_head None -> n_head or 8; num_experts 1;
    position_encoding "rope"; ffn_variant "swiglu").  Any mismatch -> False.
  * ``build_train_command(..., resume_from=...)``: extended signature — when
    ``resume_from`` is not None the command gains ``--resume
    <str(resume_from)>``; without it ``--resume`` never appears.
  * ``run_train_stage``: for each not-yet-completed group, probe the group
    checkpoint; a matching checkpoint injects ``--resume`` into the training
    command, and a failing resumed run falls back to one rerun *without*
    ``--resume``.  The returned dict gains
    ``"resumed_from_iter": Dict[str, Optional[int]]`` (group -> resumed
    start iter; groups that never resumed are absent or None).

Completed-group skip behavior (manifest status == "completed") is a
pre-existing FR4 contract re-asserted here as a regression guard.

No real training is executed: subprocess.run is monkeypatched at the module
attribute and group checkpoints are faked on disk with minimal
``torch.save`` dicts (iter_num + config), following the style of
tests/test_component_ablations.py.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import run_component_ablations as rca


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(exp_dir, status="completed"):
    """Fake a T001 manifest.json inside an experiment dir."""
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "manifest.json").write_text(
        json.dumps({"exp_id": exp_dir.name, "status": status}),
        encoding="utf-8",
    )
    return exp_dir


def _write_group_ckpt(out_root, name, iter_num, config):
    """Fake a group ``latest_ckpt.pt`` with a minimal loadable dict."""
    group_dir = Path(rca.group_out_dir(str(out_root), name))
    group_dir.mkdir(parents=True, exist_ok=True)
    path = group_dir / "latest_ckpt.pt"
    torch.save({"iter_num": iter_num, "config": dict(config)}, path)
    return path


# ---------------------------------------------------------------------------
# 1. Completed-group skip (pre-existing FR4 contract regression)
# ---------------------------------------------------------------------------

def test_completed_group_skipped(tmp_path, monkeypatch):
    """A completed manifest skips the group without launching training."""
    _write_manifest(
        Path(rca.group_exp_dir(str(tmp_path), "baseline")), status="completed"
    )
    called = []

    def fake_run(cmd, *args, **kwargs):
        called.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rca.subprocess, "run", fake_run)

    args = rca.get_args([
        "--stage", "train",
        "--groups", "baseline",
        "--out_root", str(tmp_path),
        "--config", str(tmp_path / "cfg.yaml"),
    ])
    result = rca.run_train_stage(args)

    assert called == []  # subprocess.run never invoked
    assert "baseline" in result["skipped"]
    assert result["completed"] == []
    assert result["errors"] == []


# ---------------------------------------------------------------------------
# 2. find_group_checkpoint: present / missing / corrupt
# ---------------------------------------------------------------------------

def test_find_group_checkpoint_three_states(tmp_path):
    out_root = str(tmp_path)
    name = "modern-full"
    cfg = {
        "model_type": "modern",
        "n_kv_head": 2,
        "num_experts": 1,
        "position_encoding": "rope",
        "ffn_variant": "swiglu",
    }
    ckpt_path = _write_group_ckpt(out_root, name, iter_num=42, config=cfg)

    # State 1: file exists and loads -> (path, iter_num, config dict).
    found = rca.find_group_checkpoint(out_root, name)
    assert found is not None
    path, iter_num, loaded_cfg = found
    assert Path(path) == ckpt_path
    assert iter_num == 42
    assert isinstance(loaded_cfg, dict)
    assert loaded_cfg["model_type"] == "modern"
    assert loaded_cfg["n_kv_head"] == 2

    # State 2: no checkpoint file -> None.
    assert rca.find_group_checkpoint(out_root, "moe") is None

    # State 3: corrupt file (garbage bytes) -> None, no exception raised.
    corrupt_dir = Path(rca.group_out_dir(out_root, "moe"))
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / "latest_ckpt.pt").write_bytes(
        b"these are not torch archive bytes"
    )
    assert rca.find_group_checkpoint(out_root, "moe") is None


# ---------------------------------------------------------------------------
# 3. ckpt_matches_group: architecture-field comparison
# ---------------------------------------------------------------------------

def test_ckpt_matches_group():
    cfg = {
        "model_type": "modern",
        "n_kv_head": 2,
        "num_experts": 1,
        "position_encoding": "rope",
        "ffn_variant": "swiglu",
    }
    # Fully matching config vs flags.
    assert rca.ckpt_matches_group(
        cfg, ["--model", "modern", "--n_kv_head", "2"]
    ) is True

    # n_kv_head mismatch (ckpt 4 vs flag 2).
    assert rca.ckpt_matches_group(
        dict(cfg, n_kv_head=4), ["--model", "modern", "--n_kv_head", "2"]
    ) is False

    # n_kv_head=None in ckpt falls back to n_head (2) and matches the flag.
    gqa_cfg = dict(cfg, n_kv_head=None, n_head=2)
    assert rca.ckpt_matches_group(
        gqa_cfg, ["--model", "modern", "--n_kv_head", "2"]
    ) is True

    # num_experts: explicit flag matches.
    moe_cfg = {
        "model_type": "modern",
        "num_experts": 4,
        "position_encoding": "rope",
        "ffn_variant": "swiglu",
    }
    assert rca.ckpt_matches_group(
        moe_cfg, ["--model", "modern", "--num_experts", "4"]
    ) is True
    # num_experts=4 vs no flag (default 1) -> False.
    assert rca.ckpt_matches_group(moe_cfg, ["--model", "modern"]) is False

    # model_type vs --model value.
    base_cfg = {"model_type": "baseline"}
    assert rca.ckpt_matches_group(base_cfg, ["--model", "modern"]) is False
    assert rca.ckpt_matches_group(base_cfg, ["--model", "baseline"]) is True

    # position_encoding / ffn_variant mismatches.
    assert rca.ckpt_matches_group(
        dict(cfg, position_encoding="learned"),
        ["--model", "modern", "--n_kv_head", "2"],
    ) is False
    assert rca.ckpt_matches_group(
        dict(cfg, ffn_variant="gelu"),
        ["--model", "modern", "--n_kv_head", "2"],
    ) is False


# ---------------------------------------------------------------------------
# 4. build_train_command: --resume injection
# ---------------------------------------------------------------------------

def test_build_train_command_injects_resume():
    # Without resume_from the command never contains --resume.
    cmd = rca.build_train_command("moe", "cfg.yaml", out_root="o")
    assert "--resume" not in cmd

    # Path resume_from -> ["--resume", str(resume_from)].
    resume_from = Path("x") / "latest_ckpt.pt"
    cmd = rca.build_train_command(
        "moe", "cfg.yaml", out_root="o", resume_from=resume_from
    )
    assert "--resume" in cmd
    assert cmd[cmd.index("--resume") + 1] == str(resume_from)
    # The rest of the FR4 command contract is unchanged.
    assert cmd[cmd.index("--exp_id") + 1] == "ablation-moe"

    # String resume_from is passed through verbatim.
    cmd = rca.build_train_command(
        "moe", "cfg.yaml", out_root="o", resume_from="x/latest_ckpt.pt"
    )
    assert cmd[cmd.index("--resume") + 1] == "x/latest_ckpt.pt"


# ---------------------------------------------------------------------------
# 5. run_train_stage: resumed failure falls back to a rerun
# ---------------------------------------------------------------------------

def test_resume_failure_falls_back_to_rerun(tmp_path, monkeypatch):
    out_root = str(tmp_path)
    name = "baseline"
    _write_group_ckpt(
        out_root, name, iter_num=77, config={"model_type": "baseline"}
    )
    # No manifest -> group not completed.

    calls = []

    def fake_run(cmd, *args, **kwargs):
        cmd = list(cmd)
        calls.append(cmd)
        # First attempt (with --resume) fails; rerun (no --resume) succeeds.
        rc = 1 if "--resume" in cmd else 0
        return SimpleNamespace(returncode=rc)

    monkeypatch.setattr(rca.subprocess, "run", fake_run)

    args = rca.get_args([
        "--stage", "train",
        "--groups", name,
        "--out_root", out_root,
        "--config", str(tmp_path / "cfg.yaml"),
    ])
    result = rca.run_train_stage(args)

    # Exactly two launches: resume attempt then plain rerun.
    assert len(calls) == 2
    assert "--resume" in calls[0]
    assert "--resume" not in calls[1]
    # Both launches target the same group.
    for cmd in calls:
        assert cmd[cmd.index("--exp_id") + 1] == f"ablation-{name}"

    # The rerun succeeded: group completed and recorded as resumed.
    assert name in result["completed"]
    assert name in result["resumed_from_iter"]
    assert result["skipped"] == []


# ---------------------------------------------------------------------------
# 6. run_train_stage: resumed_from_iter recording
# ---------------------------------------------------------------------------

def test_run_train_stage_records_resumed_from_iter(tmp_path, monkeypatch):
    out_root = str(tmp_path)
    name = "baseline"
    _write_group_ckpt(
        out_root, name, iter_num=123, config={"model_type": "baseline"}
    )

    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rca.subprocess, "run", fake_run)

    args = rca.get_args([
        "--stage", "train",
        "--groups", name,
        "--out_root", out_root,
        "--config", str(tmp_path / "cfg.yaml"),
    ])
    result = rca.run_train_stage(args)

    # One launch, carrying --resume pointing at the group's latest_ckpt.pt.
    assert len(calls) == 1
    assert "--resume" in calls[0]
    resume_val = calls[0][calls[0].index("--resume") + 1]
    expected_ckpt = Path(rca.group_out_dir(out_root, name)) / "latest_ckpt.pt"
    assert Path(resume_val) == expected_ckpt

    # resumed_from_iter maps the group to the checkpoint's iter_num.
    assert result["resumed_from_iter"][name] == 123
    assert name in result["completed"]
    assert result["errors"] == []
