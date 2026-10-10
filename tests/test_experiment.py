"""Tests for FR1 (AR-v0.4.0-experiments / T001): ExperimentRecorder + BaseTrainer integration.

Red phase: ``utils/experiment.py`` does not exist yet, so this module fails at
collection with ``ModuleNotFoundError`` until the Green implementation lands.
"""

import argparse
import builtins
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from training.trainer_base import BaseTrainer
from utils.experiment import (
    ExperimentRecorder,
    collect_env,
    load_metrics,
    summarize_experiment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_recorder(exp_dir, **kwargs):
    """Build an ExperimentRecorder with a default hyperparam config."""
    kwargs.setdefault("config", {"seed": 1337, "lr": 6e-4})
    return ExperimentRecorder(exp_dir=str(exp_dir), exp_id="exp1", **kwargs)


def _read_jsonl(path):
    """Parse a jsonl file strictly: every non-empty line must be valid JSON."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _read_manifest(exp_dir):
    return json.loads((Path(exp_dir) / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# collect_env
# ---------------------------------------------------------------------------


def test_collect_env_has_required_fields():
    env = collect_env()
    for key in ("gpu", "vram_gb", "sm", "torch", "cuda", "python", "os", "git_commit"):
        assert key in env, f"collect_env missing field: {key}"


def test_collect_env_git_missing_degrades_to_unknown(monkeypatch):
    """When git is unavailable, git_commit must be 'unknown' and no error raised."""

    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(subprocess, "run", _raise)
    monkeypatch.setattr(subprocess, "check_output", _raise)
    monkeypatch.setattr(subprocess, "check_call", _raise)
    env = collect_env()
    assert env["git_commit"] == "unknown"


# ---------------------------------------------------------------------------
# ExperimentRecorder: per-event jsonl writing
# ---------------------------------------------------------------------------


def test_record_step_writes_train_event(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.record_step(
        1234,
        loss=3.21,
        lr=5.1e-4,
        grad_norm=0.87,
        tokens_per_sec=41200.0,
        mem_gb=9.8,
    )
    events = _read_jsonl(tmp_path / "metrics.jsonl")
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "train"
    assert ev["step"] == 1234
    assert ev["loss"] == pytest.approx(3.21)
    assert ev["lr"] == pytest.approx(5.1e-4)
    assert ev["grad_norm"] == pytest.approx(0.87)
    assert ev["tokens_per_sec"] == pytest.approx(41200.0)
    assert ev["mem_gb"] == pytest.approx(9.8)
    assert "ts" in ev


def test_record_eval_writes_eval_event(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.record_eval(2000, val_loss=3.05)
    events = _read_jsonl(tmp_path / "metrics.jsonl")
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "eval"
    assert ev["step"] == 2000
    assert ev["val_loss"] == pytest.approx(3.05)
    assert "ts" in ev


def test_record_event_writes_monitor_event(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.record_event("loss_spike", 2500, z_score=3.4, lr_reduce_factor=0.5)
    events = _read_jsonl(tmp_path / "metrics.jsonl")
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "event"
    assert ev["event"] == "loss_spike"
    assert ev["step"] == 2500
    assert ev["z_score"] == pytest.approx(3.4)
    assert ev["lr_reduce_factor"] == pytest.approx(0.5)
    assert "ts" in ev


def test_record_generic_event_auto_ts(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.record({"step": 0, "type": "info", "msg": "hello"})
    events = _read_jsonl(tmp_path / "metrics.jsonl")
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "info"
    assert ev["msg"] == "hello"
    assert "ts" in ev


def test_record_ts_is_iso_string(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.record_step(1, loss=1.0)
    ev = _read_jsonl(tmp_path / "metrics.jsonl")[0]
    assert isinstance(ev["ts"], str)
    # Must be a valid ISO timestamp.
    datetime.fromisoformat(ev["ts"])


def test_record_appends_and_preserves_loss_curve(tmp_path):
    rec = _make_recorder(tmp_path)
    losses = [4.0, 3.5, 3.2, 3.1, 3.0]
    for step, loss in enumerate(losses, start=1):
        rec.record_step(step, loss=loss)
    rec.record_eval(5, val_loss=3.0)

    events = load_metrics(str(tmp_path / "metrics.jsonl"))
    assert len(events) == 6
    curve = [e["loss"] for e in events if e["type"] == "train"]
    assert curve == losses
    assert events[-1]["type"] == "eval"
    assert events[-1]["val_loss"] == pytest.approx(3.0)


def test_metrics_jsonl_records_fifty_step_run(tmp_path):
    """FR1: a >=50-step run leaves a parseable metrics.jsonl with >=50 records
    plus a manifest containing environment and hyperparameter fields."""
    rec = _make_recorder(tmp_path)
    for step in range(1, 51):
        rec.record_step(step, loss=4.0 - step * 0.01, lr=6e-4, tokens_per_sec=40000.0)
        if step % 10 == 0:
            rec.record_eval(step, val_loss=3.5 - step * 0.005)
    rec.record_event("loss_spike", 25, z_score=3.4, lr_reduce_factor=0.5)

    path = tmp_path / "metrics.jsonl"
    assert path.exists()
    events = load_metrics(str(path))
    assert len(events) >= 50

    # Loss curve and monitoring events are losslessly recoverable in pure Python.
    train_curve = [e["loss"] for e in events if e["type"] == "train"]
    assert len(train_curve) == 50
    evals = [e for e in events if e["type"] == "eval"]
    assert len(evals) == 5
    spikes = [e for e in events if e["type"] == "event" and e["event"] == "loss_spike"]
    assert len(spikes) == 1
    assert spikes[0]["z_score"] == pytest.approx(3.4)
    assert spikes[0]["lr_reduce_factor"] == pytest.approx(0.5)

    rec.write_manifest()
    manifest = _read_manifest(tmp_path)
    assert isinstance(manifest["env"], dict)
    assert "torch" in manifest["env"]
    assert isinstance(manifest["config"], dict)
    assert manifest["config"]["seed"] == 1337


# ---------------------------------------------------------------------------
# ExperimentRecorder: manifest
# ---------------------------------------------------------------------------


def test_write_manifest_is_idempotent(tmp_path):
    rec = _make_recorder(tmp_path)
    rec.record_step(1, loss=1.0)

    path1 = rec.write_manifest()
    assert isinstance(path1, str)
    assert Path(path1).exists()

    path2 = rec.write_manifest()
    assert Path(path2) == Path(path1)

    manifest = _read_manifest(tmp_path)
    assert manifest["exp_id"] == "exp1"
    assert manifest["status"] == "running"
    assert manifest["seed"] == 1337
    assert manifest["resume_count"] == 0
    assert manifest["epoch_repetition"] == 0.0
    assert "git_commit" in manifest
    assert "start_ts" in manifest
    assert isinstance(manifest["env"], dict)
    assert isinstance(manifest["config"], dict)
    assert manifest["config"]["lr"] == pytest.approx(6e-4)


def test_write_manifest_merges_extras_across_calls(tmp_path):
    """Extras passed to earlier write_manifest calls must survive later rewrites
    (startup -> save_checkpoint -> finalize all rewrite the same file)."""
    rec = _make_recorder(tmp_path)
    rec.write_manifest(artifacts={"out_dir": str(tmp_path), "checkpoints": ["ckpt_001.pt"]})
    rec.write_manifest(
        final_metrics={
            "best_val_loss": 3.0,
            "final_val_loss": 3.1,
            "wall_time_s": 12.5,
        }
    )
    manifest = _read_manifest(tmp_path)
    assert manifest["artifacts"]["out_dir"] == str(tmp_path)
    assert manifest["artifacts"]["checkpoints"] == ["ckpt_001.pt"]
    assert manifest["final_metrics"]["best_val_loss"] == pytest.approx(3.0)
    assert manifest["final_metrics"]["final_val_loss"] == pytest.approx(3.1)
    assert manifest["final_metrics"]["wall_time_s"] == pytest.approx(12.5)


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_finalize_status_transitions(tmp_path, status):
    rec = _make_recorder(tmp_path)
    rec.record_step(1, loss=1.0)
    rec.finalize(
        status,
        {"best_val_loss": 3.0, "final_val_loss": 3.1, "wall_time_s": 60.0},
    )
    manifest = _read_manifest(tmp_path)
    assert manifest["status"] == status
    assert manifest["final_metrics"]["best_val_loss"] == pytest.approx(3.0)
    assert manifest["final_metrics"]["wall_time_s"] == pytest.approx(60.0)
    assert "end_ts" in manifest


# ---------------------------------------------------------------------------
# ExperimentRecorder: resume semantics
# ---------------------------------------------------------------------------


def test_resume_appends_to_existing_jsonl(tmp_path):
    rec1 = _make_recorder(tmp_path)
    rec1.record_step(1, loss=4.0)
    rec1.record_step(2, loss=3.5)
    rec1.record_eval(2, val_loss=3.4)
    assert len(load_metrics(str(tmp_path / "metrics.jsonl"))) == 3

    rec2 = ExperimentRecorder(
        exp_dir=str(tmp_path), exp_id="exp1", config={"seed": 1337}, resume=True
    )
    rec2.record_step(3, loss=3.2)
    rec2.record_eval(3, val_loss=3.1)

    events = load_metrics(str(tmp_path / "metrics.jsonl"))
    assert len(events) == 5
    assert [e["step"] for e in events] == [1, 2, 2, 3, 3]
    # Earlier records are preserved verbatim.
    assert events[0]["loss"] == pytest.approx(4.0)


def test_default_overwrites_existing_jsonl(tmp_path):
    rec1 = _make_recorder(tmp_path)
    rec1.record_step(1, loss=4.0)
    rec1.record_step(2, loss=3.5)
    assert len(load_metrics(str(tmp_path / "metrics.jsonl"))) == 2

    # Default resume=False starts fresh (overwrites the file).
    rec2 = _make_recorder(tmp_path)
    rec2.record_step(1, loss=2.0)

    events = load_metrics(str(tmp_path / "metrics.jsonl"))
    assert len(events) == 1
    assert events[0]["loss"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# load_metrics / summarize_experiment
# ---------------------------------------------------------------------------


def test_load_metrics_tolerates_truncated_last_line(tmp_path):
    """Crash protection: a partially-written final line must be dropped."""
    path = tmp_path / "metrics.jsonl"
    good = [
        json.dumps({"ts": "2026-01-01T00:00:00", "step": 1, "type": "train", "loss": 4.0}),
        json.dumps({"ts": "2026-01-01T00:00:01", "step": 2, "type": "train", "loss": 3.5}),
        json.dumps({"ts": "2026-01-01T00:00:02", "step": 2, "type": "eval", "val_loss": 3.4}),
    ]
    truncated = '{"ts": "2026-01-01T00:00:03", "step": 3, "type": "train", "loss": 3.2'
    path.write_text("\n".join(good) + "\n" + truncated, encoding="utf-8")

    events = load_metrics(str(path))
    assert len(events) == 3
    assert events[-1]["type"] == "eval"
    assert events[-1]["val_loss"] == pytest.approx(3.4)


def test_load_metrics_empty_file(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_metrics(str(path)) == []


def test_summarize_experiment_from_fake_dir(tmp_path):
    exp_dir = tmp_path / "fake_exp"
    exp_dir.mkdir()
    events = [
        {
            "ts": "2026-01-01T00:00:00",
            "step": 100,
            "type": "train",
            "loss": 4.0,
            "tokens_per_sec": 100.0,
            "mem_gb": 5.0,
        },
        {
            "ts": "2026-01-01T00:00:01",
            "step": 200,
            "type": "train",
            "loss": 3.5,
            "tokens_per_sec": 200.0,
            "mem_gb": 8.0,
        },
        {"ts": "2026-01-01T00:00:02", "step": 200, "type": "eval", "val_loss": 3.2},
        {"ts": "2026-01-01T00:00:03", "step": 400, "type": "eval", "val_loss": 3.05},
        {
            "ts": "2026-01-01T00:00:04",
            "step": 250,
            "type": "event",
            "event": "loss_spike",
            "z_score": 3.4,
        },
    ]
    (exp_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    manifest = {
        "exp_id": "fake_exp",
        "status": "completed",
        "final_metrics": {
            "best_val_loss": 2.9,
            "final_val_loss": 3.05,
            "wall_time_s": 42.5,
        },
    }
    (exp_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = summarize_experiment(str(exp_dir))
    assert summary["best_val_loss"] == pytest.approx(2.9)
    assert summary["final_val_loss"] == pytest.approx(3.05)
    assert summary["mean_tokens_per_sec"] == pytest.approx(150.0)
    assert summary["peak_mem_gb"] == pytest.approx(8.0)
    assert summary["wall_time_s"] == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# Write-failure degradation (FR1: OSError must not interrupt training)
# ---------------------------------------------------------------------------


def test_record_write_failure_warns_and_does_not_raise(tmp_path, monkeypatch, caplog):
    """When opening metrics.jsonl raises OSError, recording must degrade to a
    console warning instead of raising."""
    real_open = builtins.open

    def failing_open(file, *args, **kwargs):
        if "metrics.jsonl" in str(file):
            raise OSError("disk full")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)

    with caplog.at_level(logging.WARNING):
        rec = ExperimentRecorder(exp_dir=str(tmp_path), exp_id="exp1", config={"seed": 1})
        rec.record_step(1, loss=1.0)
        rec.record_eval(1, val_loss=2.0)
        rec.record_event("loss_spike", 1, z_score=3.0)
        rec.record({"step": 1, "type": "info", "msg": "still alive"})

    assert not (tmp_path / "metrics.jsonl").exists()
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# BaseTrainer integration (lightweight)
# ---------------------------------------------------------------------------


class DummyTrainer(BaseTrainer):
    """Minimal concrete trainer: all template hooks keep their pass defaults."""

    def train(self):
        pass


def _trainer_args(out_dir, **overrides):
    defaults = dict(
        seed=1337,
        device="cpu",
        backend="gloo",
        out_dir=str(out_dir),
        use_wandb=False,
        exp_id=None,
        no_metrics=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_base_trainer_sets_up_recorder_on_master(tmp_path):
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir))
    try:
        assert trainer.master_process is True
        assert trainer.recorder is not None

        # exp_dir defaults to <out_dir>/experiments/<basename(out_dir)>.
        exp_dir = out_dir / "experiments" / "run42"
        manifest_path = exp_dir / "manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "running"
        assert manifest["exp_id"] == "run42"
        assert manifest["seed"] == 1337
        assert isinstance(manifest["env"], dict)
        assert isinstance(manifest["config"], dict)
    finally:
        trainer.cleanup()


def test_base_trainer_exp_id_override(tmp_path):
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir, exp_id="custom_exp"))
    try:
        assert trainer.recorder is not None
        exp_dir = out_dir / "experiments" / "custom_exp"
        manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["exp_id"] == "custom_exp"
    finally:
        trainer.cleanup()


def test_base_trainer_no_metrics_disables_recorder(tmp_path):
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir, no_metrics=True))
    try:
        assert trainer.recorder is None
        # val/loss logging must not raise without a recorder.
        trainer.log_scalar("val/loss", 3.14, step=10)
    finally:
        trainer.cleanup()


def test_base_trainer_log_scalar_val_loss_dual_write(tmp_path):
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir))
    try:
        metrics_path = out_dir / "experiments" / "run42" / "metrics.jsonl"

        # Non-eval tags do not produce metrics.jsonl events.
        trainer.log_scalar("train/loss", 4.2, step=10)
        assert (not metrics_path.exists()) or load_metrics(str(metrics_path)) == []

        # val/loss is dual-written as an eval event.
        trainer.log_scalar("val/loss", 3.14, step=100)
        events = load_metrics(str(metrics_path))
        eval_events = [e for e in events if e["type"] == "eval"]
        assert len(eval_events) == 1
        assert eval_events[0]["step"] == 100
        assert eval_events[0]["val_loss"] == pytest.approx(3.14)

        # Repeated logging appends further eval events.
        trainer.log_scalar("val/loss", 3.10, step=200)
        events = load_metrics(str(metrics_path))
        assert len([e for e in events if e["type"] == "eval"]) == 2
    finally:
        trainer.cleanup()
