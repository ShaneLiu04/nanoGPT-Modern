"""Tests for FR2 (AR-v0.4.0-experiments / T002): monitoring integration via on_step_end.

Red phase: ``on_step_end`` / ``_setup_monitors`` do not exist yet, so these
tests fail with ``AttributeError`` until the Green implementation lands.

Contract under test (training/trainer_base.py):
  * ``_setup_monitors()`` (end of the BaseTrainer template chain):
      - master-only; ``args.no_monitoring=True`` leaves ``spike_detector`` and
        ``grad_monitor`` as None (detector never built)
      - default builds ``LossSpikeDetector()`` (window_size=100, threshold=3.0,
        auto_reduce_lr=True, lr_reduce_factor=0.5) and
        ``GradientNormMonitor(self.logger)`` (logger may be None)
      - construction failures degrade to a warning, never raise
  * ``on_step_end(step, loss, lr=None, grad_norm=None, tokens_per_sec=None)``:
      - non-master processes return immediately without writing anything
      - mem_gb: CUDA-only (torch.cuda.max_memory_allocated()/2**30, rounded to
        3 decimals); absent on CPU
      - monitoring path (spike_detector is not None):
          a. grad_norm None + model present -> grad_monitor.compute(model,
             step)["total_norm"]; compute failures degrade to None
          b. spike_detector.update(float(loss), step) -> dict; on spike:
             WARN print, lr *= lr_multiplier on all param_groups (when both
             lr_multiplier and optimizer exist), and
             recorder.record_event("loss_spike", step, loss=..., z_score=...,
             lr_reduce_factor=..., window_mean=..., window_std=...)
          c. any monitoring exception only warns; record_step still runs
      - record path (recorder is not None): record_step(step, loss=...,
        lr=..., grad_norm=..., tokens_per_sec=...[, mem_gb=...]); None-valued
        fields are omitted from the event
  * the five training scripts expose a ``--no_monitoring`` store_true CLI flag.
"""

import argparse
import importlib
import sys

import pytest
import torch

from training import trainer_base as trainer_base_module
from training.trainer_base import BaseTrainer
from utils import monitoring as monitoring_module
from utils.experiment import load_metrics


# ---------------------------------------------------------------------------
# Helpers (same pattern as tests/test_experiment.py)
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
        no_monitoring=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _metrics_path(out_dir, name="run42"):
    return out_dir / "experiments" / name / "metrics.jsonl"


def _events(out_dir, name="run42"):
    return load_metrics(str(_metrics_path(out_dir, name)))


def _tiny_model_with_grads():
    """A tiny nn.Module whose parameters all carry gradients."""
    model = torch.nn.Linear(8, 8)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    return model


# ---------------------------------------------------------------------------
# _setup_monitors: construction
# ---------------------------------------------------------------------------

def test_setup_monitors_default_configuration(tmp_path):
    """Default (no_monitoring=False): both monitors are built on master with
    the agreed default LossSpikeDetector parameters."""
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir))
    try:
        assert trainer.master_process is True
        assert trainer.spike_detector is not None
        assert trainer.grad_monitor is not None

        det = trainer.spike_detector
        assert det.window_size == 100
        assert det.threshold == 3.0
        assert det.auto_reduce_lr is True
        assert det.lr_reduce_factor == 0.5
    finally:
        trainer.cleanup()


def test_setup_monitors_disabled_with_no_monitoring(tmp_path):
    """no_monitoring=True: detector/monitor are never built, and on_step_end
    produces only train events (no loss_spike detection even for spiky
    losses)."""
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir, no_monitoring=True))
    try:
        assert trainer.spike_detector is None
        assert trainer.grad_monitor is None

        # A spiky loss must not produce any monitoring event.
        trainer.on_step_end(step=1, loss=3.0, lr=6e-4)
        trainer.on_step_end(step=2, loss=10.0, lr=6e-4)
        trainer.on_step_end(step=3, loss=3.0, lr=6e-4)

        events = _events(out_dir)
        assert len(events) == 3
        assert all(e["type"] == "train" for e in events)
        assert not any(
            e["type"] == "event" and e.get("event") == "loss_spike" for e in events
        )
    finally:
        trainer.cleanup()


def test_setup_monitors_construction_failure_degrades(tmp_path):
    """If monitor construction raises, _setup_monitors must degrade (both
    monitors None) instead of propagating the exception."""
    def _boom(*args, **kwargs):
        raise RuntimeError("monitor construction failed")

    # Patch every plausible import style used by the Green implementation:
    # ``from utils.monitoring import X`` binds in trainer_base, while
    # ``import utils.monitoring`` resolves through the utils module.
    mp = pytest.MonkeyPatch()
    try:
        for mod in (trainer_base_module, monitoring_module):
            mp.setattr(mod, "LossSpikeDetector", _boom, raising=False)
            mp.setattr(mod, "GradientNormMonitor", _boom, raising=False)

        # Must not raise despite the failing constructors.
        out_dir = tmp_path / "run42"
        trainer = DummyTrainer(_trainer_args(out_dir))
        try:
            assert trainer.spike_detector is None
            assert trainer.grad_monitor is None
        finally:
            trainer.cleanup()
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
# on_step_end: recording path (FR2 acceptance: grad_norm in metrics.jsonl)
# ---------------------------------------------------------------------------

def test_on_step_end_records_grad_norm_over_100_steps(tmp_path):
    """FR2 acceptance: with monitoring enabled and a tiny model with grads,
    100 stable steps must leave 100 train events in metrics.jsonl, each
    carrying a positive grad_norm field, with no spurious loss spikes."""
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir))
    try:
        trainer.model = _tiny_model_with_grads()
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=6e-4)

        # Stable loss sequence (std > 0 but max z << 3) avoids false spikes.
        for step in range(1, 101):
            loss = 3.0 + (step % 7) * 1e-3
            trainer.on_step_end(step=step, loss=loss, lr=6e-4)

        events = _events(out_dir)
        train_events = [e for e in events if e["type"] == "train"]
        assert len(train_events) == 100

        for ev in train_events:
            assert "grad_norm" in ev, (
                f"train event at step {ev['step']} is missing grad_norm"
            )
            assert ev["grad_norm"] > 0.0
            assert ev["lr"] == pytest.approx(6e-4)
            assert ev["loss"] == pytest.approx(3.0 + (ev["step"] % 7) * 1e-3)

        # Stable loss sequence: no spike events may be emitted.
        assert not any(e["type"] == "event" for e in events)
    finally:
        trainer.cleanup()


def test_on_step_end_pass_through_fields(tmp_path):
    """Explicitly supplied loss/lr/grad_norm/tokens_per_sec land verbatim in
    the train event; on CPU the mem_gb field must be absent."""
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir))
    try:
        trainer.model = None  # grad_monitor.compute path not needed
        trainer.on_step_end(
            step=7,
            loss=2.5,
            lr=6e-4,
            grad_norm=0.42,
            tokens_per_sec=123.5,
        )

        events = _events(out_dir)
        assert len(events) == 1
        ev = events[0]
        assert ev["type"] == "train"
        assert ev["step"] == 7
        assert ev["loss"] == pytest.approx(2.5)
        assert ev["lr"] == pytest.approx(6e-4)
        assert ev["grad_norm"] == pytest.approx(0.42)
        assert ev["tokens_per_sec"] == pytest.approx(123.5)

        if torch.cuda.is_available():
            # On CUDA, mem_gb is max_memory_allocated()/2**30 (3 decimals).
            assert isinstance(ev.get("mem_gb"), float)
        else:
            # On CPU, mem_gb is None and therefore must not appear.
            assert "mem_gb" not in ev
    finally:
        trainer.cleanup()


def test_on_step_end_non_master_is_noop(tmp_path):
    """Non-master processes must not write anything to metrics.jsonl."""
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir))
    try:
        assert _events(out_dir) == []

        trainer.master_process = False
        trainer.on_step_end(step=1, loss=3.0, lr=6e-4, grad_norm=0.5)

        assert _events(out_dir) == []
    finally:
        trainer.cleanup()


# ---------------------------------------------------------------------------
# on_step_end: monitoring path (spike detection + LR reduction)
# ---------------------------------------------------------------------------

def test_on_step_end_spike_writes_event_and_reduces_lr(tmp_path):
    """Injected spike (stable window, then a big loss) must:
      1. write a loss_spike event with z_score > 3, lr_reduce_factor=0.5 and
         the window_mean/window_std fields,
      2. multiply every optimizer param_group lr by the multiplier (1.0 -> 0.5),
      3. still record the regular train event for that step.
    """
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir))
    try:
        model = torch.nn.Linear(8, 8)
        trainer.model = model
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

        # Pre-fill the detector window with a stable loss so the next big
        # loss triggers z > 3.
        for i in range(15):
            trainer.spike_detector.update(3.0, i)

        assert all(g["lr"] == 1.0 for g in trainer.optimizer.param_groups)
        assert _events(out_dir) == []

        # Spike step.
        trainer.on_step_end(step=15, loss=10.0, lr=1.0)

        # LR was reduced on every param group.
        for group in trainer.optimizer.param_groups:
            assert group["lr"] == pytest.approx(0.5)

        events = _events(out_dir)
        spikes = [e for e in events if e["type"] == "event" and e["event"] == "loss_spike"]
        assert len(spikes) == 1
        ev = spikes[0]
        assert ev["step"] == 15
        assert ev["loss"] == pytest.approx(10.0)
        assert ev["z_score"] > 3.0
        assert ev["lr_reduce_factor"] == pytest.approx(0.5)
        assert "window_mean" in ev
        assert "window_std" in ev
        assert ev["window_mean"] > 0.0

        # The regular train event for the spike step is still recorded.
        trains = [e for e in events if e["type"] == "train" and e["step"] == 15]
        assert len(trains) == 1
    finally:
        trainer.cleanup()


def test_on_step_end_monitoring_exception_degrades(tmp_path):
    """If spike_detector.update raises, on_step_end must only warn: no
    exception propagates and the train event is still written (recorder is
    independent of monitoring)."""
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir))
    try:
        trainer.model = _tiny_model_with_grads()

        def _raise_update(loss, step):
            raise RuntimeError("detector update failed")

        trainer.spike_detector.update = _raise_update

        # Must not raise.
        trainer.on_step_end(step=3, loss=3.0, lr=6e-4)

        events = _events(out_dir)
        trains = [e for e in events if e["type"] == "train"]
        assert len(trains) == 1
        assert trains[0]["step"] == 3
    finally:
        trainer.cleanup()


def test_on_step_end_grad_compute_exception_degrades(tmp_path):
    """If grad_monitor.compute raises, grad_norm degrades to None (field
    omitted from the train event) and on_step_end does not raise."""
    out_dir = tmp_path / "run42"
    trainer = DummyTrainer(_trainer_args(out_dir))
    try:
        trainer.model = _tiny_model_with_grads()

        def _raise_compute(model, step):
            raise RuntimeError("grad compute failed")

        trainer.grad_monitor.compute = _raise_compute

        trainer.on_step_end(step=5, loss=3.0, lr=6e-4)

        events = _events(out_dir)
        trains = [e for e in events if e["type"] == "train"]
        assert len(trains) == 1
        assert trains[0]["step"] == 5
        # Degraded grad_norm is None -> field must not be written.
        assert "grad_norm" not in trains[0]
    finally:
        trainer.cleanup()


# ---------------------------------------------------------------------------
# CLI flags: --no_monitoring on the five training scripts
# ---------------------------------------------------------------------------

_CLI_CASES = [
    ("training.train_pretrain", []),
    ("training.train_sft", ["--init_from", "dummy.pt"]),
    ("training.train_grpo", ["--init_from", "dummy.pt", "--ref_from", "dummy.pt"]),
    ("training.train_dpo", ["--init_from", "dummy.pt"]),
    (
        "training.iterative_grpo",
        ["--init_from", "dummy.pt", "--ref_from", "dummy.pt"],
    ),
]


@pytest.mark.parametrize("module_name,extra_argv", _CLI_CASES)
def test_cli_flags_expose_no_monitoring(module_name, extra_argv, monkeypatch):
    """Every training script must expose --no_monitoring (store_true, default
    False) alongside the existing --no_metrics."""
    module = importlib.import_module(module_name)
    argv = [module_name.rsplit(".", 1)[-1] + ".py"] + extra_argv
    monkeypatch.setattr(sys, "argv", argv)

    args = module.get_args()

    assert hasattr(args, "no_metrics")
    assert hasattr(args, "no_monitoring")
    assert args.no_monitoring is False
