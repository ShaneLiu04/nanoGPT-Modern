"""Structured experiment recording for nanoGPT-Modern (AR-v0.4.0 / FR1).

Provides:
  - ``collect_env``: hardware / software environment snapshot for manifests.
  - ``ExperimentRecorder``: per-event ``metrics.jsonl`` writer plus an
    idempotent ``manifest.json`` (env + config + seed + artifacts + status).
  - ``load_metrics`` / ``summarize_experiment``: lossless pure-Python readers.

Design contract: ``specs/changes/AR-v0.4.0-experiments/design.md`` §4.3.1.
Recording failures (``OSError``) degrade to console warnings and never
interrupt training.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_METRICS_FILENAME = "metrics.jsonl"
_MANIFEST_FILENAME = "manifest.json"
_TERMINAL_STATUSES = ("completed", "failed", "interrupted")


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def collect_env() -> Dict[str, Any]:
    """Snapshot the execution environment for the experiment manifest.

    Returns a dict with GPU name/VRAM/compute-capability, torch/CUDA/Python
    versions, OS description and the current git commit (``"unknown"`` when
    git is unavailable — never raises).
    """
    import torch

    env: Dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "python": sys.version.split()[0],
        "os": platform.platform(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["gpu"] = props.name
        env["vram_gb"] = round(props.total_memory / 1024**3, 2)
        env["sm"] = f"{props.major}.{props.minor}"
    else:
        env["gpu"] = "cpu"
        env["vram_gb"] = 0
        env["sm"] = "none"

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        env["git_commit"] = result.stdout.strip()
    except Exception:
        env["git_commit"] = "unknown"

    return env


def load_metrics(path: str) -> List[Dict[str, Any]]:
    """Parse a ``metrics.jsonl`` file into a list of event dicts.

    A partially-written final line (crash protection) is dropped; malformed
    lines anywhere earlier raise ``ValueError``.
    """
    events: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return events
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    non_empty = [(i, line) for i, line in enumerate(lines) if line.strip()]
    for idx, (pos, line) in enumerate(non_empty):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if idx == len(non_empty) - 1:
                break  # truncated tail from a crashed run
            raise ValueError(f"Malformed metrics line {pos + 1} in {path}")
    return events


def summarize_experiment(exp_dir: str) -> Dict[str, Any]:
    """Summarize a finished experiment directory.

    Prefers ``final_metrics`` from the manifest (authoritative), falling back
    to values computed from the recorded events.
    """
    exp_dir = str(exp_dir)
    manifest: Dict[str, Any] = {}
    manifest_path = os.path.join(exp_dir, _MANIFEST_FILENAME)
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    events = load_metrics(os.path.join(exp_dir, _METRICS_FILENAME))
    evals = [e for e in events if e.get("type") == "eval" and "val_loss" in e]
    trains = [e for e in events if e.get("type") == "train"]

    final_metrics = dict(manifest.get("final_metrics") or {})
    summary: Dict[str, Any] = {
        "exp_id": manifest.get("exp_id", os.path.basename(exp_dir)),
        "status": manifest.get("status"),
        "num_steps_recorded": len(trains),
    }
    if "best_val_loss" in final_metrics:
        summary["best_val_loss"] = final_metrics["best_val_loss"]
    elif evals:
        summary["best_val_loss"] = min(e["val_loss"] for e in evals)
    if "final_val_loss" in final_metrics:
        summary["final_val_loss"] = final_metrics["final_val_loss"]
    elif evals:
        summary["final_val_loss"] = evals[-1]["val_loss"]
    if "wall_time_s" in final_metrics:
        summary["wall_time_s"] = final_metrics["wall_time_s"]

    tps = [e["tokens_per_sec"] for e in trains if "tokens_per_sec" in e]
    if tps:
        summary["mean_tokens_per_sec"] = sum(tps) / len(tps)
    mems = [e["mem_gb"] for e in trains if "mem_gb" in e]
    if mems:
        summary["peak_mem_gb"] = max(mems)
    return summary


class ExperimentRecorder:
    """Append-only metrics recorder plus idempotent experiment manifest.

    Parameters
    ----------
    exp_dir:
        Directory for ``metrics.jsonl`` and ``manifest.json`` (created if
        missing).
    exp_id:
        Unique experiment identifier.
    config:
        Merged training configuration (hyperparameters, seed, ...).
    resume:
        When True the existing ``metrics.jsonl`` is appended to (not
        truncated) and ``resume_count`` is incremented in the manifest.
    """

    def __init__(
        self,
        exp_dir: str,
        exp_id: str,
        config: Optional[Dict[str, Any]] = None,
        resume: bool = False,
    ) -> None:
        self.exp_dir = str(exp_dir)
        self.exp_id = exp_id
        self.config: Dict[str, Any] = dict(config) if config else {}
        self.metrics_path = os.path.join(self.exp_dir, _METRICS_FILENAME)
        self.manifest_path = os.path.join(self.exp_dir, _MANIFEST_FILENAME)

        self._env = collect_env()
        self._extra: Dict[str, Any] = {}
        self._status = "running"
        self._start_ts = _now_iso()
        self._end_ts: Optional[str] = None
        self._resume_count = 0

        # Carry over state when resuming a previous run of the same exp_id.
        if resume and os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    previous = json.load(f)
                self._start_ts = previous.get("start_ts", self._start_ts)
                self._resume_count = int(previous.get("resume_count", 0)) + 1
                for key in ("artifacts", "final_metrics"):
                    if key in previous:
                        self._extra[key] = previous[key]
            except (OSError, json.JSONDecodeError):
                logger.warning(
                    "[ExperimentRecorder] Could not read previous manifest at %s; "
                    "starting fresh manifest.",
                    self.manifest_path,
                )

        try:
            os.makedirs(self.exp_dir, exist_ok=True)
        except OSError:
            logger.warning(
                "[ExperimentRecorder] Cannot create experiment dir %s", self.exp_dir
            )

        if not resume:
            try:
                with open(self.metrics_path, "w", encoding="utf-8"):
                    pass
            except OSError:
                logger.warning(
                    "[ExperimentRecorder] Cannot initialize %s; recording disabled "
                    "for this file (training continues).",
                    self.metrics_path,
                )

        self.write_manifest()

    # ------------------------------------------------------------------
    # metrics.jsonl
    # ------------------------------------------------------------------

    def record(self, event: Dict[str, Any]) -> None:
        """Append one event line; ``ts`` is filled in when missing."""
        event = dict(event)
        event.setdefault("ts", _now_iso())
        try:
            with open(self.metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
                f.flush()
        except OSError as exc:
            logger.warning(
                "[ExperimentRecorder] metrics write failed (training continues): %s",
                exc,
            )

    def record_step(self, step: int, **fields: Any) -> None:
        """Record a training-step event (``type="train"``)."""
        self.record({"step": step, "type": "train", **fields})

    def record_eval(self, step: int, **fields: Any) -> None:
        """Record an evaluation event (``type="eval"``)."""
        self.record({"step": step, "type": "eval", **fields})

    def record_event(self, event: str, step: int, **fields: Any) -> None:
        """Record a monitoring/alert event (``type="event"``)."""
        self.record({"step": step, "type": "event", "event": event, **fields})

    # ------------------------------------------------------------------
    # manifest.json
    # ------------------------------------------------------------------

    def write_manifest(self, **extra: Any) -> str:
        """Rewrite ``manifest.json`` (idempotent); extras are merged across calls."""
        self._extra.update(extra)
        manifest: Dict[str, Any] = {
            "exp_id": self.exp_id,
            "status": self._status,
            "git_commit": self._env.get("git_commit", "unknown"),
            "seed": self.config.get("seed"),
            "start_ts": self._start_ts,
            "resume_count": self._resume_count,
            "epoch_repetition": self._extra.get("epoch_repetition", 0.0),
            "env": self._env,
            "config": self.config,
        }
        if self._end_ts is not None:
            manifest["end_ts"] = self._end_ts
        manifest.update(self._extra)
        manifest["epoch_repetition"] = self._extra.get("epoch_repetition", 0.0)
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning(
                "[ExperimentRecorder] manifest write failed (training continues): %s",
                exc,
            )
        return self.manifest_path

    @property
    def finalized(self) -> bool:
        """True once finalize() has moved the run out of 'running' status."""
        return self._status != "running"

    def wall_time(self) -> Optional[float]:
        """Seconds elapsed since the (original, resume-preserving) start."""
        try:
            start = datetime.fromisoformat(self._start_ts)
            return (datetime.now(timezone.utc) - start).total_seconds()
        except (ValueError, TypeError):
            return None

    def finalize(self, status: str, final_metrics: Dict[str, Any]) -> str:
        """Mark the experiment finished and persist final metrics."""
        if status not in _TERMINAL_STATUSES:
            raise ValueError(
                f"status must be one of {_TERMINAL_STATUSES}, got '{status}'"
            )
        self._status = status
        self._end_ts = _now_iso()
        return self.write_manifest(final_metrics=dict(final_metrics))
