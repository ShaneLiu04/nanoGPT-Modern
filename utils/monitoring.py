"""Training monitoring and alerting utilities for nanoGPT-Modern.

Provides:
  - ``LossSpikeDetector``: sliding-window detection of loss spikes (>3σ)
    with optional LR auto-reduction and logging alerts.
  - ``GradientNormMonitor``: per-layer grad-norm tracking for wandb/TensorBoard.
  - ``MemoryProfiler``: peak-memory breakdown (params / activations / cache / fragmentation).
  - ``ThroughputMonitor``: real-time tokens/s and samples/s measurement.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class LossSpikeDetector:
    """Detect anomalous loss spikes using a sliding-window z-score.

    When the current loss exceeds ``mean + threshold * std`` of the recent
    window, the detector optionally logs a warning and emits a callback that
    can be used to lower the learning rate or trigger a checkpoint.

    Parameters
    ----------
    window_size : int
        Number of recent steps to keep in the rolling buffer.
    threshold : float
        Z-score threshold (default 3.0 = 3σ).
    auto_reduce_lr : bool
        If True, ``on_spike`` will return a suggested LR multiplier < 1.0.
    lr_reduce_factor : float
        Multiplicative factor applied to LR when a spike is detected.
    """

    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 3.0,
        auto_reduce_lr: bool = True,
        lr_reduce_factor: float = 0.5,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.auto_reduce_lr = auto_reduce_lr
        self.lr_reduce_factor = lr_reduce_factor
        self._window: NDArray[np.float64] = np.zeros(window_size, dtype=np.float64)
        self._idx: int = 0
        self._filled: bool = False
        self._spike_count: int = 0

    def update(self, loss: float, step: int) -> Dict[str, Any]:
        """Record a new loss value and check for spikes.

        Returns
        -------
        dict
            Contains ``spike_detected`` (bool), ``z_score`` (float),
            ``lr_multiplier`` (float or None), and ``message`` (str).
        """
        result: Dict[str, Any] = {
            "spike_detected": False,
            "z_score": 0.0,
            "lr_multiplier": None,
            "message": "",
        }

        self._window[self._idx] = loss
        self._idx = (self._idx + 1) % self.window_size
        if self._idx == 0:
            self._filled = True

        if not self._filled and self._idx < 10:
            # Not enough data yet.
            return result

        n = self.window_size if self._filled else self._idx
        vals = self._window[:n]
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1))

        if std > 0.0:
            z_score = (loss - mean) / std
        else:
            z_score = 0.0

        result["z_score"] = round(z_score, 3)
        result["window_mean"] = round(mean, 6)
        result["window_std"] = round(std, 6)

        if z_score > self.threshold and mean > 0.0:
            self._spike_count += 1
            result["spike_detected"] = True
            msg = (
                f"[LossSpikeDetector] Step {step}: loss={loss:.4f} exceeds "
                f"3σ (mean={mean:.4f}, std={std:.4f}, z={z_score:.2f}). "
                f"Spike count={self._spike_count}."
            )
            result["message"] = msg
            logger.warning(msg)

            if self.auto_reduce_lr:
                result["lr_multiplier"] = self.lr_reduce_factor
                logger.warning(
                    "[LossSpikeDetector] Auto-reducing LR by factor %.3f",
                    self.lr_reduce_factor,
                )

        return result

    def reset(self) -> None:
        """Clear the sliding window and spike counter."""
        self._window.fill(0.0)
        self._idx = 0
        self._filled = False
        self._spike_count = 0


class GradientNormMonitor:
    """Track per-layer gradient norms and log them to a wandb/TensorBoard logger.

    Usage
    -----
    Call ``monitor.compute(model)`` after ``loss.backward()`` and (for fp16)
    after ``scaler.unscale_(optimizer)`` so gradients are in fp32.
    """

    def __init__(self, logger_obj: Any, tag_prefix: str = "grad") -> None:
        """Parameters
        ----------
        logger_obj : Logger or None
            An instance of :class:`utils.logging.Logger` (or any object
            with ``log_scalar`` / ``log_histogram``).
        tag_prefix : str
            Prefix for logged metric keys.
        """
        self.logger = logger_obj
        self.tag_prefix = tag_prefix

    def compute(self, model: nn.Module, step: int) -> Dict[str, float]:
        """Compute total and per-layer gradient norms.

        Returns
        -------
        dict
            ``total_norm`` plus per-layer norms keyed by sanitized parameter name.
        """
        total_norm = 0.0
        norms: Dict[str, float] = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                norms[name] = param_norm
                total_norm += param_norm**2
        total_norm = total_norm**0.5

        if self.logger is not None:
            self.logger.log_scalar(f"{self.tag_prefix}/total_norm", total_norm, step)
            if norms:
                import torch

                layer_norms = torch.tensor(list(norms.values()), dtype=torch.float32)
                self.logger.log_histogram(
                    f"{self.tag_prefix}/layer_norms", layer_norms, step
                )

        return {"total_norm": total_norm, **{k: round(v, 6) for k, v in norms.items()}}


class MemoryProfiler:
    """CUDA memory profiler that produces a peak-memory breakdown.

    Wraps ``torch.cuda.memory_summary()`` and extracts key statistics
    into a structured report suitable for logging or JSON export.
    """

    def __init__(self, device: torch.device | int | str = "cuda") -> None:
        self.device = device
        self._peak_allocated: int = 0
        self._peak_reserved: int = 0

    def snapshot(self) -> Dict[str, Any]:
        """Return a structured memory report.

        Returns
        -------
        dict
            Keys: ``allocated_mb``, ``reserved_mb``, ``max_allocated_mb``,
            ``max_reserved_mb``, ``active_mb``, ``inactive_mb``,
            ``pool_fraction`` (active / reserved), and a raw
            ``summary`` string from ``torch.cuda.memory_summary()``.
        """
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}

        torch.cuda.synchronize(self.device)
        alloc = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_alloc = torch.cuda.max_memory_allocated(self.device)
        max_reserved = torch.cuda.max_memory_reserved(self.device)

        self._peak_allocated = max(self._peak_allocated, max_alloc)
        self._peak_reserved = max(self._peak_reserved, max_reserved)

        active = alloc
        inactive = reserved - alloc
        pool_fraction = active / reserved if reserved > 0 else 0.0

        summary = torch.cuda.memory_summary(self.device, abbreviated=True)

        return {
            "allocated_mb": round(alloc / (1024**2), 2),
            "reserved_mb": round(reserved / (1024**2), 2),
            "max_allocated_mb": round(max_alloc / (1024**2), 2),
            "max_reserved_mb": round(max_reserved / (1024**2), 2),
            "active_mb": round(active / (1024**2), 2),
            "inactive_mb": round(inactive / (1024**2), 2),
            "pool_fraction": round(pool_fraction, 4),
            "summary": summary,
        }

    def reset_peak(self) -> None:
        """Reset CUDA peak memory counters."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def print_report(self, prefix: str = "MemoryProfiler") -> None:
        """Pretty-print the current snapshot to stdout."""
        snap = self.snapshot()
        if "error" in snap:
            print(f"[{prefix}] {snap['error']}")
            return
        print(f"[{prefix}] Allocated: {snap['allocated_mb']} MB")
        print(f"[{prefix}] Reserved:  {snap['reserved_mb']} MB")
        print(f"[{prefix}] Peak Allocated: {snap['max_allocated_mb']} MB")
        print(f"[{prefix}] Peak Reserved:  {snap['max_reserved_mb']} MB")
        print(f"[{prefix}] Pool Active Fraction: {snap['pool_fraction']}")


class ThroughputMonitor:
    """Measure training throughput in tokens per second and samples per second.

    Call ``tick()`` every N steps with the number of tokens processed since
    the last tick.  The monitor computes elapsed wall time and derives rates.
    """

    def __init__(self, smoothing: float = 0.9) -> None:
        """Parameters
        ----------
        smoothing : float
            Exponential smoothing factor for the moving average (0 = no smoothing,
            1 = keep previous value).
        """
        self.smoothing = smoothing
        self._last_time: Optional[float] = None
        self._tokens_per_s: float = 0.0
        self._samples_per_s: float = 0.0

    def tick(
        self,
        tokens_since_last: int,
        samples_since_last: int,
    ) -> Dict[str, float]:
        """Record a timing tick and return throughput metrics.

        Parameters
        ----------
        tokens_since_last : int
            Number of tokens (batch_size * seq_len) processed since the last tick.
        samples_since_last : int
            Number of individual sequences processed since the last tick.

        Returns
        -------
        dict
            ``tokens_per_s``, ``samples_per_s``, and their smoothed versions.
        """
        now = time.perf_counter()
        if self._last_time is None or tokens_since_last < 0 or samples_since_last < 0:
            self._last_time = now
            return {
                "tokens_per_s": 0.0,
                "samples_per_s": 0.0,
                "tokens_per_s_smoothed": 0.0,
                "samples_per_s_smoothed": 0.0,
            }

        dt = now - self._last_time
        self._last_time = now

        if dt <= 0.0:
            dt = 1e-6

        raw_tok = tokens_since_last / dt
        raw_sam = samples_since_last / dt

        self._tokens_per_s = (
            self.smoothing * self._tokens_per_s + (1.0 - self.smoothing) * raw_tok
        )
        self._samples_per_s = (
            self.smoothing * self._samples_per_s + (1.0 - self.smoothing) * raw_sam
        )

        return {
            "tokens_per_s": round(raw_tok, 2),
            "samples_per_s": round(raw_sam, 2),
            "tokens_per_s_smoothed": round(self._tokens_per_s, 2),
            "samples_per_s_smoothed": round(self._samples_per_s, 2),
        }

    def reset(self) -> None:
        """Reset the internal timer and moving averages."""
        self._last_time = None
        self._tokens_per_s = 0.0
        self._samples_per_s = 0.0


class MetricsHeartbeatWatchdog(threading.Thread):
    """Heartbeat watchdog over a group's ``metrics.jsonl`` (FR3b).

    Establishes a baseline from the ``ts`` gaps of recent ``train`` events
    and alerts (stderr + optional callback + webhook) when no new event has
    appeared for more than ``timeout_factor`` x baseline — a likely stall
    (design.md §4.2 flow C / boundary E4: alert only, never kill).  Alerts
    are debounced: at most one per watchdog instance.
    """

    def __init__(
        self,
        metrics_path: str,
        group_name: str,
        on_alert: Optional[Callable[[str], None]] = None,
        poll_interval: float = 5.0,
        baseline_window: int = 10,
        timeout_factor: float = 3.0,
    ) -> None:
        super().__init__(daemon=True, name=f"watchdog-{group_name}")
        self.metrics_path = str(metrics_path)
        self.group_name = str(group_name)
        self.on_alert = on_alert
        self.poll_interval = float(poll_interval)
        self.baseline_window = int(baseline_window)
        self.timeout_factor = float(timeout_factor)
        self.alerted = False
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Core check (synchronous, unit-testable)
    # ------------------------------------------------------------------

    def _train_ts(self) -> list:
        """Epoch seconds of all ``train`` events recorded so far."""
        from utils.experiment import load_metrics

        try:
            events = load_metrics(self.metrics_path)
        except (OSError, ValueError):
            return []
        out = []
        for e in events:
            if e.get("type") != "train" or not e.get("ts"):
                continue
            try:
                out.append(datetime.fromisoformat(e["ts"]).timestamp())
            except (ValueError, TypeError):
                continue
        return out

    def check(self, now: float):
        """Run one stall check at wall-clock time ``now`` (epoch seconds).

        Returns the alert message when a stall is detected (first time
        only), otherwise ``None``.
        """
        if self.alerted:
            return None
        ts = self._train_ts()
        if len(ts) < 2:
            return None  # baseline not established yet
        recent = ts[-self.baseline_window :]
        gaps = [b - a for a, b in zip(recent[:-1], recent[1:]) if b - a > 0]
        if not gaps:
            return None
        baseline = sum(gaps) / len(gaps)
        age = now - ts[-1]
        if age <= self.timeout_factor * baseline:
            return None

        self.alerted = True
        message = (
            f"[watchdog] {self.group_name}: training stalled "
            f"(no metrics for {age:.0f}s, baseline {baseline:.1f}s)"
        )
        print(message, file=sys.stderr)
        if self.on_alert is not None:
            try:
                self.on_alert(message)
            except Exception:  # alerting must never break the watchdog
                pass
        try:
            from utils.notify import notify_webhook

            notify_webhook(message)
        except Exception:
            pass
        return message

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            try:
                self.check(time.time())
            except Exception:  # never let polling kill the thread
                pass

    def stop(self) -> None:
        """Signal the polling loop to exit; idempotent."""
        self._stop_event.set()
