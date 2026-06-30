"""PyTorch Profiler and CUDA memory profiling utilities for nanoGPT-Modern.

Provides thin wrappers around ``torch.profiler`` and
``torch.cuda.memory_summary()`` so that training and inference scripts can
capture Chrome traces and peak-memory reports with minimal boilerplate.

Example
-------
>>> from utils.profiler import Profiler
>>> with Profiler(output_dir="profiles", chrome_trace=True) as prof:
...     model(x)
>>> prof.export_chrome_trace("profiles/trace.json")
>>> prof.export_memory_summary("profiles/memory.txt")
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import torch


class Profiler:
    """Context-manager wrapper around ``torch.profiler`` with optional memory stats.

    Parameters
    ----------
    output_dir : str
        Directory where exported traces and summaries are written.
    chrome_trace : bool
        If ``True``, record a full Chrome trace (activities: CPU + CUDA).
    memory_summary : bool
        If ``True``, capture ``torch.cuda.memory_summary()`` at context exit.
    record_shapes : bool
        Passed to ``torch.profiler.profile``.
    profile_memory : bool
        Passed to ``torch.profiler.profile``.
    with_stack : bool
        Passed to ``torch.profiler.profile``.
    with_flops : bool
        Passed to ``torch.profiler.profile``.
    activities : list or None
        If ``None``, defaults to ``[CPU, CUDA]`` when CUDA is available.
    """

    def __init__(
        self,
        output_dir: str = "profiles",
        chrome_trace: bool = True,
        memory_summary: bool = True,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = False,
        activities: Optional[Any] = None,
    ):
        self.output_dir = output_dir
        self.chrome_trace = chrome_trace
        self.memory_summary = memory_summary
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.activities = activities
        self.prof: Optional[Any] = None
        self._start_mem: Optional[int] = None
        self._peak_mem_mb: float = 0.0

    def __enter__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.activities is None:
            if torch.cuda.is_available():
                self.activities = [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            else:
                self.activities = [torch.profiler.ProfilerActivity.CPU]

        if self.chrome_trace:
            self.prof = torch.profiler.profile(
                activities=self.activities,
                record_shapes=self.record_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                with_flops=self.with_flops,
            )
            self.prof.__enter__()
        if torch.cuda.is_available() and self.memory_summary:
            torch.cuda.reset_peak_memory_stats()
            self._start_mem = torch.cuda.memory_allocated()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prof is not None:
            self.prof.__exit__(exc_type, exc_val, exc_tb)
        if torch.cuda.is_available() and self.memory_summary:
            peak = torch.cuda.max_memory_allocated()
            self._peak_mem_mb = (peak - (self._start_mem or 0)) / 1024**2
        return False

    def export_chrome_trace(self, filename: str = "chrome_trace.json") -> str:
        """Export the captured trace to ``output_dir/filename``."""
        if self.prof is None:
            raise RuntimeError("Chrome trace was not enabled")
        path = os.path.join(self.output_dir, filename)
        self.prof.export_chrome_trace(path)
        return path

    def export_memory_summary(self, filename: str = "memory_summary.txt") -> str:
        """Export ``torch.cuda.memory_summary()`` to a text file."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(torch.cuda.memory_summary())
        return path

    def export_stats(self, filename: str = "profiler_stats.json") -> str:
        """Export key profile statistics (self CPU time total, CUDA time total)."""
        if self.prof is None:
            raise RuntimeError("Profiler not enabled")
        path = os.path.join(self.output_dir, filename)
        import json

        stats = self.prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"top_events_table": stats}, f, indent=2)
        return path

    @property
    def peak_mem_mb(self) -> float:
        """Peak GPU memory increase (MB) during the profiling window."""
        return self._peak_mem_mb


class InferenceProfiler:
    """Lightweight profiler for the autoregressive decode loop.

    Records wall-time per token and optional peak memory, without the heavy
    overhead of a full ``torch.profiler`` trace.
    """

    def __init__(self, output_dir: str = "profiles", device: str = "cuda"):
        self.output_dir = output_dir
        self.device = device
        self.times: list[float] = []
        self.peak_mem_mb = 0.0

    def __enter__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        self._start = time.perf_counter()
        return self

    def step(self):
        """Call once per generated token to record incremental timing."""
        now = time.perf_counter()
        self.times.append(now - self._start)
        self._start = now

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            self.peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        return False

    def summary(self) -> dict:
        """Return a dict with median / p95 / max token latency and throughput."""
        if not self.times:
            return {}
        sorted_t = sorted(self.times)
        n = len(sorted_t)
        median = sorted_t[n // 2]
        p95 = sorted_t[int(0.95 * n)]
        total = sum(sorted_t)
        return {
            "tokens": n,
            "total_sec": total,
            "median_ms": median * 1000.0,
            "p95_ms": p95 * 1000.0,
            "max_ms": sorted_t[-1] * 1000.0,
            "tok_per_sec": n / total if total > 0 else 0.0,
            "peak_mem_mb": self.peak_mem_mb,
        }

    def export(self, filename: str = "inference_profile.json") -> str:
        """Save summary to JSON."""
        import json

        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)
        return path
