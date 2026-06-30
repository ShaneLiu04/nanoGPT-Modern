"""Standard benchmark suite integration for nanoGPT-Modern.

Wraps `lm-eval` (EleutherAI lm-evaluation-harness) with pre-configured task sets
and provides a single ``run_benchmark_suite`` entry point.  If ``lm-eval`` is not
installed, the module falls back to a helpful message and returns empty results.

Supported tasks (P1):
  * MMLU (multiple-choice reasoning)
  * ARC (science questions)
  * HellaSwag (commonsense completion)
  * Winogrande (pronoun resolution)
  * HumanEval (code generation)
  * GSM8K (grade-school math)

Usage
-----
>>> from evaluation.benchmark_suite import run_benchmark_suite
>>> results = run_benchmark_suite(
...     checkpoint="out/best_ckpt.pt",
...     tasks=["mmlu", "arc_challenge", "hellaswag"],
...     output_json="eval_results.json",
... )
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from model.modern_gpt import ModernGPT, ModernGPTConfig
from model.baseline_gpt import BaselineGPT, BaselineGPTConfig


TASK_ALIASES = {
    "mmlu": "mmlu",
    "arc": "arc_challenge",
    "arc_challenge": "arc_challenge",
    "arc_easy": "arc_easy",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "humaneval": "humaneval",
    "human_eval": "humaneval",
    "gsm8k": "gsm8k",
}

DEFAULT_TASKS = ["mmlu", "arc_challenge", "hellaswag", "winogrande", "humaneval", "gsm8k"]


def _lm_eval_available() -> bool:
    """Check whether ``lm-eval`` CLI is importable."""
    try:
        import lm_eval  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model_for_eval(checkpoint_path: str, device: Union[str, torch.device] = "cuda"):
    """Load a nanoGPT-Modern checkpoint and return a HuggingFace-compatible wrapper.

    Returns
    -------
    model : nn.Module
        The loaded model placed on ``device`` in eval mode.
    config : ModernGPTConfig or BaselineGPTConfig
    tokenizer : Any
        tiktoken encoding or a minimal fallback.
    """
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = ckpt.get("config")
    model_type = raw_config.pop("model_type", "modern") if isinstance(raw_config, dict) else "modern"
    if model_type == "baseline":
        config = BaselineGPTConfig.from_dict(raw_config) if isinstance(raw_config, dict) else BaselineGPTConfig()
        model = BaselineGPT(config)
    else:
        config = ModernGPTConfig.from_dict(raw_config) if isinstance(raw_config, dict) else ModernGPTConfig()
        model = ModernGPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception:
        tokenizer = None
    return model, config, tokenizer


def run_benchmark_suite(
    checkpoint: str,
    tasks: Optional[List[str]] = None,
    output_json: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 1,
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a suite of standard benchmarks on a nanoGPT-Modern checkpoint.

    Parameters
    ----------
    checkpoint : str
        Path to a ``.pt`` checkpoint produced by the training scripts.
    tasks : list[str] or None
        Task names or aliases.  Defaults to ``DEFAULT_TASKS``.
    output_json : str or None
        If set, write aggregated results to this path.
    device : str
    batch_size : int
        Batch size for lm-eval (most tasks use 1 for correctness).
    num_fewshot : int or None
        Override few-shot count for all tasks.  ``None`` uses task defaults.
    limit : int or None
        If set, limit each task to ``limit`` samples (useful for quick smoke tests).
    model_name : str or None
        Custom model name label in the output JSON.

    Returns
    -------
    results : dict
        Mapping from task name to metric dict (or error string).
    """
    if tasks is None:
        tasks = DEFAULT_TASKS
    tasks = [TASK_ALIASES.get(t.lower(), t.lower()) for t in tasks]

    if not _lm_eval_available():
        # Local fallback: return a structured prompt so users know what to install.
        fallback = {
            "_error": "lm-eval is not installed. Install it with:  pip install lm-eval",
            "checkpoint": checkpoint,
            "requested_tasks": tasks,
            "instructions": (
                "After installing lm-eval, run:\n"
                f"  lm_eval --model hf --model_args pretrained={checkpoint} "
                f"--tasks {','.join(tasks)} --batch_size {batch_size}"
            ),
        }
        if output_json is not None:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(fallback, f, indent=2)
        return fallback

    # Build lm-eval command line.
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={checkpoint},dtype=auto",
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
        "--device", device,
    ]
    if num_fewshot is not None:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if output_json is not None:
        cmd += ["--output_path", os.path.dirname(output_json) or "."]

    print(f"[BenchmarkSuite] Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        stdout = result.stdout
        stderr = result.stderr
    except Exception as e:
        stdout = ""
        stderr = str(e)

    # Parse stdout for known metrics (heuristic fallback when JSON is not emitted).
    parsed: Dict[str, Any] = {}
    for line in stdout.splitlines():
        for task in tasks:
            # Look for lines like "|  mmlu  |acc |0.2345| ..."
            if task in line.lower() and "|" in line:
                parts = [p.strip() for p in line.split("|")]
                parts = [p for p in parts if p]
                if len(parts) >= 3:
                    metric_name = parts[1]
                    try:
                        metric_val = float(parts[2])
                        parsed.setdefault(task, {})[metric_name] = metric_val
                    except ValueError:
                        pass

    aggregated = {
        "checkpoint": checkpoint,
        "model_name": model_name or os.path.basename(checkpoint),
        "tasks": tasks,
        "parsed": parsed,
        "raw_stdout": stdout[-2000:] if len(stdout) > 2000 else stdout,
        "raw_stderr": stderr[-1000:] if len(stderr) > 1000 else stderr,
        "returncode": getattr(result, "returncode", -1),
    }

    if output_json is not None:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)
        print(f"[BenchmarkSuite] Results saved to {output_json}")

    return aggregated


class BenchmarkSuite:
    """Object-oriented wrapper around ``run_benchmark_suite`` for repeated evaluation.

    Example
    -------
    >>> suite = BenchmarkSuite(checkpoint="out/best_ckpt.pt")
    >>> suite.run(tasks=["mmlu", "gsm8k"])
    >>> print(suite.last_results)
    """

    def __init__(self, checkpoint: str, device: str = "cuda", model_name: Optional[str] = None):
        self.checkpoint = checkpoint
        self.device = device
        self.model_name = model_name
        self.last_results: Optional[Dict[str, Any]] = None

    def run(
        self,
        tasks: Optional[List[str]] = None,
        output_json: Optional[str] = None,
        batch_size: int = 1,
        num_fewshot: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        self.last_results = run_benchmark_suite(
            checkpoint=self.checkpoint,
            tasks=tasks,
            output_json=output_json,
            device=self.device,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=limit,
            model_name=self.model_name,
        )
        return self.last_results

    def print_summary(self) -> None:
        """Print a human-readable summary of the last benchmark run."""
        if self.last_results is None:
            print("No results yet. Call run() first.")
            return
        print(f"Benchmark: {self.last_results.get('model_name', 'unknown')}")
        print("-" * 40)
        parsed = self.last_results.get("parsed", {})
        if not parsed:
            print("No metrics parsed.  Check raw_stdout for details.")
        for task, metrics in parsed.items():
            for m, v in metrics.items():
                print(f"  {task:20s} {m:10s} = {v:.4f}")
