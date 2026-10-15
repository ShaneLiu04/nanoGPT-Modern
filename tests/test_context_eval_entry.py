"""Contract tests for ``tools/experiments/run_context_eval.py``
(AR-v0.8.5-moe-long-context / T004, E-A-8 four-way RoPE extension comparison
entry; design.md I6). Written against the design contract BEFORE the tool
exists -- TDD Red is the expected state.

Red phase expectations (must fail until the implementation lands):
- subprocess cases (IT6a / IT6b / IT9 / IT11) fail at the ``TOOL_PATH.is_file()``
  guard. The guard is load-bearing for IT6b: a missing script would make bare
  ``python missing.py`` exit 2 and mask the invalid-method expectation.
- in-process cases (IT6c quick-subset, mid-eval abort) fail at the lazy import
  (ModuleNotFoundError: No module named 'tools.experiments.run_context_eval').

Repo-contract anchors:
- Checkpoint fixture mirrors ``training/trainer_base.py::
  load_model_from_checkpoint`` exactly: the state dict lives under the
  ``"model"`` key (trainer_base.py L200, NOT "model_state_dict") and
  ``"config"`` carries ``ModernGPTConfig.to_dict()``; no optimizer / scheduler
  / iter_num fields are required.
- Subprocess runs inject ``PYTHONPATH=REPO_ROOT`` because the E-A-7 sibling
  ``run_ntk_eval.py`` carries no sys.path self-fix and cannot ``import model``
  when invoked bare (verified: ModuleNotFoundError, exit 1).
- The in-process abort case follows the v0.7.5 pattern
  (tests/test_precision_probe.py / tests/test_fit_scaling.py
  ``test_abort_mid_*``): monkeypatch the module-level ``perplexity`` to raise,
  run ``main`` in-process, expect exit code 1 and no partial JSON (atomic
  tmp+rename write). ``main`` is invoked under either repo convention (argv
  list à la precision_probe, or no-arg + sys.argv à la run_ntk_eval), with
  SystemExit normalized to its code.
- IT6c contract: the --quick overrides (lengths=[512, 1024], num_batches=4)
  must be visible at ``get_args()`` output level -- "覆盖用户给的同类参数"
  means the quick semantics is resolved at the parse layer, not buried in
  main().
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from helpers import tiny_modern_config

from model import ModernGPT

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "experiments" / "run_context_eval.py"
SUBPROCESS_TIMEOUT_SECONDS = 300
METHODS = ("naive", "ntk", "pi", "yarn")
ROW_KEYS = {"length", "method", "factor", "nll", "ppl", "tokens", "temperature"}


def run_cli(*args):
    """Subprocess-invoke the entry from the repo root (Red guard: must exist).

    PYTHONPATH=REPO_ROOT is injected so ``import model`` resolves regardless of
    whether the entry carries its own sys.path fixup (run_ntk_eval.py does not).
    """
    assert TOOL_PATH.is_file(), f"tool not implemented yet: {TOOL_PATH}"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(TOOL_PATH), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        env=env,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def _entry():
    """Lazily import the not-yet-implemented entry module (Red-safe)."""
    import tools.experiments.run_context_eval as entry

    return entry


def _get_args(entry, monkeypatch, argv):
    """get_args under either repo convention (no-arg via sys.argv, or argv list)."""
    monkeypatch.setattr(sys, "argv", ["run_context_eval.py", *argv])
    try:
        return entry.get_args()
    except TypeError:
        return entry.get_args(list(argv))


def _run_main(entry, monkeypatch, argv):
    """main() under either repo convention; SystemExit normalized to its code."""
    try:
        try:
            result = entry.main(list(argv))
        except TypeError:
            monkeypatch.setattr(sys, "argv", ["run_context_eval.py", *argv])
            result = entry.main()
    except SystemExit as exc:
        return exc.code
    return result


# ---------------------------------------------------------------------------
# Fixtures: minimal load_model_from_checkpoint-compatible inputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """Tiny checkpoint in the exact format load_model_from_checkpoint reads:
    ``{"model": state_dict, "config": ModernGPTConfig.to_dict()}`` (the weights
    key is "model" per trainer_base.py L200; no extra fields are required)."""
    path = tmp_path_factory.mktemp("ctx_eval") / "tiny_ckpt.pth"
    torch.manual_seed(0)
    cfg = tiny_modern_config(n_layer=1, n_head=2, n_embd=32, block_size=64, vocab_size=256)
    model = ModernGPT(cfg)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, path)
    return path


@pytest.fixture(scope="module")
def tiny_val_bin(tmp_path_factory):
    """Tiny uint16 token file: 4098 random tokens in [0, 256) (>= 1024*4+2)."""
    path = tmp_path_factory.mktemp("ctx_eval") / "val.bin"
    rng = np.random.default_rng(2024)
    tokens = rng.integers(0, 256, size=1024 * 4 + 2).astype(np.uint16)
    tokens.tofile(path)
    return path


# ---------------------------------------------------------------------------
# CLI contract (IT6a / IT6b / IT6c / IT9)
# ---------------------------------------------------------------------------


def test_it6a_help_succeeds():
    """IT6a: --help exits 0 and documents --checkpoint / --methods / --quick."""
    proc = run_cli("--help")
    assert proc.returncode == 0, proc.stderr
    for flag in ("--checkpoint", "--methods", "--quick"):
        assert flag in proc.stdout, f"--help must document {flag}"


def test_it6b_invalid_method_exits_two(tmp_path, tiny_ckpt):
    """IT6b: --methods banana is a usage error: exit 2, stderr non-empty, no JSON."""
    out = tmp_path / "context_eval.json"
    proc = run_cli(
        "--checkpoint",
        str(tiny_ckpt),
        "--output_json",
        str(out),
        "--methods",
        "banana",
    )
    assert proc.returncode == 2
    assert proc.stderr.strip()
    assert not out.exists()


def test_it6c_quick_lengths_subset_of_full_defaults(monkeypatch):
    """IT6c: --quick pins lengths=[512, 1024] (⊆ full defaults, which must cover
    [512, 1024, 2048, 4096]) and num_batches=4, resolved at get_args() level."""
    entry = _entry()
    quick = _get_args(entry, monkeypatch, ["--quick"])
    full = _get_args(entry, monkeypatch, [])
    assert quick.lengths == [512, 1024]
    assert set(quick.lengths) <= set(full.lengths)
    assert {512, 1024, 2048, 4096} <= set(full.lengths)
    assert quick.num_batches == 4


@pytest.mark.parametrize("missing", ["checkpoint", "val_bin"])
def test_it9_missing_input_exits_one(tmp_path, tiny_ckpt, tiny_val_bin, missing):
    """IT9: a missing checkpoint or val_bin must fail loudly with exit 1
    (no silent degradation) and write no JSON."""
    out = tmp_path / "context_eval.json"
    ckpt = str(tmp_path / "missing.ckpt") if missing == "checkpoint" else str(tiny_ckpt)
    val = str(tmp_path / "missing.bin") if missing == "val_bin" else str(tiny_val_bin)
    proc = run_cli(
        "--checkpoint",
        ckpt,
        "--val_bin",
        val,
        "--output_json",
        str(out),
        "--lengths",
        "64",
        "--methods",
        "naive",
        "--train_block",
        "64",
        "--batch_size",
        "1",
        "--num_batches",
        "1",
    )
    assert proc.returncode == 1
    assert proc.stderr.strip()
    assert not out.exists()


# ---------------------------------------------------------------------------
# End-to-end (IT11) and abort atomicity
# ---------------------------------------------------------------------------


def test_it11_end_to_end_four_methods_two_lengths(tmp_path, tiny_ckpt, tiny_val_bin):
    """IT11: full subprocess run over 4 methods x 2 lengths -> exit 0 and 8
    schema-complete rows; factor follows max(1.0, length/train_block); naive
    rows keep temperature 1.0; tokens/nll/ppl follow the E-A-7 window semantics
    (tokens == batch_size * num_batches * length, ppl == exp(nll))."""
    out = tmp_path / "context_eval.json"
    proc = run_cli(
        "--checkpoint",
        str(tiny_ckpt),
        "--val_bin",
        str(tiny_val_bin),
        "--output_json",
        str(out),
        "--lengths",
        "64",
        "128",
        "--methods",
        *METHODS,
        "--batch_size",
        "2",
        "--num_batches",
        "2",
        "--train_block",
        "64",
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(out.read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert len(rows) == len(METHODS) * 2
    assert {(r["length"], r["method"]) for r in rows} == {
        (length, method) for length in (64, 128) for method in METHODS
    }
    for row in rows:
        missing = ROW_KEYS - set(row)
        assert not missing, f"row {row!r} missing schema keys {sorted(missing)}"
        assert row["tokens"] == 2 * 2 * row["length"]
        assert math.isfinite(row["nll"])
        assert row["nll"] > 0.0
        assert row["ppl"] == pytest.approx(math.exp(row["nll"]), rel=1e-6)
        if row["length"] == 128:
            assert row["factor"] == pytest.approx(2.0)
        else:
            assert row["length"] == 64
            assert row["factor"] == pytest.approx(1.0)
        if row["method"] == "naive":
            assert row["temperature"] == pytest.approx(1.0)


def test_abort_mid_eval_no_partial_json(tmp_path, monkeypatch, tiny_ckpt, tiny_val_bin):
    """中断原子性：实验中途异常（in-process 注入模块级 perplexity）→ 退出 1
    且不写部分 JSON（原子写 tmp+rename；仿 tests/test_precision_probe.py /
    tests/test_fit_scaling.py 的 test_abort_mid_* 范式，子进程无法打桩）。"""
    entry = _entry()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-eval failure")

    monkeypatch.setattr(entry, "perplexity", boom)
    out = tmp_path / "context_eval.json"
    rc = _run_main(
        entry,
        monkeypatch,
        [
            "--checkpoint",
            str(tiny_ckpt),
            "--val_bin",
            str(tiny_val_bin),
            "--output_json",
            str(out),
            "--lengths",
            "64",
            "--methods",
            "naive",
            "--train_block",
            "64",
            "--batch_size",
            "1",
            "--num_batches",
            "1",
        ],
    )
    assert rc == 1
    assert not out.exists(), "中断后不得留下部分 JSON"
