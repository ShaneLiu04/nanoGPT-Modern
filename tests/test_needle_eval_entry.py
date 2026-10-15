"""Contract tests for ``tools/experiments/run_needle_eval.py``
(AR-v0.8.5-moe-long-context / T005, E-A-9 needle-in-haystack grid experiment
entry; design.md I7). Written against the design contract BEFORE the tool
exists -- TDD Red is the expected state.

Red phase expectations (must fail until the implementation lands):
- subprocess cases (IT7 / IT9 / IT10 / IT12) fail at the ``TOOL_PATH.is_file()``
  guard. The guard is load-bearing for IT12: a missing script would make bare
  ``python missing.py`` exit 2 and mask the invalid-argument expectations.
- in-process cases (IT8 tokenizer guard, mid-eval abort, quick-grid contract)
  fail at the lazy import (ModuleNotFoundError: No module named
  'tools.experiments.run_needle_eval').

Repo-contract anchors:
- Checkpoint fixture mirrors ``training/trainer_base.py::
  load_model_from_checkpoint`` exactly: the state dict lives under the
  ``"model"`` key (trainer_base.py L200) and ``"config"`` carries
  ``ModernGPTConfig.to_dict()``.
- Fixture vocab_size is **50257**, not the usual tiny 256: the entry tokenizes
  with tiktoken "gpt2" (50257 tokens) and NeedleEvaluator feeds raw token ids
  into the model, so a 256-vocab embedding would index out of bounds
  (evaluation/needle_in_haystack.py L97-98). 50257 x 32 embeddings ~ 1.6M
  params, verified CPU-friendly (~0.3 s for the 4-cell grid).
- block_size=256 >= max(context_lengths) + max_new_tokens so forward/generate
  stay within ``effective_block_size`` (model/modern_gpt.py L1799-1808).
- Subprocess runs inject ``PYTHONPATH=REPO_ROOT`` (sister convention,
  tests/test_context_eval_entry.py::run_cli).
- IT8 forces a **lazy** tiktoken import inside main/_run: the module-level
  import must succeed at collection time (tiktoken exists on this machine),
  and ``monkeypatch.setitem(sys.modules, "tiktoken", None)`` then makes the
  in-main ``import tiktoken`` raise ImportError, which must be reported as a
  clear "[error] ... tiktoken ..." on stderr with exit 1 -- no bare traceback.
- The abort case monkeypatches the module-level ``NeedleEvaluator`` binding:
  the entry must call it through the module global (i.e. keep
  ``from evaluation.needle_in_haystack import NeedleEvaluator`` at module
  level and reference it unqualified), mirroring the sister seam
  (``entry.perplexity`` in test_context_eval_entry.py).
- Quick-grid contract (IT6c-style) is resolved at ``get_args()`` level:
  --quick pins context_lengths=[512, 1024], depths=[0.0, 0.5, 1.0],
  num_trials=1; full defaults must cover [512, 1024, 2048, 4096] x
  [0.0, 0.25, 0.5, 0.75, 1.0] x 3 trials.
- tiktoken availability was verified on this machine
  (``python -c "import tiktoken"`` -> OK, gpt2 n_vocab=50257); the two
  end-to-end tests still carry ``pytest.importorskip("tiktoken")`` so a
  tiktoken-less CI skips instead of failing confusingly (the entry itself
  must exit 1 with a clear message there -- IT8).
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch
from helpers import tiny_modern_config

from model import ModernGPT

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "experiments" / "run_needle_eval.py"
SUBPROCESS_TIMEOUT_SECONDS = 300
ROPE_METHODS = ("none", "ntk", "pi", "yarn")
PAYLOAD_KEYS = {
    "checkpoint",
    "rope_method",
    "rope_factor",
    "grid_params",
    "summary",
    "scores",
    "raw",
}
SCORE_ROW_KEYS = {"context_length", "depth", "correct"}


def run_cli(*args):
    """Subprocess-invoke the entry from the repo root (Red guard: must exist).

    PYTHONPATH=REPO_ROOT is injected so ``import model`` /
    ``import evaluation.needle_in_haystack`` resolve regardless of whether the
    entry carries its own sys.path fixup (sister convention).
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
    import tools.experiments.run_needle_eval as entry

    return entry


def _get_args(entry, monkeypatch, argv):
    """get_args under either repo convention (no-arg via sys.argv, or argv list)."""
    monkeypatch.setattr(sys, "argv", ["run_needle_eval.py", *argv])
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
            monkeypatch.setattr(sys, "argv", ["run_needle_eval.py", *argv])
            result = entry.main()
    except SystemExit as exc:
        return exc.code
    return result


# ---------------------------------------------------------------------------
# Fixture: minimal load_model_from_checkpoint-compatible checkpoint.
# vocab_size=50257 is load-bearing (gpt2 token ids must fit the embedding);
# block_size=256 covers context_lengths [64, 128] + max_new_tokens margin.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """Tiny checkpoint in the exact format load_model_from_checkpoint reads:
    ``{"model": state_dict, "config": ModernGPTConfig.to_dict()}`` (the weights
    key is "model" per trainer_base.py L200; no extra fields are required)."""
    path = tmp_path_factory.mktemp("needle_eval") / "tiny_ckpt.pth"
    torch.manual_seed(0)
    cfg = tiny_modern_config(n_layer=1, n_head=2, n_embd=32, block_size=256, vocab_size=50257)
    model = ModernGPT(cfg)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, path)
    return path


# ---------------------------------------------------------------------------
# IT7: end-to-end grid run (structure only -- a random model retrieves
# nothing, so accuracy is NOT asserted, only schema/exit-code/grid shape).
# ---------------------------------------------------------------------------


def test_it7_end_to_end_grid_contract(tmp_path, tiny_ckpt):
    """IT7: subprocess run over 2 lengths x 2 depths x 1 trial -> exit 0 and a
    schema-complete payload; scores cover exactly the 4 requested grid cells."""
    pytest.importorskip("tiktoken")
    out = tmp_path / "needle_eval.json"
    proc = run_cli(
        "--checkpoint",
        str(tiny_ckpt),
        "--output_json",
        str(out),
        "--context_lengths",
        "64",
        "128",
        "--depths",
        "0.0",
        "1.0",
        "--num_trials",
        "1",
        "--max_new_tokens",
        "4",
        "--max_context_length",
        "128",
        "--rope_method",
        "none",
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(out.read_text(encoding="utf-8"))
    missing = PAYLOAD_KEYS - set(payload)
    assert not missing, f"payload missing schema keys {sorted(missing)}"
    assert payload["checkpoint"] == str(tiny_ckpt)
    assert payload["rope_method"] == "none"
    assert isinstance(payload["rope_factor"], (int, float))
    assert isinstance(payload["grid_params"], dict)

    # Soft contract: when grid_params echoes the grid, it must match the CLI.
    gp = payload["grid_params"]
    for key, expected in (
        ("context_lengths", [64, 128]),
        ("depths", [0.0, 1.0]),
        ("num_trials", 1),
    ):
        if key in gp:
            assert gp[key] == expected, f"grid_params[{key!r}] mismatch"

    summary = payload["summary"]
    assert {"overall_accuracy", "by_depth", "by_length"} <= set(summary)
    assert isinstance(summary["overall_accuracy"], (int, float))
    assert {float(k) for k in summary["by_depth"]} == {0.0, 1.0}
    assert {int(k) for k in summary["by_length"]} == {64, 128}

    # Grid completeness: 2 lengths x 2 depths x 1 trial = 4 score rows.
    scores = payload["scores"]
    assert len(scores) == 4
    for row in scores:
        row_missing = SCORE_ROW_KEYS - set(row)
        assert not row_missing, f"score row {row!r} missing {sorted(row_missing)}"
    cells = Counter((int(s["context_length"]), float(s["depth"])) for s in scores)
    assert cells == {(length, depth): 1 for length in (64, 128) for depth in (0.0, 1.0)}
    assert len(payload["raw"]) == 4
    for row in payload["raw"]:
        assert {"context_length", "depth", "answer", "correct"} <= set(row)


# ---------------------------------------------------------------------------
# IT8: tiktoken unavailable -> clear error, exit 1 (no bare ImportError).
# ---------------------------------------------------------------------------


def test_it8_tiktoken_unavailable_exits_one(tmp_path, monkeypatch, tiny_ckpt, capsys):
    """IT8: with tiktoken unimportable (sys.modules["tiktoken"] = None), main
    must exit 1, print a clear tiktoken-related error on stderr (not a raw
    traceback), and write no JSON. Requires the entry to import tiktoken
    lazily inside main/_run so the monkeypatch can take effect."""
    pytest.importorskip("tiktoken")  # ensure the guard is what fails, not absence
    entry = _entry()
    out = tmp_path / "needle_eval.json"
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    rc = _run_main(
        entry,
        monkeypatch,
        [
            "--checkpoint",
            str(tiny_ckpt),
            "--output_json",
            str(out),
            "--context_lengths",
            "64",
            "--depths",
            "0.0",
            "--num_trials",
            "1",
            "--max_new_tokens",
            "4",
        ],
    )
    assert rc == 1
    stderr = capsys.readouterr().err
    assert "tiktoken" in stderr.lower()
    assert "Traceback" not in stderr, "must report a clear error, not a bare stack"
    assert not out.exists()


# ---------------------------------------------------------------------------
# IT9 (reused from context_eval semantics): missing checkpoint -> exit 1.
# ---------------------------------------------------------------------------


def test_it9_missing_checkpoint_exits_one(tmp_path):
    """IT9: a missing checkpoint must fail loudly with exit 1 (no silent
    degradation) and write no JSON."""
    out = tmp_path / "needle_eval.json"
    proc = run_cli(
        "--checkpoint",
        str(tmp_path / "missing.ckpt"),
        "--output_json",
        str(out),
        "--context_lengths",
        "64",
        "--depths",
        "0.0",
        "--num_trials",
        "1",
        "--max_new_tokens",
        "4",
    )
    assert proc.returncode == 1
    assert proc.stderr.strip()
    assert not out.exists()


# ---------------------------------------------------------------------------
# IT10: --rope_method yarn --rope_factor 2 must be applied and recorded.
# (The retune math itself is covered by UT11c / tests/test_rope_retune.py;
# this case pins the *entry -> retune* parameter plumbing via the JSON echo.)
# ---------------------------------------------------------------------------


def test_it10_rope_yarn_params_recorded(tmp_path, tiny_ckpt):
    """IT10: --rope_method yarn --rope_factor 2 runs end-to-end (exit 0) and
    the JSON records rope_method=="yarn" and rope_factor==2.0."""
    pytest.importorskip("tiktoken")
    out = tmp_path / "needle_eval_yarn.json"
    proc = run_cli(
        "--checkpoint",
        str(tiny_ckpt),
        "--output_json",
        str(out),
        "--context_lengths",
        "64",
        "--depths",
        "0.0",
        "--num_trials",
        "1",
        "--max_new_tokens",
        "4",
        "--max_context_length",
        "128",
        "--rope_method",
        "yarn",
        "--rope_factor",
        "2",
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rope_method"] == "yarn"
    assert payload["rope_factor"] == pytest.approx(2.0)
    assert len(payload["scores"]) == 1


def test_it10_retune_actually_applied(tmp_path, tiny_ckpt, monkeypatch):
    """IT10 (strengthened, review fix): --rope_method yarn must route through
    ``retune_rope`` before evaluating.  A JSON echo of rope_method alone would
    pass even if the entry skipped the retune, so this spies the module-level
    binding and asserts the call (and that the real retune ran on the model)."""
    pytest.importorskip("tiktoken")
    import tools.experiments.run_needle_eval as entry

    calls = []
    real_retune = entry.retune_rope

    def spy(model, method, factor, max_len, train_block):
        from model.modern_gpt import RotaryEmbedding

        rope = next(m for m in model.modules() if isinstance(m, RotaryEmbedding))
        before = rope.inv_freq.clone()
        real_retune(model, method, factor, max_len, train_block)
        assert not torch.allclose(before, rope.inv_freq), "retune must change inv_freq"
        calls.append((method, float(factor)))

    monkeypatch.setattr(entry, "retune_rope", spy)
    out = tmp_path / "needle_eval_yarn_spy.json"
    rc = entry.main(
        [
            "--checkpoint",
            str(tiny_ckpt),
            "--output_json",
            str(out),
            "--context_lengths",
            "64",
            "--depths",
            "0.0",
            "--num_trials",
            "1",
            "--max_new_tokens",
            "4",
            "--max_context_length",
            "128",
            "--rope_method",
            "yarn",
            "--rope_factor",
            "2",
        ]
    )
    assert rc == 0
    assert calls == [("yarn", 2.0)]


# ---------------------------------------------------------------------------
# IT12: invalid CLI inputs are argparse usage errors -> exit 2.
# ---------------------------------------------------------------------------


def test_it12_invalid_rope_method_exits_two(tmp_path, tiny_ckpt):
    """IT12: --rope_method banana must be rejected by argparse choices (exit 2,
    stderr non-empty) before anything runs; no JSON is written."""
    out = tmp_path / "needle_eval.json"
    proc = run_cli(
        "--checkpoint",
        str(tiny_ckpt),
        "--output_json",
        str(out),
        "--context_lengths",
        "64",
        "--depths",
        "0.0",
        "--rope_method",
        "banana",
    )
    assert proc.returncode == 2
    assert proc.stderr.strip()
    assert not out.exists()


def test_it12_invalid_depths_exits_two(tmp_path, tiny_ckpt):
    """IT12: --depths abc is not a float -> argparse type error, exit 2."""
    out = tmp_path / "needle_eval.json"
    proc = run_cli(
        "--checkpoint",
        str(tiny_ckpt),
        "--output_json",
        str(out),
        "--context_lengths",
        "64",
        "--depths",
        "abc",
    )
    assert proc.returncode == 2
    assert proc.stderr.strip()
    assert not out.exists()


# ---------------------------------------------------------------------------
# Abort atomicity: mid-eval exception -> exit 1, no partial JSON.
# ---------------------------------------------------------------------------


def test_abort_mid_eval_no_partial_json(tmp_path, monkeypatch, tiny_ckpt):
    """中断原子性：实验中途异常（in-process 注入模块级 NeedleEvaluator，其
    evaluate 抛异常）→ 退出 1 且不写部分 JSON（原子写 tmp+rename；仿
    tests/test_context_eval_entry.py::test_abort_mid_eval_no_partial_json，
    子进程无法打桩）。入口必须通过模块级全局名引用 NeedleEvaluator，
    此 monkeypatch 才能生效。"""
    entry = _entry()

    class BoomEvaluator:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, *args, **kwargs):
            raise RuntimeError("simulated mid-eval failure")

    monkeypatch.setattr(entry, "NeedleEvaluator", BoomEvaluator)
    out = tmp_path / "needle_eval.json"
    rc = _run_main(
        entry,
        monkeypatch,
        [
            "--checkpoint",
            str(tiny_ckpt),
            "--output_json",
            str(out),
            "--context_lengths",
            "64",
            "--depths",
            "0.0",
            "--num_trials",
            "1",
            "--max_new_tokens",
            "4",
        ],
    )
    assert rc == 1
    assert not out.exists(), "中断后不得留下部分 JSON"


# ---------------------------------------------------------------------------
# IT6c-style quick contract: resolved at get_args() level.
# ---------------------------------------------------------------------------


def test_quick_grid_contract(monkeypatch):
    """IT6c 同款：--quick 网格 = [512,1024] x [0.0,0.5,1.0] x 1 trial，且在
    get_args() 层可见；全量默认须覆盖 [512,1024,2048,4096] x
    [0.0,0.25,0.5,0.75,1.0] x 3，rope_method 缺省 none。"""
    entry = _entry()
    base = ["--checkpoint", "dummy.ckpt", "--output_json", "dummy.json"]
    quick = _get_args(entry, monkeypatch, [*base, "--quick"])
    full = _get_args(entry, monkeypatch, list(base))

    assert quick.context_lengths == [512, 1024]
    assert quick.depths == [0.0, 0.5, 1.0]
    assert quick.num_trials == 1

    assert {512, 1024, 2048, 4096} <= set(full.context_lengths)
    assert {0.0, 0.25, 0.5, 0.75, 1.0} <= set(full.depths)
    assert full.num_trials == 3
    assert full.rope_method == "none", "缺省不得改 RoPE"
    assert quick.rope_method == "none"
