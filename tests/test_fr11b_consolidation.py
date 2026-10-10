"""FR11b (T018): load_model consolidation, conftest tiny fixtures and
test_bugfixes.py split.

Covers three acceptance points:

A. The seven per-script ``load_model`` copies collapse into
   ``training.trainer_base.load_model_from_checkpoint`` with
   ``strict``/``eval_mode`` knobs, and every removed copy is asserted
   gone (static) while the unified loader is asserted to behave
   (eval mode, strict handling, round-trip smoke).

B. ``tests/conftest.py`` provides ``tiny_modern_config`` /
   ``tiny_model`` factory fixtures with the de-facto standard tiny
   hyperparameters, pytest markers are registered in pyproject, and
   the manual ``ModernGPTConfig(`` count in tests/ drops below 40
   (FR11-AC3).

C. ``tests/test_bugfixes.py`` (670-line grab-bag) is split by theme
   into focused files, its stale manual runner disappears, and all 18
   test functions are re-collected at their new homes.
"""

import inspect
import re
from pathlib import Path

import pytest
import tomllib
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"


def _read(rel):
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A. load_model consolidation
# ---------------------------------------------------------------------------


def _make_tiny_ckpt(path, extra_state=None, drop_config=False, vocab_size=50):
    from model.modern_gpt import ModernGPT, ModernGPTConfig

    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=vocab_size)
    model = ModernGPT(cfg)
    ckpt = {"model": model.state_dict(), "iter_num": 7}
    if not drop_config:
        ckpt["config"] = {
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 32,
            "block_size": 16,
            "vocab_size": vocab_size,
            "model_type": "modern",
        }
    if extra_state:
        ckpt.update(extra_state)
    torch.save(ckpt, str(path))
    return cfg


def test_unified_loader_has_strict_and_eval_mode():
    from training.trainer_base import load_model_from_checkpoint

    sig = inspect.signature(load_model_from_checkpoint)
    assert "strict" in sig.parameters
    assert "eval_mode" in sig.parameters
    assert sig.parameters["strict"].default is True
    assert sig.parameters["eval_mode"].default is False


def test_unified_loader_eval_mode(tmp_path):
    from model.modern_gpt import ModernGPT
    from training.trainer_base import load_model_from_checkpoint

    ckpt_file = tmp_path / "tiny.pt"
    _make_tiny_ckpt(ckpt_file)

    model, ckpt = load_model_from_checkpoint(str(ckpt_file), device="cpu", eval_mode=True)
    assert isinstance(model, ModernGPT)
    assert model.training is False
    assert ckpt["iter_num"] == 7

    model2, _ = load_model_from_checkpoint(str(ckpt_file), device="cpu", eval_mode=False)
    assert model2.training is True


def test_unified_loader_strict_flag(tmp_path):
    from training.trainer_base import load_model_from_checkpoint

    ckpt_file = tmp_path / "tiny.pt"
    _make_tiny_ckpt(ckpt_file, extra_state={"model": {}})

    with pytest.raises(RuntimeError):
        load_model_from_checkpoint(str(ckpt_file), device="cpu", strict=True)

    model, _ = load_model_from_checkpoint(str(ckpt_file), device="cpu", strict=False)
    assert model is not None


def test_inference_generate_load_model_removed():
    src = _read("inference/generate.py")
    assert "def load_model(" not in src
    assert "load_model_from_checkpoint" in src


def test_run_full_evaluation_load_model_removed():
    src = _read("run_full_evaluation.py")
    assert "def load_model(" not in src
    assert "load_model_from_checkpoint" in src


def test_export_to_onnx_local_loader_removed():
    src = _read("tools/export/export_to_onnx.py")
    assert "def _load_model(" not in src
    assert "load_model_from_checkpoint" in src


def test_benchmark_suite_dead_loader_removed():
    src = _read("evaluation/benchmark_suite.py")
    assert "_load_model_for_eval" not in src


def test_api_server_load_delegates_to_unified(tmp_path, monkeypatch):
    import api_server

    ckpt_file = tmp_path / "tiny.pt"
    _make_tiny_ckpt(ckpt_file)

    calls = []
    real = api_server.load_model_from_checkpoint

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(api_server, "load_model_from_checkpoint", spy)

    state = api_server.ModelState(str(ckpt_file), "modern", "cpu")
    assert calls, "ModelState._load must delegate to the unified loader"
    assert state.model.training is False
    assert state.config is not None
    assert state.config.vocab_size == 50


def test_eval_benchmark_auto_export_uses_unified():
    src = _read("evaluation/eval_benchmark.py")
    assert "load_model_from_checkpoint" in src
    # the copy that loaded the checkpoint by hand must be gone
    assert "torch.load(" not in src


def test_export_gguf_uses_unified_loader():
    src = _read("tools/export/export_gguf.py")
    assert "load_model_from_checkpoint" in src


def test_export_to_hf_uses_unified_loader():
    src = _read("tools/export/export_to_hf.py")
    assert "load_model_from_checkpoint" in src


def test_generate_smoke_via_unified_loader(tmp_path, monkeypatch):
    """inference.generate end-to-end smoke after the consolidation."""
    import sys

    ckpt_file = tmp_path / "tiny.pt"
    # vocab covers ord() values of the fallback tokenizer path
    cfg = _make_tiny_ckpt(ckpt_file, vocab_size=1000)

    # force the ord()/chr() fallback tokenizer: gpt2 token ids would
    # exceed the tiny vocab
    import tiktoken

    def _no_tiktoken(*_a, **_k):
        raise ImportError("tiktoken disabled in this test")

    monkeypatch.setattr(tiktoken, "get_encoding", _no_tiktoken)

    import inference.generate as gen

    monkeypatch_argv = [
        "prog",
        "--checkpoint",
        str(ckpt_file),
        "--model",
        "modern",
        "--prompt",
        "hello",
        "--prompt_len",
        "4",
        "--max_new_tokens",
        "2",
        "--num_samples",
        "1",
        "--device",
        "cpu",
    ]
    old_argv = list(sys.argv)
    sys.argv[:] = monkeypatch_argv
    try:
        out = gen.main()
        assert out is None or isinstance(out, str)
    finally:
        sys.argv[:] = old_argv
    assert cfg is not None


# ---------------------------------------------------------------------------
# B. conftest tiny fixtures + markers + ModernGPTConfig count
# ---------------------------------------------------------------------------


def test_conftest_exists_with_fixtures():
    p = TESTS_DIR / "conftest.py"
    assert p.exists(), "tests/conftest.py is required (FR11)"
    src = p.read_text(encoding="utf-8")
    assert "def tiny_modern_config" in src
    assert "def tiny_model" in src


def test_tiny_fixture_defaults(tiny_modern_config, tiny_model):
    cfg = tiny_modern_config()
    assert cfg.n_layer == 2
    assert cfg.n_head == 2
    assert cfg.n_embd == 64
    assert cfg.block_size == 32
    assert cfg.vocab_size == 100
    assert cfg.dropout == 0.0

    model = tiny_model()
    assert model.config.n_embd == 64

    over = tiny_modern_config(n_embd=128, qk_norm=True)
    assert over.n_embd == 128
    assert over.qk_norm is True


def test_pyproject_registers_markers():
    data = tomllib.loads(_read("pyproject.toml"))
    opts = data["tool"]["pytest"]["ini_options"]
    markers = opts.get("markers", [])
    joined = "\n".join(markers)
    assert "perf" in joined
    assert "slow" in joined


def test_modern_gpt_config_count_below_40():
    """FR11-AC3: fewer than 40 manual ModernGPTConfig( constructions."""
    count = 0
    for p in TESTS_DIR.glob("test_*.py"):
        src = p.read_text(encoding="utf-8")
        count += len(re.findall(r"(?<![\w.])ModernGPTConfig\(", src))
    assert count < 40, count


# ---------------------------------------------------------------------------
# C. test_bugfixes.py split
# ---------------------------------------------------------------------------

BUGFIX_CASES = {
    "test_modern_gpt_kv_cache_dtype": "test_kv_append_layout",
    "test_kv_cache_ring_buffer_order": "test_kv_append_layout",
    "test_kv_cache_sliding_window_eviction": "test_kv_append_layout",
    "test_kv_cache_long_generate_consistency": "test_kv_append_layout",
    "test_kv_cache_set_restore": "test_kv_append_layout",
    "test_grpo_batched_logprobs_consistency": "test_grpo",
    "test_grpo_gradient_accumulation": "test_grpo",
    "test_iterative_grpo_rejection_pool": "test_grpo",
    "test_optimizer_weight_tying_dedup": "test_trainer_base",
    "test_ma_update_and_checkpoint": "test_trainer_base",
    "test_doc_boundary_dataset_resume_offset": "test_packing",
    "test_doc_boundary_dataset_state_dict": "test_packing",
    "test_prepare_shard_writer_dtype_and_index": "test_prepare_data_files",
    "test_prepare_dtype_selection": "test_prepare_data_files",
    "test_arithmetic_dataset_pre_tokenize": "test_arithmetic_data",
    "test_generate_medium_no_division_by_zero": "test_arithmetic_data",
    "test_rule_reward_perfect_partial_and_malformed": "test_rule_reward",
    "test_rule_reward_process_score": "test_rule_reward",
}


def test_bugfixes_file_removed():
    assert not (TESTS_DIR / "test_bugfixes.py").exists()


def test_bugfixes_manual_runner_gone():
    this = Path(__file__).name
    for p in TESTS_DIR.glob("test_*.py"):
        if p.name == this:
            continue  # this test file quotes the pattern itself
        src = p.read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' not in src, p.name


def test_bugfixes_cases_live_at_new_homes():
    import importlib

    modules = {}
    for modname in set(BUGFIX_CASES.values()):
        # tests/ has no __init__.py; pytest prepends it to sys.path, so
        # bare module names resolve at test time.
        modules[modname] = importlib.import_module(modname)
    for case, home in BUGFIX_CASES.items():
        mod = modules[home]
        assert hasattr(mod, case), (home, case)
        assert callable(getattr(mod, case))
