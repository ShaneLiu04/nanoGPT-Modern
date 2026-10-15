"""Tests for AR-v0.8.5-moe-long-context / T004: shared RoPE retune module
``tools/experiments/rope_retune.py`` (design.md I5).

Red phase: ``rope_retune.py`` does not exist yet, so every test fails at the
lazy ``_retune_module()`` import (ModuleNotFoundError: No module named
'tools.experiments.rope_retune'). Expected Red set:

- UT11a test_retune_dispatch_naive_restores_baseline
- UT11b test_retune_dispatch_pi
- UT11c test_retune_dispatch_yarn
- UT12  test_retune_invalid_method_lists_legal_domain
- UT14  test_retune_ntk_matches_formula_and_ea7_anchor
- helper direct tests: test_ntk_theta_helper_formula /
  test_pi_inv_freq_helper_divides_baseline /
  test_yarn_inv_freq_helper_delegates_to_t002_reference

Deliberate deviations from the task sheet (documented):

- The tiny model (n_layer=1, n_head=2, n_embd=32) has rotary dim = head_dim =
  16, so per-module expectations are computed via ``rope.dim`` rather than a
  hardcoded 64; the standalone helper tests do use dim=64 as written.
- ``yarn_inv_freq(64, 2.0, 64)[0] == baseline[0]`` from the task sheet is not
  attainable: with train_len=64 the ramp lower bound d_corr(32) is negative
  (≈ -3.98), so index 0 sits inside the YaRN blend region and is NOT
  extrapolated unchanged. The delegation contract is pinned bit-close against
  ``model.modern_gpt._yarn_scaled_inv_freq`` (T002 reference), and the
  high-frequency anchor ``[0] == baseline[0]`` uses train_len=1024 where the
  ramp is ≈ [5.66, 17.70] (same calibration as tests/test_rope_pi_yarn.py).
"""

import math

import pytest
import torch
from helpers import tiny_modern_config

from model import ModernGPT
from model.modern_gpt import CausalSelfAttention, RotaryEmbedding, _yarn_scaled_inv_freq


def _retune_module():
    """Lazily import the shared retune module (Red-safe: fails inside the test)."""
    from tools.experiments import rope_retune

    return rope_retune


def _baseline_inv_freq(dim, base=10000.0):
    """Directly computed unscaled inv_freq: 1 / base^(2i/dim)."""
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))


def _tiny_rope_model():
    """Tiny ModernGPT (n_layer=1, n_head=2, n_embd=32, block_size=64), eval mode."""
    torch.manual_seed(0)
    cfg = tiny_modern_config(n_layer=1, n_head=2, n_embd=32, block_size=64, vocab_size=256)
    model = ModernGPT(cfg)
    model.eval()
    return model


def _rotaries(model):
    """All RotaryEmbedding submodules, in model.modules() order."""
    return [m for m in model.modules() if isinstance(m, RotaryEmbedding)]


def _attentions(model):
    """All CausalSelfAttention submodules, in model.modules() order."""
    return [m for m in model.modules() if isinstance(m, CausalSelfAttention)]


def _warm_rope_caches(model):
    """Force one forward per rotary module so _cos/_sin caches become non-None."""
    for rope in _rotaries(model):
        x = torch.empty(1, 4, rope.dim)
        rope(x, seq_len=4)
        assert rope._cos_cached is not None, "precondition: cache must exist before retune"


# ---------------------------------------------------------------------------
# retune_rope dispatch (UT11a / UT11b / UT11c / UT12 / UT14)
# ---------------------------------------------------------------------------


def test_retune_dispatch_naive_restores_baseline():
    """UT11a: yarn retune first, then naive retune restores the theta=10000
    baseline on every module: inv_freq == direct baseline computation,
    attn_temperature == 1.0, config.rope_scaling is None, caches cleared,
    max_seq_len extended to max_len, config.block_size synced to max_len."""
    rr = _retune_module()
    model = _tiny_rope_model()

    rr.retune_rope(model, "yarn", 2.0, 128, 64)
    assert any(
        a.attn_temperature != 1.0 for a in _attentions(model)
    ), "precondition: yarn retune must have moved the temperature off 1.0"
    _warm_rope_caches(model)

    rr.retune_rope(model, "naive", 1.0, 128, 64)

    for rope in _rotaries(model):
        assert torch.allclose(rope.inv_freq, _baseline_inv_freq(rope.dim), atol=1e-7)
        assert rope.base == 10000.0
        assert rope.max_seq_len == 128
        assert rope._cos_cached is None
        assert rope._sin_cached is None
    for attn in _attentions(model):
        assert attn.attn_temperature == 1.0
    assert model.config.rope_scaling is None
    assert model.config.block_size == 128


def test_retune_dispatch_pi():
    """UT11b: pi retune divides every module's inv_freq by the factor (theta
    untouched), extends max_seq_len to 128, clears the cos/sin caches, keeps
    temperature at 1.0 and syncs config.rope_scaling to the pi entry."""
    rr = _retune_module()
    model = _tiny_rope_model()
    _warm_rope_caches(model)

    rr.retune_rope(model, "pi", 2.0, 128, 64)

    for rope in _rotaries(model):
        expected = rr.pi_inv_freq(rope.dim, 2.0)
        assert torch.allclose(rope.inv_freq, expected, atol=1e-7)
        assert torch.allclose(rope.inv_freq, _baseline_inv_freq(rope.dim) / 2.0, atol=1e-7)
        assert rope.base == 10000.0
        assert rope.max_seq_len == 128
        assert rope._cos_cached is None
        assert rope._sin_cached is None
    for attn in _attentions(model):
        assert attn.attn_temperature == 1.0
    assert model.config.rope_scaling == {"type": "pi", "factor": 2.0}


def test_retune_dispatch_yarn():
    """UT11c: yarn retune applies the T002 YaRN ramp with train_len=train_block
    and sets the YaRN attention temperature 1 + 0.1*ln(factor) on every
    CausalSelfAttention module."""
    rr = _retune_module()
    model = _tiny_rope_model()

    rr.retune_rope(model, "yarn", 2.0, 128, 64)

    expected_temp = 1.0 + 0.1 * math.log(2.0)
    for rope in _rotaries(model):
        expected = _yarn_scaled_inv_freq(rope.dim, 2.0, 64)
        assert torch.allclose(rope.inv_freq, expected, atol=1e-7)
        assert rope.max_seq_len == 128
    for attn in _attentions(model):
        assert attn.attn_temperature == pytest.approx(expected_temp)
    scaling = model.config.rope_scaling
    assert scaling is not None
    assert scaling["type"] == "yarn"
    assert scaling["factor"] == 2.0


def test_retune_invalid_method_lists_legal_domain():
    """UT12: method="banana" -> ValueError whose message lists naive/ntk/pi/yarn."""
    rr = _retune_module()
    model = _tiny_rope_model()
    with pytest.raises(ValueError) as excinfo:
        rr.retune_rope(model, "banana", 2.0, 128, 64)
    msg = str(excinfo.value)
    for legal in ("naive", "ntk", "pi", "yarn"):
        assert legal in msg, f"error message must list legal method {legal!r}: {msg}"


def test_retune_ntk_matches_formula_and_ea7_anchor():
    """UT14: ntk retune reproduces theta = base*factor^(dim/(dim-2)) directly
    and stays numerically identical to the E-A-7 anchor run_ntk_eval.retune_rope
    (same formula as run_ntk_eval.ntk_theta, temperature kept at 1.0)."""
    rr = _retune_module()
    from tools.experiments import run_ntk_eval

    model = _tiny_rope_model()
    anchor = _tiny_rope_model()

    rr.retune_rope(model, "ntk", 2.0, 128, 64)
    run_ntk_eval.retune_rope(anchor, 2.0, 128, 64)

    for rope, anchor_rope in zip(_rotaries(model), _rotaries(anchor)):
        theta = 10000.0 * (2.0 ** (rope.dim / (rope.dim - 2)))
        expected = 1.0 / (theta ** (torch.arange(0, rope.dim, 2).float() / rope.dim))
        assert torch.allclose(rope.inv_freq, expected, atol=1e-7)
        assert torch.allclose(rope.inv_freq, anchor_rope.inv_freq, atol=1e-7)
        assert rr.ntk_theta(rope.dim, 2.0) == pytest.approx(run_ntk_eval.ntk_theta(rope.dim, 2.0))
    for attn in _attentions(model):
        assert attn.attn_temperature == 1.0
    assert model.config.rope_scaling == {"type": "ntk", "factor": 2.0}


# ---------------------------------------------------------------------------
# Helper functions (ntk_theta / pi_inv_freq / yarn_inv_freq)
# ---------------------------------------------------------------------------


def test_ntk_theta_helper_formula():
    """Helper: ntk_theta(dim, factor) == base * factor^(dim/(dim-2)); factor 1 -> base."""
    rr = _retune_module()
    assert rr.ntk_theta(64, 2.0) == pytest.approx(10000.0 * 2.0 ** (64 / 62))
    assert rr.ntk_theta(64, 1.0) == 10000.0


def test_pi_inv_freq_helper_divides_baseline():
    """Helper: pi_inv_freq(64, 2.0) == baseline inv_freq / 2 (theta unchanged)."""
    rr = _retune_module()
    dim = 64
    assert torch.allclose(rr.pi_inv_freq(dim, 2.0), _baseline_inv_freq(dim) / 2.0, atol=1e-7)


def test_yarn_inv_freq_helper_delegates_to_t002_reference():
    """Helper: yarn_inv_freq delegates to model.modern_gpt._yarn_scaled_inv_freq.

    Direction anchors at train_len=1024 (ramp ≈ [5.66, 17.70]): idx 0 is
    extrapolated unchanged, idx dim/2-1 is fully interpolated (divided by the
    factor). See the module docstring for why the task sheet's
    ``yarn_inv_freq(64, 2.0, 64)[0] == baseline[0]`` is not attainable.
    """
    rr = _retune_module()
    dim, factor, train_len = 64, 2.0, 1024
    baseline = _baseline_inv_freq(dim)
    yarn = rr.yarn_inv_freq(dim, factor, train_len)
    assert torch.allclose(yarn, _yarn_scaled_inv_freq(dim, factor, train_len), atol=1e-7)
    assert yarn[0].item() == pytest.approx(baseline[0].item())
    assert yarn[dim // 2 - 1].item() == pytest.approx(
        baseline[dim // 2 - 1].item() / factor, rel=1e-5
    )
