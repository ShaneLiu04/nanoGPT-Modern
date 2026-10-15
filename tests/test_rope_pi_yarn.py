"""Tests for AR-v0.8.5-moe-long-context / T002: RotaryEmbedding rope_scaling
extension (PI / YaRN), ModernGPTModelConfig validation, and train CLI flags.

Red phase: the pi/yarn branches, the pydantic field_validator, the
effective_block_size factor extension and the CLI flags do not exist yet.

Expected Red (must fail until Green):
- UT3  test_pi_table_matches_baseline_at_half_position
- UT4  test_yarn_large_factor_finite_monotone_directions
- UT5  test_yarn_ramp_boundary_regions
- UT6  test_pydantic_invalid_rope_type_rejected,
       test_rotary_invalid_rope_type_raises_valueerror
- UT7  test_pydantic_factor_below_one_rejected
- UT9  test_effective_block_size_pi, test_effective_block_size_yarn
- UT10 test_cli_rope_scaling_flags_parse, test_cli_rope_scaling_defaults
- UT13 test_pydantic_train_len_non_positive_rejected
- UT15 test_rope_type_none_without_scaling,
       test_rope_type_reflects_scaling_type

Regression locks (already satisfied by current code; kept to pin behaviour,
accidental passes are expected and fine):
- UT1  default / empty rope_scaling inv_freq (bit-exact)
- UT2  ntk rope_scaling inv_freq (bit-exact)
- UT7  factor=1.0 identity (pi/yarn currently fall through to no-scaling)
- UT8  learned + rope_scaling combo rejection (already type-agnostic)
- UT9  ntk / default effective_block_size
- UT10 factor<1.0 -> SystemExit 2 (currently exits 2 on unrecognized flag)
- UT13 train_len valid passthrough
"""

import math
import sys

import pytest
import torch
from pydantic import ValidationError

from model.modern_gpt import ModernGPTConfig, RotaryEmbedding
from utils.config_models import ModernGPTModelConfig


def _baseline_inv_freq(dim: int, base: float = 10000.0) -> torch.Tensor:
    """Directly computed unscaled inv_freq: 1 / base^(2i/dim)."""
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))


def _pydantic_kwargs(**overrides):
    """Tiny explicit head counts for ModernGPTModelConfig (repo style)."""
    base = dict(n_embd=64, n_head=2, n_kv_head=2)
    base.update(overrides)
    return base


def _yarn_ramp_bounds(dim: int, train_len: int, base: float = 10000.0):
    """NTK-by-parts ramp interval [min(d_corr(1), d_corr(32)), max(...)]."""
    beta_fast, beta_slow = 32, 1

    def d_corr(r: int) -> float:
        return dim * math.log(train_len / (r * 2 * math.pi)) / (2 * math.log(base))

    return min(d_corr(beta_slow), d_corr(beta_fast)), max(d_corr(beta_slow), d_corr(beta_fast))


# ---------------------------------------------------------------------------
# RotaryEmbedding: inv_freq math (UT1-UT5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rope_scaling", [None, {}])
def test_default_rope_scaling_inv_freq_bit_exact(rope_scaling):
    """UT1: rope_scaling=None / {} -> inv_freq == direct base=10000 computation."""
    dim = 64
    rope = RotaryEmbedding(dim, rope_scaling=rope_scaling)
    assert torch.equal(rope.inv_freq, _baseline_inv_freq(dim))


def test_ntk_rope_scaling_inv_freq_bit_exact():
    """UT2: {"type":"ntk","factor":2} -> inv_freq == direct theta=base*2^(d/(d-2))."""
    dim = 64
    base = 10000.0
    factor = 2.0
    rope = RotaryEmbedding(dim, base=base, rope_scaling={"type": "ntk", "factor": factor})
    theta = base * (factor ** (dim / (dim - 2)))
    expected = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    assert torch.equal(rope.inv_freq, expected)


def test_pi_table_matches_baseline_at_half_position():
    """UT3 (AC2): pi factor=2 table at position m == baseline table at m/2.

    Positional interpolation divides inv_freq by factor, so
    m*inv/factor == (m/factor)*inv.  Even m keeps m/2 an exact integer.
    """
    dim = 64
    max_seq_len = 256
    factor = 2.0
    base_rope = RotaryEmbedding(dim, max_seq_len=max_seq_len)
    pi_rope = RotaryEmbedding(
        dim, max_seq_len=max_seq_len, rope_scaling={"type": "pi", "factor": factor}
    )
    x = torch.empty(1, 8, dim)
    cos_base, sin_base = base_rope(x, seq_len=max_seq_len)
    cos_pi, sin_pi = pi_rope(x, seq_len=max_seq_len)
    for m in (0, 64, 100, 126):
        half = m // 2
        assert torch.allclose(cos_pi[m], cos_base[half], atol=1e-6), f"cos mismatch at m={m}"
        assert torch.allclose(sin_pi[m], sin_base[half], atol=1e-6), f"sin mismatch at m={m}"


def test_yarn_large_factor_finite_monotone_directions():
    """UT4 (AC3): yarn factor=16, dim=64, train_len=1024.

    Tables are finite, inv_freq stays strictly decreasing, and the frequency
    direction is correct: i=0 (high frequency) extrapolated unchanged,
    i=dim/2-1 (low frequency) interpolated (divided by factor).
    """
    dim = 64
    factor = 16.0
    train_len = 1024
    max_seq_len = 2048
    yarn = RotaryEmbedding(
        dim,
        max_seq_len=max_seq_len,
        rope_scaling={"type": "yarn", "factor": factor, "train_len": train_len},
    )
    baseline_inv = _baseline_inv_freq(dim)
    x = torch.empty(1, 8, dim)
    cos, sin = yarn(x, seq_len=max_seq_len)
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()
    assert torch.all(yarn.inv_freq[:-1] > yarn.inv_freq[1:])
    assert torch.allclose(yarn.inv_freq[0], baseline_inv[0], atol=1e-7)
    assert torch.allclose(
        yarn.inv_freq[dim // 2 - 1], baseline_inv[dim // 2 - 1] / factor, rtol=1e-5
    )


def test_yarn_ramp_boundary_regions():
    """UT5: yarn ramp regions follow design.md / UT4 direction.

    Small i (high frequency, short wavelength) is extrapolated unchanged;
    large i (low frequency) is interpolated (divided by factor); indices
    inside the ramp are a linear mix strictly between inv/factor and inv.
    With dim=64, train_len=1024, base=10000 the ramp is ~[5.66, 17.70].
    """
    dim = 64
    factor = 16.0
    train_len = 1024
    ramp_lo, ramp_hi = _yarn_ramp_bounds(dim, train_len)
    yarn = RotaryEmbedding(
        dim, rope_scaling={"type": "yarn", "factor": factor, "train_len": train_len}
    )
    baseline_inv = _baseline_inv_freq(dim)
    i_extrap = 0
    i_interp = dim // 2 - 1
    assert torch.allclose(yarn.inv_freq[i_extrap], baseline_inv[i_extrap], atol=1e-7)
    assert torch.allclose(yarn.inv_freq[i_interp], baseline_inv[i_interp] / factor, rtol=1e-5)
    i_mid = int(round((ramp_lo + ramp_hi) / 2))
    assert ramp_lo + 1 < i_mid < ramp_hi - 1, "mid index must sit safely inside the ramp"
    lo = baseline_inv[i_mid].item() / factor
    hi = baseline_inv[i_mid].item()
    margin = 0.01 * (hi - lo)
    assert lo + margin < yarn.inv_freq[i_mid].item() < hi - margin


# ---------------------------------------------------------------------------
# RotaryEmbedding: rope_type introspection (UT15)
# ---------------------------------------------------------------------------


def test_rope_type_none_without_scaling():
    """UT15: rope_type is None when rope_scaling is None or {}."""
    assert RotaryEmbedding(64).rope_type is None
    assert RotaryEmbedding(64, rope_scaling={}).rope_type is None


@pytest.mark.parametrize("scaling_type", ["ntk", "pi", "yarn"])
def test_rope_type_reflects_scaling_type(scaling_type):
    """UT15: rope_type returns the rope_scaling "type" verbatim."""
    rope = RotaryEmbedding(
        64, rope_scaling={"type": scaling_type, "factor": 2.0, "train_len": 1024}
    )
    assert rope.rope_type == scaling_type


def test_rotary_invalid_rope_type_raises_valueerror():
    """UT6 (AC4): RotaryEmbedding with type="banana" -> ValueError listing legal values."""
    with pytest.raises(ValueError) as excinfo:
        RotaryEmbedding(64, rope_scaling={"type": "banana", "factor": 2.0})
    msg = str(excinfo.value)
    for legal in ("ntk", "pi", "yarn"):
        assert legal in msg, f"error message must list legal type {legal!r}: {msg}"


# ---------------------------------------------------------------------------
# ModernGPTModelConfig: rope_scaling validation (UT6, UT7, UT8, UT13)
# ---------------------------------------------------------------------------


def test_pydantic_invalid_rope_type_rejected():
    """UT6 (AC4): config type="banana" -> ValidationError listing legal values."""
    with pytest.raises(ValidationError) as excinfo:
        ModernGPTModelConfig(**_pydantic_kwargs(rope_scaling={"type": "banana", "factor": 2.0}))
    msg = str(excinfo.value)
    for legal in ("ntk", "pi", "yarn"):
        assert legal in msg, f"error message must list legal type {legal!r}: {msg}"


def test_pydantic_factor_below_one_rejected():
    """UT7: config factor=0.5 -> ValidationError (factor must be >= 1.0)."""
    with pytest.raises(ValidationError):
        ModernGPTModelConfig(**_pydantic_kwargs(rope_scaling={"type": "pi", "factor": 0.5}))


def test_factor_one_is_valid_identity():
    """UT7: factor=1.0 is accepted by config and behaves exactly like no scaling."""
    cfg = ModernGPTModelConfig(**_pydantic_kwargs(rope_scaling={"type": "pi", "factor": 1.0}))
    assert cfg.rope_scaling["factor"] == 1.0
    dim = 64
    expected = _baseline_inv_freq(dim)
    for scaling_type in ("pi", "yarn"):
        rope = RotaryEmbedding(
            dim, rope_scaling={"type": scaling_type, "factor": 1.0, "train_len": 1024}
        )
        assert torch.allclose(rope.inv_freq, expected, atol=1e-7), scaling_type


def test_pydantic_learned_with_pi_scaling_rejected():
    """UT8: position_encoding="learned" + pi rope_scaling -> ValidationError (regression
    lock: the combo check is already type-agnostic and must keep mentioning learned)."""
    with pytest.raises(ValidationError) as excinfo:
        ModernGPTModelConfig(
            **_pydantic_kwargs(
                position_encoding="learned",
                rope_scaling={"type": "pi", "factor": 2.0},
            )
        )
    assert "learned" in str(excinfo.value)


@pytest.mark.parametrize("train_len", [0, -1])
def test_pydantic_train_len_non_positive_rejected(train_len):
    """UT13: config train_len=0 / -1 -> ValidationError (must be int > 0)."""
    with pytest.raises(ValidationError):
        ModernGPTModelConfig(
            **_pydantic_kwargs(rope_scaling={"type": "yarn", "factor": 4.0, "train_len": train_len})
        )


def test_pydantic_train_len_valid_passthrough():
    """UT13: train_len=1024 is legal and passed through untouched."""
    cfg = ModernGPTModelConfig(
        **_pydantic_kwargs(rope_scaling={"type": "yarn", "factor": 4.0, "train_len": 1024})
    )
    assert cfg.rope_scaling["train_len"] == 1024


# ---------------------------------------------------------------------------
# ModernGPTConfig.effective_block_size (UT9)
# ---------------------------------------------------------------------------


def _block_size_cfg(**rope_overrides):
    kwargs = dict(n_layer=2, n_head=2, n_embd=32, block_size=64)
    kwargs.update(rope_overrides)
    return ModernGPTConfig(**kwargs)


def test_effective_block_size_pi():
    """UT9: pi factor=4 -> effective_block_size == block_size * 4 == 256."""
    cfg = _block_size_cfg(rope_scaling={"type": "pi", "factor": 4.0})
    assert cfg.effective_block_size == 256


def test_effective_block_size_yarn():
    """UT9: yarn factor=4 -> effective_block_size == block_size * 4 == 256."""
    cfg = _block_size_cfg(rope_scaling={"type": "yarn", "factor": 4.0})
    assert cfg.effective_block_size == 256


def test_effective_block_size_ntk_and_default_regression():
    """UT9 (regression lock): ntk factor=4 -> 256; no scaling -> 64."""
    cfg_ntk = _block_size_cfg(rope_scaling={"type": "ntk", "factor": 4.0})
    assert cfg_ntk.effective_block_size == 256
    cfg_plain = _block_size_cfg()
    assert cfg_plain.effective_block_size == 64


# ---------------------------------------------------------------------------
# CLI: training/train_pretrain.py get_args  (UT10)
#
# CLI parsing lives in the module-level get_args() and is tested directly.
# The args -> ModernGPTConfig(rope_scaling={"type": ..., "factor": ...})
# assembly is coupled inside PretrainTrainer._build_model (heavy data/env/DDP
# setup), so that assertion is deferred:
# TODO(T002-green): test the args->config.rope_scaling assembly once it is
# extractable from _build_model.
# ---------------------------------------------------------------------------


def test_cli_rope_scaling_flags_parse(monkeypatch):
    """UT10: --rope_scaling_type pi --rope_factor 2 parse through."""
    monkeypatch.setattr(
        sys, "argv", ["train_pretrain.py", "--rope_scaling_type", "pi", "--rope_factor", "2"]
    )
    from training.train_pretrain import get_args

    args = get_args()
    assert args.rope_scaling_type == "pi"
    assert args.rope_factor == 2.0


def test_cli_rope_scaling_defaults(monkeypatch):
    """UT10: defaults are rope_scaling_type="none", rope_factor=1.0."""
    monkeypatch.setattr(sys, "argv", ["train_pretrain.py"])
    from training.train_pretrain import get_args

    args = get_args()
    assert args.rope_scaling_type == "none"
    assert args.rope_factor == 1.0


def test_cli_rope_factor_below_one_exits_two(monkeypatch):
    """UT10: --rope_factor 0.5 -> argparse SystemExit with code 2."""
    monkeypatch.setattr(sys, "argv", ["train_pretrain.py", "--rope_factor", "0.5"])
    from training.train_pretrain import get_args

    with pytest.raises(SystemExit) as excinfo:
        get_args()
    assert excinfo.value.code == 2
