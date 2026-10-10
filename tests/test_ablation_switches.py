"""Tests for FR3 (AR-v0.4.0-experiments / T003+T004): position_encoding=learned
and ffn_variant=gelu ablation switches. Red phase: the new config fields /
branches do not exist yet."""

import sys

import pytest
import torch
from helpers import tiny_modern_config

from model.modern_gpt import ModernGPT, ModernGPTConfig


def _tiny_config(**overrides):
    """Tiny CPU-friendly config (same shape as tests/test_bugfixes.py)."""
    kwargs = dict(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        vocab_size=100,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return tiny_modern_config(**kwargs)


# ---------------------------------------------------------------------------
# ModernGPTConfig: new fields, validation, defaults
# ---------------------------------------------------------------------------


def test_config_invalid_position_encoding_raises_valueerror():
    """position_encoding must be one of {"rope","learned"}; anything else is a ValueError."""
    cfg = _tiny_config(position_encoding="learned")
    assert cfg.position_encoding == "learned"
    with pytest.raises(ValueError):
        tiny_modern_config(position_encoding="sinusoid")


def test_config_invalid_ffn_variant_raises_valueerror():
    """ffn_variant must be one of {"swiglu","gelu"}; anything else is a ValueError."""
    cfg = _tiny_config(ffn_variant="gelu")
    assert cfg.ffn_variant == "gelu"
    with pytest.raises(ValueError):
        tiny_modern_config(ffn_variant="relu")


def test_config_intermediate_size_gelu_default():
    """gelu + intermediate_size=None -> 4*n_embd (no 128-alignment).

    n_embd=96 distinguishes the two rules: swiglu-aligned would be 256,
    gelu must be 384.
    """
    cfg = tiny_modern_config(n_embd=96, ffn_variant="gelu")
    assert cfg.intermediate_size == 4 * 96  # 384, not 256


def test_config_intermediate_size_swiglu_default_regression():
    """Default (swiglu) intermediate_size alignment rule is unchanged.

    n_embd=64: int(8/3*64)=170 -> aligned up to 256.
    """
    cfg = tiny_modern_config(n_embd=64)
    assert cfg.intermediate_size == 256
    assert cfg.ffn_variant == "swiglu"


def test_config_learned_with_rope_scaling_raises_valueerror():
    """learned absolute PE + rope_scaling is an invalid combination."""
    scaling = {"type": "ntk", "factor": 2.0}
    # rope + scaling remains valid
    cfg_rope = tiny_modern_config(n_embd=64, rope_scaling=scaling)
    assert cfg_rope.rope_scaling == scaling
    with pytest.raises(ValueError):
        tiny_modern_config(n_embd=64, position_encoding="learned", rope_scaling=scaling)


def test_config_to_dict_includes_ablation_fields():
    """to_dict serializes the new fields; defaults are rope/swiglu."""
    d = tiny_modern_config(n_embd=64).to_dict()
    assert d["position_encoding"] == "rope"
    assert d["ffn_variant"] == "swiglu"

    d2 = _tiny_config(position_encoding="learned", ffn_variant="gelu").to_dict()
    assert d2["position_encoding"] == "learned"
    assert d2["ffn_variant"] == "gelu"


def test_config_from_dict_old_checkpoint_defaults():
    """Old checkpoint dicts (no new keys) load with rope/swiglu defaults.

    Simulates an old checkpoint by deleting the two new keys from to_dict(),
    then verifies from_dict round-trip + strict state_dict loading.
    """
    torch.manual_seed(1337)
    cfg = _tiny_config()
    old_dict = cfg.to_dict()
    old_dict.pop("position_encoding", None)
    old_dict.pop("ffn_variant", None)

    cfg2 = ModernGPTConfig.from_dict(old_dict)
    assert cfg2.position_encoding == "rope"
    assert cfg2.ffn_variant == "swiglu"

    model_src = ModernGPT(cfg)
    model_loaded = ModernGPT(cfg2)
    model_loaded.load_state_dict(model_src.state_dict(), strict=True)
    assert model_loaded.config.position_encoding == "rope"
    assert model_loaded.config.ffn_variant == "swiglu"


# ---------------------------------------------------------------------------
# ModernGPT: position_encoding="learned"
# ---------------------------------------------------------------------------


def test_learned_pe_creates_wpe_and_param_delta():
    """learned PE creates transformer.wpe; param delta vs rope == block_size*n_embd."""
    torch.manual_seed(1337)
    cfg_learned = _tiny_config(position_encoding="learned")
    model_learned = ModernGPT(cfg_learned)
    assert hasattr(model_learned.transformer, "wpe")
    assert model_learned.transformer.wpe.weight.shape == (
        cfg_learned.block_size,
        cfg_learned.n_embd,
    )

    cfg_rope = _tiny_config()
    model_rope = ModernGPT(cfg_rope)
    assert not hasattr(model_rope.transformer, "wpe")

    n_learned = sum(p.numel() for p in model_learned.parameters())
    n_rope = sum(p.numel() for p in model_rope.parameters())
    # wte/lm_head weight tying is unaffected; the only delta is wpe.
    assert n_learned - n_rope == cfg_learned.block_size * cfg_learned.n_embd


def test_learned_pe_forward_shape_and_loss():
    """learned PE forward produces correct logits shape / finite loss."""
    torch.manual_seed(1337)
    cfg = _tiny_config(position_encoding="learned")
    model = ModernGPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    targets = torch.randint(0, cfg.vocab_size, (2, 10))
    with torch.no_grad():
        logits, loss, _ = model(idx, targets=targets)
    assert logits.shape == (2, 10, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert loss is not None
    assert loss.dim() == 0
    assert torch.isfinite(loss).all()


def test_learned_pe_kv_cache_generate_consistency():
    """Absolute PE must not couple to start_pos: cache/no-cache generation match.

    Mirrors tests/test_bugfixes.py::test_kv_cache_long_generate_consistency
    (greedy top_k=1, eval mode, no_grad, bitwise torch.equal).
    """
    torch.manual_seed(1337)
    cfg = _tiny_config(position_encoding="learned")
    model = ModernGPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        out_cache = model.generate(idx, max_new_tokens=8, use_cache=True, top_k=1)
        out_nocache = model.generate(idx, max_new_tokens=8, use_cache=False, top_k=1)
    assert out_cache.shape == out_nocache.shape
    assert torch.equal(out_cache, out_nocache)


def test_learned_pe_context_exceeds_block_size_raises_valueerror():
    """Absolute positions (cache prefix included) beyond block_size -> ValueError.

    Uses the cache-prefix path: t=3 <= block_size, but start_pos(30) +
    past_len(30) + t(3) > 32. The rope model under the same scenario must
    NOT raise (no absolute-position limit).
    """
    torch.manual_seed(1337)
    cfg = _tiny_config(position_encoding="learned")
    model = ModernGPT(cfg)
    model.eval()

    idx_prefill = torch.randint(0, cfg.vocab_size, (1, 30))
    with torch.no_grad():
        _, _, raw_kvs = model(idx_prefill, use_cache=True, start_pos=0)
    idx_next = torch.randint(0, cfg.vocab_size, (1, 3))
    with torch.no_grad():
        with pytest.raises(ValueError):
            model(idx_next, past_kvs=raw_kvs, use_cache=True, start_pos=30)

    # Control: rope model, same cache-prefix scenario, no ValueError.
    cfg_rope = _tiny_config()
    model_rope = ModernGPT(cfg_rope)
    model_rope.eval()
    with torch.no_grad():
        _, _, raw_kvs_rope = model_rope(idx_prefill, use_cache=True, start_pos=0)
        logits, _, _ = model_rope(idx_next, past_kvs=raw_kvs_rope, use_cache=True, start_pos=30)
    assert logits.shape == (1, 3, cfg_rope.vocab_size)


# ---------------------------------------------------------------------------
# ModernGPT: ffn_variant="gelu"
# ---------------------------------------------------------------------------


def test_gelu_ffn_per_layer_param_count():
    """gelu FFN params per layer == 2 * n_embd * 4*n_embd (fc + proj, no bias)."""
    torch.manual_seed(1337)
    cfg = _tiny_config(ffn_variant="gelu")
    model = ModernGPT(cfg)
    expected = 2 * cfg.n_embd * (4 * cfg.n_embd)
    for block in model.transformer.h:
        mlp_params = sum(p.numel() for p in block.mlp.parameters())
        assert mlp_params == expected


def test_gelu_forward_shape_and_finite():
    """gelu model forward produces correct logits shape / finite loss."""
    torch.manual_seed(1337)
    cfg = _tiny_config(ffn_variant="gelu")
    model = ModernGPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    targets = torch.randint(0, cfg.vocab_size, (2, 10))
    with torch.no_grad():
        logits, loss, _ = model(idx, targets=targets)
    assert logits.shape == (2, 10, cfg.vocab_size)
    assert torch.isfinite(logits).all()
    assert loss is not None
    assert loss.dim() == 0
    assert torch.isfinite(loss).all()


def test_geluffn_module_interface():
    """GELUFFN: fc(bias=False) -> F.gelu -> proj(bias=False), SwiGLU-compatible output."""
    from model.modern_gpt import GELUFFN  # ImportError in Red phase

    cfg = _tiny_config(ffn_variant="gelu")
    ffn = GELUFFN(cfg)
    x = torch.randn(2, 7, cfg.n_embd)
    out, aux = ffn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # SwiGLU-compatible interface: (out, zero aux tensor)
    assert torch.is_tensor(aux)
    assert aux.numel() == 1
    assert float(aux) == 0.0


# ---------------------------------------------------------------------------
# Pydantic: utils/config_models.py ModernGPTModelConfig
#
# NOTE: imports are lazy (inside each test) because the module currently
# fails to import at all (pre-existing default-validation bug: pydantic v2
# does not run field validators on defaults, so ``FullConfig``'s class-level
# default ``ModernGPTModelConfig()`` trips ``assert n_kv_head is not None``).
# The FR3 contract requires default construction to work (e.g.
# ``ModernGPTModelConfig(n_embd=64, ffn_variant="gelu")``), so this is a
# legitimate Red failure that Green must resolve.  Imports are deliberately
# kept OUTSIDE any ``pytest.raises(ValidationError)`` block to avoid
# spurious passes from the import-time ValidationError.
# ---------------------------------------------------------------------------


def _pydantic_cls():
    """Import ModernGPTModelConfig lazily (module currently fails to import).

    MUST be called outside any ``pytest.raises(ValidationError)`` block so
    the import-time failure is reported as a test error, never swallowed
    as a spurious pass.
    """
    from utils.config_models import ModernGPTModelConfig

    return ModernGPTModelConfig


def _pydantic_kwargs(**overrides):
    """Tiny explicit head counts to sidestep the pre-existing n_kv_head
    default-validation bug, so post-import failures are attributable to
    the FR3 fields under test."""
    base = dict(n_embd=64, n_head=2, n_kv_head=2)
    base.update(overrides)
    return base


def test_pydantic_model_default_construction():
    """ModernGPTModelConfig must be constructible with all defaults.

    The FR3 contract examples (e.g. n_embd=64 + ffn_variant="gelu" only)
    require default n_head/n_kv_head handling to work, so the module must
    import and default-construct cleanly.
    """
    cls = _pydantic_cls()
    cfg = cls()
    assert cfg.n_kv_head == cfg.n_head


def test_pydantic_ablation_field_defaults():
    """ModernGPTModelConfig gains position_encoding/ffn_variant with defaults."""
    cls = _pydantic_cls()
    cfg = cls(**_pydantic_kwargs())
    assert cfg.position_encoding == "rope"
    assert cfg.ffn_variant == "swiglu"


def test_pydantic_invalid_position_encoding_rejected():
    """Non-enum position_encoding -> ValidationError."""
    cls = _pydantic_cls()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        cls(**_pydantic_kwargs(position_encoding="sinusoid"))


def test_pydantic_invalid_ffn_variant_rejected():
    """Non-enum ffn_variant -> ValidationError."""
    cls = _pydantic_cls()
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        cls(**_pydantic_kwargs(ffn_variant="relu"))


def test_pydantic_learned_with_rope_scaling_rejected():
    """learned + rope_scaling -> ValidationError mentioning learned (Chinese hint)."""
    cls = _pydantic_cls()
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as excinfo:
        cls(
            **_pydantic_kwargs(
                position_encoding="learned",
                rope_scaling={"type": "ntk", "factor": 2.0},
            )
        )
    assert "learned" in str(excinfo.value)


def test_pydantic_intermediate_size_follows_ffn_variant():
    """intermediate_size auto-computation switches with ffn_variant.

    n_embd=96: gelu -> 384; swiglu -> int(8/3*96)=256 aligned to 256.
    """
    cls = _pydantic_cls()
    cfg_gelu = cls(**_pydantic_kwargs(n_embd=96, ffn_variant="gelu"))
    assert cfg_gelu.intermediate_size == 384
    cfg_swiglu = cls(**_pydantic_kwargs(n_embd=96))
    assert cfg_swiglu.intermediate_size == 256


# ---------------------------------------------------------------------------
# CLI: training/train_pretrain.py get_args
# ---------------------------------------------------------------------------


def test_cli_ablation_flag_defaults(monkeypatch):
    """get_args exposes --position_encoding/--ffn_variant/--qk_norm/--num_experts."""
    monkeypatch.setattr(sys, "argv", ["train_pretrain.py"])
    from training.train_pretrain import get_args

    args = get_args()
    assert args.position_encoding == "rope"
    assert args.ffn_variant == "swiglu"
    assert args.qk_norm is False
    assert args.num_experts == 1


def test_cli_ablation_flag_explicit_values(monkeypatch):
    """CLI-provided ablation values are parsed through."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_pretrain.py", "--position_encoding", "learned", "--ffn_variant", "gelu"],
    )
    from training.train_pretrain import get_args

    args = get_args()
    assert args.position_encoding == "learned"
    assert args.ffn_variant == "gelu"
