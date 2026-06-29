"""Tests for the optional flash-attention backend and its enhanced wrappers.

Covers:
- Basic availability and version introspection
- Standard ``flash_attention`` wrapper (training / prefill)
- ``flash_attention_varlen`` wrapper (packed / variable-length sequences)
- ``flash_attention_with_cache`` wrapper (decode step)
- Model-level integration with ``use_flash_attn=True``
- Graceful fallback when ``flash-attn`` is not installed
"""
import pytest
import torch

from model import ModernGPT, ModernGPTConfig
from model import flash_attention as flash_attn_module


# ---------------------------------------------------------------------------
#  Availability introspection
# ---------------------------------------------------------------------------

def test_flash_attention_reports_availability():
    available = flash_attn_module.is_available()
    assert isinstance(available, bool)


def test_version_info_returns_tuple_or_none():
    ver = flash_attn_module.version_info()
    assert ver is None or isinstance(ver, tuple)


def test_has_varlen_returns_bool():
    assert isinstance(flash_attn_module.has_varlen(), bool)


def test_has_kvcache_returns_bool():
    assert isinstance(flash_attn_module.has_kvcache(), bool)


# ---------------------------------------------------------------------------
#  Standard flash_attention wrapper
# ---------------------------------------------------------------------------

def test_flash_attention_returns_none_when_unavailable():
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    out = flash_attn_module.flash_attention(q, k, v, causal=True)
    if flash_attn_module.is_available():
        assert out is not None
        assert out.shape == q.shape
    else:
        assert out is None


# ---------------------------------------------------------------------------
#  Varlen wrapper
# ---------------------------------------------------------------------------

def test_flash_attention_varlen_returns_none_when_unavailable():
    # Total tokens = 4, n_heads = 2, head_dim = 8
    q = torch.randn(4, 2, 8)
    k = torch.randn(4, 2, 8)
    v = torch.randn(4, 2, 8)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    out = flash_attn_module.flash_attention_varlen(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=2, max_seqlen_k=2,
        causal=True,
    )
    if flash_attn_module.has_varlen():
        assert out is not None
        assert out.shape == q.shape
    else:
        assert out is None


# ---------------------------------------------------------------------------
#  KV-cache decode wrapper
# ---------------------------------------------------------------------------

def test_flash_attention_with_cache_returns_none_when_unavailable():
    B, H, D = 1, 2, 8
    S = 4
    q = torch.randn(B, 1, H, D)
    k_cache = torch.randn(B, S, H, D)
    v_cache = torch.randn(B, S, H, D)
    k_new = torch.randn(B, 1, H, D)
    v_new = torch.randn(B, 1, H, D)
    out = flash_attn_module.flash_attention_with_cache(
        q, k_cache, v_cache, k_new, v_new,
        cache_seqlens=torch.tensor([S], dtype=torch.int32),
        causal=True,
    )
    if flash_attn_module.has_kvcache():
        assert out is not None
        assert out.shape == (B, 1, H, D)
    else:
        assert out is None


# ---------------------------------------------------------------------------
#  Model-level integration
# ---------------------------------------------------------------------------

def test_model_runs_with_flash_flag_when_unavailable():
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=8,
        use_flash_attn=True,
    )
    model = ModernGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _, _ = model(x)
    assert logits.shape == (1, 8, cfg.vocab_size)


def test_flash_flag_config_serialization():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, use_flash_attn=True)
    d = cfg.to_dict()
    cfg2 = ModernGPTConfig.from_dict(d)
    assert cfg2.use_flash_attn is True


def test_flash_attn_with_gqa_model_forward():
    """GQA + use_flash_attn should work (falls back gracefully)."""
    cfg = ModernGPTConfig(
        n_layer=1, n_head=4, n_embd=32, block_size=8,
        n_kv_head=2, use_flash_attn=True,
    )
    model = ModernGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _, _ = model(x)
    assert logits.shape == (1, 8, cfg.vocab_size)


def test_flash_attn_with_gqa_model_generate():
    """GQA + use_flash_attn generate() should work (falls back gracefully)."""
    cfg = ModernGPTConfig(
        n_layer=1, n_head=4, n_embd=32, block_size=16,
        n_kv_head=2, use_flash_attn=True,
    )
    model = ModernGPT(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=4, use_cache=True)
    assert out.shape[0] == 1
    assert out.shape[1] == 8
