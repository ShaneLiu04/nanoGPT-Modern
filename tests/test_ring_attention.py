"""Tests for the pure-PyTorch blockwise (Ring) attention fallback."""

import pytest
import torch
import torch.nn.functional as F


from model import ModernGPT, ModernGPTConfig
from model import ring_attention as ring_attn_module
from model.attention_utils import set_attention_backend


def _reference_sdpa(q, k, v, causal=True, scale=None):
    """Reference attention via PyTorch SDPA."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "block_size_q,block_size_kv",
    [(4, 4), (3, 4), (4, 3), (8, 8), (2, 6)],
)
def test_blockwise_attention_matches_sdpa(causal, block_size_q, block_size_kv):
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 8, 16
    q = torch.randn(B, H, T, D, requires_grad=True)
    k = torch.randn(B, H, T, D, requires_grad=True)
    v = torch.randn(B, H, T, D, requires_grad=True)

    ref = _reference_sdpa(q, k, v, causal=causal)
    out = ring_attn_module.blockwise_attention(
        q,
        k,
        v,
        causal=causal,
        block_size_q=block_size_q,
        block_size_kv=block_size_kv,
    )

    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    assert out.abs().max().item() == pytest.approx(ref.abs().max().item(), abs=1e-4)
    assert (ref - out).abs().max().item() < 1e-4

    # Gradient smoke test.
    loss = out.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_blockwise_attention_non_multiple_length():
    torch.manual_seed(1)
    B, H, T, D = 1, 2, 7, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    ref = _reference_sdpa(q, k, v, causal=True)
    out = ring_attn_module.blockwise_attention(
        q, k, v, causal=True, block_size_q=3, block_size_kv=4
    )

    assert out.shape == (B, H, T, D)
    assert (ref - out).abs().max().item() < 1e-4


def test_blockwise_attention_gqa():
    torch.manual_seed(2)
    B, Hq, Hkv, T, D = 1, 6, 2, 8, 8
    q = torch.randn(B, Hq, T, D)
    k = torch.randn(B, Hkv, T, D)
    v = torch.randn(B, Hkv, T, D)

    k_rep = k.repeat_interleave(Hq // Hkv, dim=1)
    v_rep = v.repeat_interleave(Hq // Hkv, dim=1)
    ref = _reference_sdpa(q, k_rep, v_rep, causal=True)
    out = ring_attn_module.blockwise_attention(
        q, k, v, causal=True, block_size_q=4, block_size_kv=4
    )

    assert out.shape == ref.shape
    assert (ref - out).abs().max().item() < 1e-4


def test_blockwise_attention_custom_softmax_scale():
    torch.manual_seed(3)
    q = torch.randn(1, 2, 8, 8)
    k = torch.randn(1, 2, 8, 8)
    v = torch.randn(1, 2, 8, 8)
    scale = 0.5

    ref = _reference_sdpa(q, k, v, causal=True, scale=scale)
    out = ring_attn_module.blockwise_attention(
        q, k, v, causal=True, softmax_scale=scale, block_size_q=4, block_size_kv=4
    )

    assert (ref - out).abs().max().item() < 1e-4


def test_blockwise_attention_availability():
    assert ring_attn_module.is_available() is True


def test_blockwise_attention_rejects_bad_shapes():
    q = torch.randn(1, 2, 8, 8)
    k = torch.randn(1, 3, 8, 8)
    v = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError):
        ring_attn_module.blockwise_attention(q, k, v)

    q2 = torch.randn(1, 2, 8, 8)
    k2 = torch.randn(1, 2, 9, 8)
    v2 = torch.randn(1, 2, 9, 8)
    with pytest.raises(ValueError):
        ring_attn_module.blockwise_attention(q2, k2, v2)


def test_model_runs_with_ring_attention():
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=8,
        n_kv_head=2,
        use_ring_attention=True,
        ring_block_size_q=4,
        ring_block_size_kv=4,
    )
    model = ModernGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _, _ = model(x)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert model.transformer.h[0].attn.use_ring_attention is True


def test_model_ring_attention_matches_sdpa():
    # Force the math backend so SDPA is deterministic and easy to align with.
    try:
        if torch.cuda.is_available():
            set_attention_backend("math")

        cfg = ModernGPTConfig(
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            n_kv_head=2,
            use_ring_attention=True,
            ring_block_size_q=4,
            ring_block_size_kv=4,
        )
        model_ring = ModernGPT(cfg)
        model_ring.eval()

        cfg2 = ModernGPTConfig(
            n_layer=1,
            n_head=2,
            n_embd=32,
            block_size=8,
            n_kv_head=2,
            use_ring_attention=False,
        )
        model_sdpa = ModernGPT(cfg2)
        model_sdpa.load_state_dict(model_ring.state_dict())
        model_sdpa.eval()

        x = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            logits_ring, _, _ = model_ring(x)
            logits_sdpa, _, _ = model_sdpa(x)

        assert logits_ring.shape == logits_sdpa.shape
        assert (logits_ring - logits_sdpa).abs().max().item() < 1e-4
    finally:
        if torch.cuda.is_available():
            # Restore all backends.
            set_attention_backend("auto")


def test_ring_attention_config_serialization():
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        use_ring_attention=True,
        ring_block_size_q=32,
        ring_block_size_kv=64,
    )
    d = cfg.to_dict()
    cfg2 = ModernGPTConfig.from_dict(d)
    assert cfg2.use_ring_attention is True
    assert cfg2.ring_block_size_q == 32
    assert cfg2.ring_block_size_kv == 64
