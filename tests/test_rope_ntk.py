"""Tests for RoPE NTK-aware length extrapolation."""
import math
import pytest
import torch

from model import ModernGPT, ModernGPTConfig
from model.modern_gpt import RotaryEmbedding


def test_ntk_scaling_increases_base():
    dim = 32
    base = 10000.0
    factor = 2.0
    rope = RotaryEmbedding(dim, base=base, rope_scaling={"type": "ntk", "factor": factor})
    expected_base = base * (factor ** (dim / (dim - 2)))
    assert math.isclose(rope.base, expected_base, rel_tol=1e-6)


def test_no_scaling_preserves_base():
    rope = RotaryEmbedding(32, base=10000.0)
    assert rope.base == 10000.0


def test_ntk_allows_longer_context():
    train_len = 16
    factor = 2.0
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=train_len,
        rope_scaling={"type": "ntk", "factor": factor},
    )
    model = ModernGPT(cfg)
    model.eval()
    long_prompt = torch.randint(0, cfg.vocab_size, (1, int(train_len * factor)))
    with torch.no_grad():
        logits, _, _ = model(long_prompt)
    assert logits.shape == (1, train_len * factor, cfg.vocab_size)


def test_unscaled_fails_beyond_block_size():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16)
    model = ModernGPT(cfg)
    long_prompt = torch.randint(0, cfg.vocab_size, (1, 33))
    with pytest.raises(AssertionError):
        model(long_prompt)


def test_config_serialization_roundtrip():
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        rope_theta=500000.0,
        rope_scaling={"type": "ntk", "factor": 4.0},
    )
    d = cfg.to_dict()
    cfg2 = ModernGPTConfig.from_dict(d)
    assert cfg2.rope_theta == 500000.0
    assert cfg2.rope_scaling == {"type": "ntk", "factor": 4.0}
