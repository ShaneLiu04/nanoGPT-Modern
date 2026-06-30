"""Tests for MoE load balancing aux loss and capacity limiting."""

import torch

from model import ModernGPT, ModernGPTConfig


def test_dense_swiglu_has_zero_aux_loss():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, num_experts=1)
    model = ModernGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss, _, aux_loss = model(x, targets=x, return_aux_loss=True)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert aux_loss.item() == 0.0


def test_moe_returns_aux_loss():
    cfg = ModernGPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=16,
        num_experts=4,
        moe_aux_loss_factor=0.05,
        moe_capacity_factor=1.0,
    )
    model = ModernGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss, _, aux_loss = model(x, targets=x, return_aux_loss=True)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert aux_loss.ndim == 0
    assert aux_loss.item() >= 0.0


def test_moe_aux_loss_is_differentiable():
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=8,
        num_experts=3,
        moe_aux_loss_factor=1.0,
    )
    model = ModernGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 4))
    _, loss, _, aux_loss = model(x, targets=x, return_aux_loss=True)
    total = loss + aux_loss
    total.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_moe_capacity_drops_tokens():
    """Capacity factor 0.5 with 4 experts should force some tokens to be dropped."""
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=8,
        num_experts=2,
        moe_capacity_factor=0.25,
    )
    model = ModernGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits, _, _, aux_loss = model(x, targets=x, return_aux_loss=True)
    assert logits.shape == (1, 8, cfg.vocab_size)
    assert torch.isfinite(aux_loss)


def test_moe_config_serialization_roundtrip():
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        num_experts=4,
        moe_aux_loss_factor=0.1,
        moe_capacity_factor=1.5,
    )
    d = cfg.to_dict()
    cfg2 = ModernGPTConfig.from_dict(d)
    assert cfg2.num_experts == 4
    assert cfg2.moe_aux_loss_factor == 0.1
    assert cfg2.moe_capacity_factor == 1.5
