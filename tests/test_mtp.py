"""Tests for Multi-Token Prediction (MTP)."""

import torch

from model import ModernGPT, ModernGPTConfig


def test_mtp_adds_no_parameters_when_disabled():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, n_future=0)
    model = ModernGPT(cfg)
    assert not hasattr(model, "future_heads") or len(model.future_heads) == 0


def test_mtp_adds_heads():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, n_future=2)
    model = ModernGPT(cfg)
    assert hasattr(model, "future_heads")
    assert len(model.future_heads) == 2


def test_mtp_forward_backward():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, n_future=2)
    model = ModernGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss, _ = model(x, targets=x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_mtp_loss_grows_with_n_future():
    """More future heads should produce a strictly positive MTP component."""
    cfg1 = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, n_future=0)
    cfg2 = ModernGPTConfig(
        n_layer=1, n_head=2, n_embd=32, block_size=16, n_future=1, mtp_weight=1.0
    )
    torch.manual_seed(0)
    model1 = ModernGPT(cfg1)
    torch.manual_seed(0)
    model2 = ModernGPT(cfg2)
    x = torch.randint(0, cfg1.vocab_size, (2, 8))
    _, loss1, _ = model1(x, targets=x)
    _, loss2, _ = model2(x, targets=x)
    assert loss2.item() > loss1.item()


def test_mtp_weight_scales_loss():
    cfg_low = ModernGPTConfig(
        n_layer=1, n_head=2, n_embd=32, block_size=16, n_future=1, mtp_weight=0.1
    )
    cfg_high = ModernGPTConfig(
        n_layer=1, n_head=2, n_embd=32, block_size=16, n_future=1, mtp_weight=1.0
    )
    torch.manual_seed(0)
    model_low = ModernGPT(cfg_low)
    torch.manual_seed(0)
    model_high = ModernGPT(cfg_high)
    x = torch.randint(0, cfg_low.vocab_size, (2, 8))
    _, loss_low, _ = model_low(x, targets=x)
    _, loss_high, _ = model_high(x, targets=x)
    assert loss_high.item() > loss_low.item()


def test_mtp_disabled_during_cache():
    """MTP heads should not be computed during cached generation."""
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, n_future=2)
    model = ModernGPT(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        logits, _, past_kvs = model(x, use_cache=True)
    assert logits.shape == (1, 4, cfg.vocab_size)
    assert past_kvs is not None


def test_mtp_config_serialization():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, n_future=3, mtp_weight=0.5)
    d = cfg.to_dict()
    cfg2 = ModernGPTConfig.from_dict(d)
    assert cfg2.n_future == 3
    assert cfg2.mtp_weight == 0.5
