"""Tests for Sliding Window Attention integration."""
import pytest
import torch

from model import ModernGPT, ModernGPTConfig
from model.kv_cache_utils import build_sliding_window_mask


def test_build_sliding_window_mask_shape():
    mask = build_sliding_window_mask(8, 4, "cpu")
    assert mask.shape == (8, 8)
    # Causal: upper triangle should be -inf.
    assert torch.isinf(mask[0, 1])
    # Window: position 7 should attend to positions >= 4.
    assert not torch.isinf(mask[7, 4])
    assert torch.isinf(mask[7, 3])


def test_forward_with_sliding_window():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, sliding_window_size=4)
    model = ModernGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss, _ = model(x, targets=x)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None


def test_sliding_window_changes_output():
    cfg_full = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16)
    cfg_win = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, sliding_window_size=2)
    torch.manual_seed(0)
    model_full = ModernGPT(cfg_full)
    torch.manual_seed(0)
    model_win = ModernGPT(cfg_win)
    x = torch.randint(0, cfg_full.vocab_size, (1, 8))
    with torch.no_grad():
        logits_full, _, _ = model_full(x)
        logits_win, _, _ = model_win(x)
    assert not torch.allclose(logits_full, logits_win, atol=1e-4, rtol=1e-4)


def test_sliding_window_cache_bounded():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=16, sliding_window_size=3)
    model = ModernGPT(cfg)
    model.eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        logits_prefill, _, past_kvs = model(prompt, use_cache=True)
        # Decode a few steps; cache length should not exceed window size.
        for _ in range(5):
            next_logits, _, past_kvs = model(
                prompt[:, -1:], use_cache=True, past_kvs=past_kvs, start_pos=0
            )
        k, v = past_kvs[0]
        assert k.shape[2] <= cfg.sliding_window_size
        assert v.shape[2] <= cfg.sliding_window_size


def test_sliding_window_config_serialization():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, sliding_window_size=128)
    d = cfg.to_dict()
    cfg2 = ModernGPTConfig.from_dict(d)
    assert cfg2.sliding_window_size == 128
