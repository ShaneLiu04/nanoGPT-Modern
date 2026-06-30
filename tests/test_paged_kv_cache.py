"""Tests for PagedKVCacheManager."""

import torch

from model import ModernGPT, ModernGPTConfig
from model.paged_kv_cache import PagedKVCacheManager


def test_paged_manager_api_matches_kv_manager():
    cfg = ModernGPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=64)
    mgr = PagedKVCacheManager.from_config(cfg, block_size=16)
    mgr.init_cache(batch_size=2, device="cpu", dtype=torch.float32)
    k = torch.randn(2, cfg.n_kv_head, 10, cfg.n_embd // cfg.n_head)
    v = torch.randn(2, cfg.n_kv_head, 10, cfg.n_embd // cfg.n_head)
    for li in range(cfg.n_layer):
        mgr.update(li, k, v)
    mgr.advance(k.shape[2])
    cache = mgr.get_cache()
    assert len(cache) == cfg.n_layer
    contiguous = mgr.get_cache_contiguous()
    assert contiguous is not None
    assert contiguous[0][0].shape[2] == 10


def test_paged_manager_handles_multiblock_sequence():
    cfg = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=64)
    mgr = PagedKVCacheManager.from_config(cfg, block_size=8)
    mgr.init_cache(batch_size=1, device="cpu", dtype=torch.float32)
    head_dim = cfg.n_embd // cfg.n_head
    k = torch.randn(1, cfg.n_kv_head, 24, head_dim)
    v = torch.randn(1, cfg.n_kv_head, 24, head_dim)
    mgr.update(0, k, v)
    mgr.advance(k.shape[2])
    contiguous = mgr.get_cache_contiguous()
    assert contiguous[0][0].shape == (1, cfg.n_kv_head, 24, head_dim)
    assert contiguous[0][1].shape == (1, cfg.n_kv_head, 24, head_dim)


def test_paged_generation_matches_ring_buffer():
    cfg_ring = ModernGPTConfig(n_layer=1, n_head=2, n_embd=32, block_size=64)
    cfg_paged = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=64,
        use_paged_kv_cache=True,
        kv_cache_block_size=8,
    )
    torch.manual_seed(0)
    model_ring = ModernGPT(cfg_ring)
    torch.manual_seed(0)
    model_paged = ModernGPT(cfg_paged)
    prompt = torch.randint(0, cfg_ring.vocab_size, (1, 16))
    with torch.no_grad():
        # Use top_k=1 to make sampling deterministic (greedy).
        out_ring = model_ring.generate(
            prompt, max_new_tokens=8, use_cache=True, top_k=1
        )
        out_paged = model_paged.generate(
            prompt, max_new_tokens=8, use_cache=True, top_k=1
        )
    assert torch.equal(out_ring, out_paged)


def test_paged_config_serialization():
    cfg = ModernGPTConfig(
        n_layer=1, n_head=2, n_embd=32, use_paged_kv_cache=True, kv_cache_block_size=32
    )
    d = cfg.to_dict()
    cfg2 = ModernGPTConfig.from_dict(d)
    assert cfg2.use_paged_kv_cache is True
    assert cfg2.kv_cache_block_size == 32
