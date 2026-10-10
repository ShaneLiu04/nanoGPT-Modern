"""Red-phase tests: PagedKVCacheManager multi-layer correctness (AR-v0.5.0-hpc-readiness T005 / FR4).

Target defects these tests must expose:

1. ``PagedKVCacheManager.get_cache`` (model/paged_kv_cache.py ~L258-263) builds
   every layer's view from ``self._get_layer_blocks(0)`` and then replicates
   that same view for all layers — every layer sees layer 0's K/V.
2. ``PagedKVCacheManager.update`` (~L151-189) appends to the per-sequence block
   table once per layer call with no layer isolation: after a layer-by-layer
   update round the block table contains one extra block per layer, so block
   capacity is exhausted ``n_layers`` times too early and later layers write
   into blocks that reads never reach.
3. End-to-end acceptance semantics (design.md L517): a tiny ModernGPT decoding
   with the paged cache manager must produce the same per-step logits as the
   ring ``KVCacheManager`` path.
"""

import pytest
import torch

from model.modern_gpt import ModernGPT, ModernGPTConfig
from model.kv_cache_utils import KVCacheManager
from model.paged_kv_cache import PagedKVCacheManager
from helpers import tiny_modern_config


def _make_mgr(n_layers=2, max_cache_len=32, block_size=8, batch=1):
    mgr = PagedKVCacheManager(
        n_layers=n_layers,
        n_heads=2,
        n_kv_heads=2,
        head_dim=8,
        max_cache_len=max_cache_len,
        block_size=block_size,
    )
    mgr.init_cache(batch, torch.device("cpu"), torch.float32)
    return mgr


def test_get_cache_returns_per_layer_views():
    """get_cache()[li] must expose layer li's own K/V, not layer 0's view.

    Two layers are updated with different K/V content; after advance(),
    get_cache() must return each layer's own content.  Current bug: every
    layer's entry is ``_get_layer_blocks(0)`` (layer 0's view), so layer 1
    reads layer 0's K/V.
    """
    mgr = _make_mgr(n_layers=2, batch=2)
    B, H, S, D = 2, 2, 6, 8
    k0 = torch.randn(B, H, S, D)
    v0 = torch.randn(B, H, S, D)
    k1 = torch.randn(B, H, S, D)
    v1 = torch.randn(B, H, S, D)
    mgr.update(0, k0, v0)
    mgr.update(1, k1, v1)
    mgr.advance(S)

    cache = mgr.get_cache()
    assert len(cache) == 2
    # Control: layer 0's view is its own content.
    assert torch.allclose(cache[0][0][0], k0, atol=1e-6)
    assert torch.allclose(cache[0][0][1], v0, atol=1e-6)
    # Layer 1's view must be layer 1's content, not a duplicate of layer 0's
    # view.
    assert torch.allclose(
        cache[1][0][0], k1, atol=1e-6
    ), "get_cache() returned layer 0's K view for layer 1 (all layers share _get_layer_blocks(0))"
    assert torch.allclose(
        cache[1][0][1], v1, atol=1e-6
    ), "get_cache() returned layer 0's V view for layer 1 (all layers share _get_layer_blocks(0))"
    # Each layer must be an independent view, not the same object repeated.
    assert cache[0] is not cache[1], "all layers share the same get_cache() object"


def test_block_table_isolated_across_layers():
    """Per-layer update rounds must not inflate the block table per layer.

    A 2-layer manager with 2 blocks of capacity (max_cache_len=32,
    block_size=16) must hold a full 32-token sequence written layer-by-layer
    in two 16-token rounds.  Current bug: every layer's update() call appends
    a new block, so the block table grows to 2 entries after the first round
    and the second round raises "out of blocks" although only half the
    capacity is used.
    """
    mgr = _make_mgr(n_layers=2, max_cache_len=32, block_size=16, batch=1)
    H, D = 2, 8
    round1 = [(torch.randn(1, H, 16, D), torch.randn(1, H, 16, D)) for _ in range(2)]
    round2 = [(torch.randn(1, H, 16, D), torch.randn(1, H, 16, D)) for _ in range(2)]

    for li in range(2):
        mgr.update(li, round1[li][0], round1[li][1])
    mgr.advance(16)

    try:
        for li in range(2):
            mgr.update(li, round2[li][0], round2[li][1])
        mgr.advance(16)
    except RuntimeError as e:
        pytest.fail(
            "update() has no layer isolation: the block table was inflated by "
            "the per-layer update calls and capacity (2 blocks) was exhausted "
            f"after a single 16-token round with 2 layers: {e}"
        )

    assert mgr.cache_len == 32
    contig = mgr.get_cache_contiguous()
    assert contig is not None and len(contig) == 2
    for li in range(2):
        k_hist = torch.cat([round1[li][0], round2[li][0]], dim=2)
        v_hist = torch.cat([round1[li][1], round2[li][1]], dim=2)
        assert contig[li][0].shape[2] == 32
        assert torch.allclose(
            contig[li][0], k_hist, atol=1e-6
        ), f"layer {li} K view does not match its own write history"
        assert torch.allclose(
            contig[li][1], v_hist, atol=1e-6
        ), f"layer {li} V view does not match its own write history"


def test_paged_matches_ring_logits_stepwise():
    """design.md L517 acceptance: paged-cache decode logits == ring-cache logits.

    One tiny 2-layer ModernGPT decodes the same sequence twice — once with the
    ring KVCacheManager, once with PagedKVCacheManager — driven token by token
    via ``model(x, past_kvs=..., use_cache=True, start_pos=...)``.  At every
    decode step the last-position logits must agree within atol=1e-5.
    """
    torch.manual_seed(7)
    cfg = tiny_modern_config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=64,
        vocab_size=100,
        dropout=0.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModernGPT(cfg).to(device).eval()
    head_dim = cfg.n_embd // cfg.n_head

    ring = KVCacheManager(
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        n_kv_heads=cfg.n_kv_head,
        head_dim=head_dim,
        max_cache_len=cfg.block_size,
    )
    paged = PagedKVCacheManager(
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        n_kv_heads=cfg.n_kv_head,
        head_dim=head_dim,
        max_cache_len=cfg.block_size,
        block_size=8,
    )
    dtype = next(model.parameters()).dtype
    dev = torch.device(device)
    ring.init_cache(1, dev, dtype)
    paged.init_cache(1, dev, dtype)

    prompt = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
    n_steps = 6  # prompt(4) + 6 = 10 tokens, crosses the paged block_size=8

    with torch.no_grad():
        # Prefill: both managers ingest the same raw K/V.
        _, _, raw_kvs = model(prompt, use_cache=True, start_pos=0)
        for li in range(cfg.n_layer):
            ring.update(li, raw_kvs[li][0], raw_kvs[li][1])
            paged.update(li, raw_kvs[li][0], raw_kvs[li][1])
        ring.advance(prompt.shape[1])
        paged.advance(prompt.shape[1])

        tok = prompt[:, -1:]
        for step in range(n_steps):
            logits_r, _, kv_r = model(
                tok,
                past_kvs=ring.get_cache(),
                use_cache=True,
                start_pos=ring.start_pos,
            )
            logits_p, _, kv_p = model(
                tok,
                past_kvs=paged.get_cache(),
                use_cache=True,
                start_pos=paged.start_pos,
            )
            assert torch.allclose(
                logits_r[:, -1, :], logits_p[:, -1, :], atol=1e-5
            ), f"paged vs ring KV cache logits diverged at decode step {step}"

            for li in range(cfg.n_layer):
                ring.update(li, kv_r[li][0], kv_r[li][1])
                paged.update(li, kv_p[li][0], kv_p[li][1])
            ring.advance(1)
            paged.advance(1)
            tok = logits_r[:, -1].argmax(dim=-1, keepdim=True)
