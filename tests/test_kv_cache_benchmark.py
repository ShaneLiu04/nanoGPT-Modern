"""Performance regression tests for KV-cache decoding.

The absolute speedup of cache vs. no-cache depends heavily on the SDPA backend
that PyTorch selects at runtime (FlashAttention vs. memory-efficient vs. math).
These tests therefore focus on:
  1. Correctness: cache and no-cache produce identical outputs.
  2. Sanity: cache path does not crash and completes in bounded time.
"""
import time

import pytest
import torch

from model.modern_gpt import ModernGPT, ModernGPTConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for timing")
def test_kv_cache_decode_completes_efficiently():
    """Cache path must complete in bounded time for long sequences."""
    config = ModernGPTConfig(
        n_layer=12, n_head=8, n_embd=512, block_size=1024,
        vocab_size=1000, dropout=0.0, n_kv_head=2,
    )
    model = ModernGPT(config).cuda().eval()
    prompt_len = 512
    max_new_tokens = 256

    idx = torch.randint(0, config.vocab_size, (1, prompt_len), device="cuda")

    # Warm-up
    with torch.no_grad():
        model.generate(idx.clone(), max_new_tokens=10, use_cache=True)
        model.generate(idx.clone(), max_new_tokens=10, use_cache=False)

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_cache = model.generate(
            idx.clone(), max_new_tokens=max_new_tokens, use_cache=True, top_k=1
        )
    torch.cuda.synchronize()
    t_cache = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        out_no_cache = model.generate(
            idx.clone(), max_new_tokens=max_new_tokens, use_cache=False, top_k=1
        )
    torch.cuda.synchronize()
    t_no_cache = time.time() - t0

    speedup = t_no_cache / t_cache if t_cache > 0 else float("inf")
    print(f"cache={t_cache:.3f}s, no-cache={t_no_cache:.3f}s, speedup={speedup:.2f}x")

    # Hard correctness: both paths produce the same tokens under greedy decoding.
    # Sampling amplifies tiny numerical differences over long sequences, so we
    # fix the decoding strategy rather than the RNG state.
    assert torch.equal(out_cache, out_no_cache)
    # Soft sanity: neither path hangs or takes an unreasonable amount of time.
    assert t_cache < 60.0, "cache path took too long"
    assert t_no_cache < 60.0, "no-cache path took too long"


def test_kv_cache_outputs_match_no_cache():
    """Cache and no-cache paths must produce identical token outputs."""
    config = ModernGPTConfig(
        n_layer=2, n_head=4, n_embd=128, block_size=128,
        vocab_size=100, dropout=0.0, n_kv_head=2,
    )
    model = ModernGPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (1, 32))

    torch.manual_seed(42)
    out_cache = model.generate(idx, max_new_tokens=64, use_cache=True)

    torch.manual_seed(42)
    out_no_cache = model.generate(idx, max_new_tokens=64, use_cache=False)

    assert torch.equal(out_cache, out_no_cache)
