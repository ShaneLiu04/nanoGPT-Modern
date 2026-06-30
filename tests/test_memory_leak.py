"""Memory regression tests for long-sequence generation."""

import pytest
import torch

from model.modern_gpt import ModernGPT, ModernGPTConfig


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for memory test"
)
def test_kv_cache_generation_memory_stable():
    """Repeated long generations should not leak memory beyond normal cache size."""
    config = ModernGPTConfig(
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=512,
        vocab_size=100,
        dropout=0.0,
        n_kv_head=2,
    )
    model = ModernGPT(config).cuda().eval()

    # Warm-up and establish baseline after a full generation.
    with torch.no_grad():
        _ = model.generate(
            torch.randint(0, config.vocab_size, (1, 64), device="cuda"),
            max_new_tokens=256,
            use_cache=True,
        )
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()

    # Run several generations and check memory stays bounded.
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(
                torch.randint(0, config.vocab_size, (1, 64), device="cuda"),
                max_new_tokens=256,
                use_cache=True,
            )
        torch.cuda.synchronize()

    final = torch.cuda.memory_allocated()
    # Allow a small tolerance for allocator fragmentation, but no unbounded growth.
    mb = 1024**2
    growth_mb = (final - baseline) / mb
    print(
        f"memory baseline={baseline/mb:.1f}MB, final={final/mb:.1f}MB, growth={growth_mb:.1f}MB"
    )
    assert (
        growth_mb <= 10.0
    ), f"Memory grew by {growth_mb:.1f}MB after repeated generations"
