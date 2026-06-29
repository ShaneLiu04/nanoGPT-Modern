"""Tests for generate() torch.compile / CUDA-graph style optimization."""
import os
import warnings

import pytest
import torch


from model.modern_gpt import ModernGPT, ModernGPTConfig


@pytest.mark.parametrize("use_cache", [False, True])
def test_generate_compile_matches_eager(use_cache):
    """generate(compile=True) must produce identical greedy output to eager mode."""
    config = ModernGPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=16,
        vocab_size=100, dropout=0.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModernGPT(config).to(device).eval()

    idx = torch.randint(0, config.vocab_size, (1, 5), device=device)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eager = model.generate(
            idx.clone(), max_new_tokens=10, use_cache=use_cache,
            top_k=1, compile=False,
        )
        compiled = model.generate(
            idx.clone(), max_new_tokens=10, use_cache=use_cache,
            top_k=1, compile=True,
        )

    assert eager.shape == compiled.shape
    assert torch.equal(eager, compiled)


def test_generate_compile_no_op_on_cpu():
    """On CPU, compile=True should gracefully fall back to eager."""
    config = ModernGPTConfig(
        n_layer=1, n_head=2, n_embd=16, block_size=16,
        vocab_size=50, dropout=0.0,
    )
    model = ModernGPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (1, 4))

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.generate(idx, max_new_tokens=5, top_k=1, compile=True)

    assert out.shape[1] == 9


@pytest.mark.parametrize("use_cache", [False, True])
def test_generate_fullgraph_matches_eager(use_cache):
    """generate(compile='fullgraph') must match eager greedy output."""
    config = ModernGPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=32,
        vocab_size=100, dropout=0.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModernGPT(config).to(device).eval()
    idx = torch.randint(0, config.vocab_size, (1, 5), device=device)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eager = model.generate(
            idx.clone(), max_new_tokens=8, use_cache=use_cache,
            top_k=1, compile=False,
        )
        fullgraph = model.generate(
            idx.clone(), max_new_tokens=8, use_cache=use_cache,
            top_k=1, compile="fullgraph",
        )

    assert eager.shape == fullgraph.shape
    assert torch.equal(eager, fullgraph)


def test_generate_fullgraph_no_op_on_cpu():
    """On CPU, compile='fullgraph' should gracefully fall back to eager."""
    config = ModernGPTConfig(
        n_layer=1, n_head=2, n_embd=16, block_size=16,
        vocab_size=50, dropout=0.0,
    )
    model = ModernGPT(config).eval()
    idx = torch.randint(0, config.vocab_size, (1, 4))

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.generate(idx, max_new_tokens=5, top_k=1, compile="fullgraph")

    assert out.shape[1] == 9
