"""Tests for speculative decoding in ModernGPT.generate()."""
import os

import pytest
import torch


from model.modern_gpt import ModernGPT, ModernGPTConfig


@pytest.mark.parametrize("draft_tokens", [1, 2, 4])
def test_speculative_self_draft_matches_target_greedy(draft_tokens):
    """Using the target model as its own draft model must yield greedy output."""
    config = ModernGPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=64,
        vocab_size=100, dropout=0.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModernGPT(config).to(device).eval()
    idx = torch.randint(0, config.vocab_size, (1, 5), device=device)

    with torch.no_grad():
        ref = model.generate(
            idx.clone(), max_new_tokens=12, use_cache=True,
            top_k=1, compile=False,
        )
        spec = model.generate(
            idx.clone(), max_new_tokens=12, use_cache=True,
            top_k=1, draft_model=model, draft_tokens=draft_tokens,
            draft_temperature=0.0, draft_top_k=1,
        )

    assert ref.shape == spec.shape
    assert torch.equal(ref, spec)


def test_speculative_different_draft_runs():
    """A separate draft model should run without crashing and respect length."""
    config = ModernGPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=64,
        vocab_size=100, dropout=0.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = ModernGPT(config).to(device).eval()
    draft = ModernGPT(config).to(device).eval()
    idx = torch.randint(0, config.vocab_size, (1, 5), device=device)

    with torch.no_grad():
        out = target.generate(
            idx.clone(), max_new_tokens=10, use_cache=True,
            top_k=1, draft_model=draft, draft_tokens=3,
            draft_temperature=0.0, draft_top_k=1,
        )

    assert out.shape[1] == idx.shape[1] + 10


def test_speculative_eos_stops():
    """Speculative decoding must stop when eos_token_id is emitted."""
    config = ModernGPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=64,
        vocab_size=100, dropout=0.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModernGPT(config).to(device).eval()
    idx = torch.randint(0, config.vocab_size, (1, 5), device=device)
    eos_token_id = 22

    with torch.no_grad():
        out = model.generate(
            idx.clone(), max_new_tokens=20, use_cache=True,
            top_k=1, draft_model=model, draft_tokens=4,
            draft_temperature=0.0, draft_top_k=1,
            eos_token_id=eos_token_id,
        )

    assert out.shape[1] <= idx.shape[1] + 20
    # Either we stopped at eos or never emitted it.
    generated = out[0, idx.shape[1]:].tolist()
    if eos_token_id in generated:
        assert generated[-1] == eos_token_id


def test_speculative_batched():
    """Speculative decoding supports batch size > 1 via _generate_speculative_batched."""
    config = ModernGPTConfig(
        n_layer=1, n_head=2, n_embd=16, block_size=32,
        vocab_size=50, dropout=0.0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModernGPT(config).to(device).eval()
    idx = torch.randint(0, config.vocab_size, (2, 4), device=device)

    with torch.no_grad():
        out = model.generate(
            idx, max_new_tokens=5, use_cache=True,
            draft_model=model, draft_tokens=2,
        )
    assert out.shape[0] == 2
    assert out.shape[1] == 4 + 5
