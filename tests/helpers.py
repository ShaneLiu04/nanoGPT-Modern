"""Shared tiny-model constructors for the test suite (FR11b).

Imported by ``tests/conftest.py`` (which re-exports them as pytest
fixtures) and directly by individual test modules that prefer plain
function calls over fixture injection.
"""

import torch

from model.modern_gpt import ModernGPT, ModernGPTConfig

TINY_DEFAULTS = dict(
    n_layer=2,
    n_head=2,
    n_embd=64,
    block_size=32,
    vocab_size=100,
    dropout=0.0,
)


def tiny_modern_config(**overrides):
    """Tiny CPU-friendly ModernGPTConfig with de-facto standard shapes."""
    kwargs = dict(TINY_DEFAULTS)
    kwargs.update(overrides)
    return ModernGPTConfig(**kwargs)


def tiny_model(**overrides):
    """Tiny CPU-friendly ModernGPT (eval mode, deterministic init)."""
    torch.manual_seed(0)
    model = ModernGPT(tiny_modern_config(**overrides))
    model.eval()
    return model
