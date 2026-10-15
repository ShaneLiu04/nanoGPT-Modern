"""Shared RoPE retune helpers for context-extension experiments (E-A-8/E-A-9).

Four-way dispatch (naive / ntk / pi / yarn) used by run_context_eval.py and
run_needle_eval.py.  Deliberately separate from run_ntk_eval.py (E-A-7): that
entry's registered command line stays byte-stable; the ntk branch here
reproduces the same theta formula and is locked to the E-A-7 anchor by
tests/test_rope_retune.py::test_retune_ntk_matches_formula_and_ea7_anchor.

YaRN temperature (1 + 0.1*ln(factor)) reuses the existing
``CausalSelfAttention.attn_temperature`` mechanism instead of adding a new
attention code path (design decision D2).
"""

from __future__ import annotations

import math

import torch

from model.modern_gpt import CausalSelfAttention, RotaryEmbedding, _yarn_scaled_inv_freq

LEGAL_METHODS = ("naive", "ntk", "pi", "yarn")


def ntk_theta(dim: int, factor: float, base: float = 10000.0) -> float:
    """NTK-aware RoPE base. factor==1 leaves the original base unchanged."""
    if factor == 1.0:
        return base
    return base * (factor ** (dim / (dim - 2)))


def pi_inv_freq(dim: int, factor: float, base: float = 10000.0) -> torch.Tensor:
    """Position interpolation: m*inv/factor == (m/factor)*inv."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    return inv_freq / factor


def yarn_inv_freq(dim: int, factor: float, train_len: int, base: float = 10000.0) -> torch.Tensor:
    """YaRN NTK-by-parts ramp; delegates to the T002 model reference."""
    return _yarn_scaled_inv_freq(dim, factor, train_len, base=base)


def retune_rope(
    model: torch.nn.Module,
    method: str,
    factor: float,
    max_len: int,
    train_block: int,
) -> None:
    """Point every RoPE table at ``method`` and allow sequences up to ``max_len``.

    - naive: restore the training base, temperature 1.0, no rope_scaling.
    - ntk:   theta = base * factor^(dim/(dim-2)) (E-A-7 formula).
    - pi:    inv_freq /= factor (positions linearly compressed).
    - yarn:  NTK-by-parts ramp + attention temperature 1 + 0.1*ln(factor).

    Applies in place: every RotaryEmbedding gets a new inv_freq, max_seq_len
    and cleared cos/sin caches; every CausalSelfAttention gets the attention
    temperature; ``model.config`` is kept in sync.
    """
    if method not in LEGAL_METHODS:
        raise ValueError(f"method must be one of {'/'.join(LEGAL_METHODS)}, got {method!r}")
    temperature = 1.0 if method != "yarn" else 1.0 + 0.1 * math.log(factor)
    base = float(getattr(model.config, "rope_theta", 10000.0))

    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            if method == "ntk":
                theta = ntk_theta(module.dim, factor, base)
                module.base = theta
                inv = 1.0 / (theta ** (torch.arange(0, module.dim, 2).float() / module.dim))
            elif method == "pi":
                module.base = base
                inv = pi_inv_freq(module.dim, factor, base)
            elif method == "yarn":
                module.base = base
                inv = yarn_inv_freq(module.dim, factor, train_block, base)
            else:  # naive
                module.base = base
                inv = 1.0 / (base ** (torch.arange(0, module.dim, 2).float() / module.dim))
            module.inv_freq.copy_(
                inv.to(device=module.inv_freq.device, dtype=module.inv_freq.dtype)
            )
            module.max_seq_len = max_len
            module._cos_cached = None
            module._sin_cached = None
        elif isinstance(module, CausalSelfAttention):
            module.attn_temperature = temperature

    # Keep the config in sync so downstream consumers (effective_block_size,
    # describe()) agree with the retuned modules.
    if method == "naive":
        model.config.block_size = max_len
        model.config.rope_scaling = None
    else:
        model.config.block_size = train_block
        model.config.rope_scaling = {"type": method, "factor": float(factor)}
    model.config.attn_temperature = temperature
