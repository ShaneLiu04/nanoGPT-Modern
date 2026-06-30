"""Attention backend selection and introspection utilities.

PyTorch's ``F.scaled_dot_product_attention`` automatically picks among
``flash_attention``, ``memory_efficient`` and ``math`` backends.  These helpers
let users force a specific backend for reproducibility or debugging, and print
which backend is actually being used.

This module also provides :func:`probe_gqa_sdpa_support`, a runtime probe that
determines whether the active SDPA stack can compute Grouped Query Attention
without materialising the KV heads to the full query-head count.  Two flavours
are probed:

* **raw broadcast** — passing ``q``/``k``/``v`` with different head counts
  directly (supported natively by some fused backends on recent PyTorch).
* **grouped broadcast** — reshaping Q to ``[B, n_kv, rep, T, D]`` and KV to
  ``[B, n_kv, 1, T, D]`` so SDPA broadcasts over the ``rep`` dimension.  This
  works on every backend we have tested and never copies KV, so it is the
  preferred memory-saving path when raw broadcast is unavailable.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import torch

VALID_BACKENDS = ("auto", "flash", "mem_efficient", "math", "default")


def set_attention_backend(backend="auto"):
    """Globally enable/disable SDPA backends.

    Parameters
    ----------
    backend : {"auto", "flash", "mem_efficient", "math", "default"}
        * ``auto`` / ``default``: let PyTorch choose (all backends enabled).
        * ``flash``: force Flash Attention only.
        * ``mem_efficient``: force memory-efficient attention only.
        * ``math``: force plain cuBLAS/math attention only.
    """
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown attention backend: {backend}. Choose from {VALID_BACKENDS}"
        )

    if not torch.cuda.is_available():
        return

    if backend in ("auto", "default"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        return

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    if backend == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
    elif backend == "mem_efficient":
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    elif backend == "math":
        torch.backends.cuda.enable_math_sdp(True)


def get_attention_backend_info():
    """Return a dict of currently enabled SDPA backends."""
    if not torch.cuda.is_available():
        return {}
    return {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }


def print_attention_backend(prefix="[Attention]"):
    """Print the currently active SDPA backends."""
    info = get_attention_backend_info()
    active = [k for k, v in info.items() if v]
    print(f"{prefix} SDPA backends enabled: {active}")


# ---------------------------------------------------------------------------
#  GQA (Grouped Query Attention) broadcast support probing
# ---------------------------------------------------------------------------
#
# Some PyTorch/SDPA stacks cannot accept Q/K/V with different head counts
# directly (the fused kernels raise ``No available kernel``).  To still get the
# KV-cache memory savings of GQA without copying KV to the full query-head
# count we use a *grouped-broadcast* reshape:
#
#     q: [B, n_head,    T, D] -> [B, n_kv, rep, T, D]
#     k: [B, n_kv_head, S, D] -> [B, n_kv, 1,   S, D]
#     v: [B, n_kv_head, S, D] -> [B, n_kv, 1,   S, D]
#
# SDPA then broadcasts the singleton ``rep`` dimension internally, producing a
# result that is bit-wise identical to ``repeat_interleave`` while keeping KV
# at its native size.

_PROBE_LOCK = threading.Lock()
_PROBE_CACHE: Dict[Tuple, Tuple[bool, bool]] = {}


def probe_gqa_sdpa_support(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    n_head: int = 4,
    n_kv_head: int = 2,
    seq_len: int = 16,
    head_dim: int = 16,
    force_refresh: bool = False,
) -> Tuple[bool, bool]:
    """Probe whether the active SDPA stack supports GQA without KV copy.

    Returns a tuple ``(supports_raw_broadcast, supports_grouped_broadcast)``.

    * ``supports_raw_broadcast`` is ``True`` when ``F.scaled_dot_product_attention``
      accepts ``q``/``k``/``v`` with ``n_head != n_kv_head`` directly.
    * ``supports_grouped_broadcast`` is ``True`` when the grouped-reshape
      trick (Q grouped into ``[B, n_kv, rep, T, D]``, KV ``[B, n_kv, 1, S, D]``)
      works on this device/dtype.

    The result is cached per ``(device, dtype)`` so repeated calls are free.
    Probing is wrapped in ``try/except`` so it never raises — on failure both
    flags are ``False`` and the caller falls back to ``repeat_interleave``.
    """
    if n_head % n_kv_head != 0:
        # Invalid GQA config; report no support so callers keep the safe path.
        return (False, False)

    key = (str(device), str(dtype), n_head, n_kv_head, seq_len, head_dim)
    if not force_refresh and key in _PROBE_CACHE:
        return _PROBE_CACHE[key]

    supports_raw = False
    supports_grouped = False

    try:
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            _PROBE_CACHE[key] = (False, False)
            return (False, False)
    except Exception:
        _PROBE_CACHE[key] = (False, False)
        return (False, False)

    rep = n_head // n_kv_head
    B = 1
    try:
        # Save the global random state so the probe does not affect external
        # sampling (e.g. torch.multinomial inside model.generate).
        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(device)

        try:
            q = torch.randn(B, n_head, seq_len, head_dim, device=device, dtype=dtype)
            k = torch.randn(B, n_kv_head, seq_len, head_dim, device=device, dtype=dtype)
            v = torch.randn(B, n_kv_head, seq_len, head_dim, device=device, dtype=dtype)
        finally:
            torch.set_rng_state(rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)

        # Reference computed with explicit repeat_interleave (always valid).
        ref = torch.nn.functional.scaled_dot_product_attention(
            q,
            k.repeat_interleave(rep, dim=1),
            v.repeat_interleave(rep, dim=1),
            is_causal=True,
        )

        # 1) Raw broadcast: different head counts passed directly.
        try:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
            )
            supports_raw = out.shape == ref.shape and bool(
                torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
            )
        except Exception:
            supports_raw = False

        # 2) Grouped broadcast: reshape Q and unsqueeze KV so SDPA broadcasts.
        try:
            q_g = q.view(B, n_kv_head, rep, seq_len, head_dim)
            k_g = k.unsqueeze(2)  # [B, n_kv, 1, S, D]
            v_g = v.unsqueeze(2)  # [B, n_kv, 1, S, D]
            out_g = torch.nn.functional.scaled_dot_product_attention(
                q_g,
                k_g,
                v_g,
                is_causal=True,
            )
            out_g = out_g.reshape(B, n_head, seq_len, head_dim)
            supports_grouped = out_g.shape == ref.shape and bool(
                torch.allclose(out_g, ref, atol=1e-3, rtol=1e-3)
            )
        except Exception:
            supports_grouped = False
    except Exception:
        # Allocation/launch failure (e.g. OOM, device not ready): be safe.
        supports_raw = False
        supports_grouped = False

    with _PROBE_LOCK:
        _PROBE_CACHE[key] = (supports_raw, supports_grouped)
    return (supports_raw, supports_grouped)


def reset_gqa_probe_cache() -> None:
    """Clear the cached GQA probe results (mainly for testing)."""
    with _PROBE_LOCK:
        _PROBE_CACHE.clear()
