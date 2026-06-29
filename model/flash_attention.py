"""Optional flash-attention backend wrapper with graceful fallback.

This module isolates the third-party ``flash-attn`` dependency.  If the package
is not installed or the current platform does not provide a working wheel, the
backend reports ``is_available() == False`` and ``CausalSelfAttention`` falls
back to ``torch.nn.functional.scaled_dot_product_attention``.

Three entry points are provided, corresponding to the three main use-cases in
a Transformer decoder:

* :func:`flash_attention` — standard forward for training / prefill.
  Calls ``flash_attn_func``; supports GQA natively (different Q/KV head
  counts are handled by the kernel without ``repeat_interleave``).
* :func:`flash_attention_varlen` — variable-length / packed sequences.
  Calls ``flash_attn_varlen_func`` with cumulative sequence-length pointers
  (``cu_seqlens``).  Used for packed training with ``document_ids``.
* :func:`flash_attention_with_cache` — decode-step attention.
  Calls ``flash_attn_with_kvcache`` when available, falling back to
  ``flash_attn_func`` with explicit cache concatenation otherwise.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
#  Availability introspection
# ---------------------------------------------------------------------------

def is_available() -> bool:
    """Return True if ``flash_attn`` is installed and imports cleanly."""
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def version_info() -> Optional[Tuple[int, ...]]:
    """Return the ``flash_attn`` version tuple, or ``None`` if unavailable."""
    try:
        import flash_attn
        return tuple(int(x) for x in flash_attn.__version__.split(".") if x.isdigit())
    except Exception:
        return None


def has_varlen() -> bool:
    """Return True if ``flash_attn_varlen_func`` is available."""
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
        return True
    except Exception:
        return False


def has_kvcache() -> bool:
    """Return True if ``flash_attn_with_kvcache`` is available."""
    try:
        from flash_attn import flash_attn_with_kvcache  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  Standard forward (training / prefill)
# ---------------------------------------------------------------------------

def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Compute memory-efficient attention using ``flash_attn.flash_attn_func``.

    Parameters
    ----------
    q, k, v
        Tensors of shape ``[B, n_heads, T, head_dim]`` (the same layout used
        by PyTorch SDPA).  They are transposed internally because flash-attn
        expects ``[B, T, n_heads, head_dim]``.
    causal
        Whether to apply a causal mask.
    softmax_scale
        Optional custom scale factor.  Defaults to ``1 / sqrt(head_dim)``.

    Returns
    -------
    torch.Tensor or None
        Attention output of shape ``[B, n_heads, T, head_dim]``, or ``None``
        if flash-attn is unavailable / fails.
    """
    try:
        from flash_attn import flash_attn_func  # type: ignore[import]
    except Exception:
        return None

    # flash-attn expects [B, T, n_heads, head_dim]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    try:
        out = flash_attn_func(
            q_t, k_t, v_t,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    except Exception:
        return None
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
#  Variable-length / packed sequences
# ---------------------------------------------------------------------------

def flash_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Compute attention for variable-length packed sequences.

    This is the flash-attn counterpart for training with packed documents
    (i.e. when ``document_ids`` is provided).  Each batch "element" in the
    packed tensor is actually a concatenation of multiple shorter sequences;
    ``cu_seqlens_q`` / ``cu_seqlens_k`` delimit their boundaries.

    Parameters
    ----------
    q, k, v
        Tensors of shape ``[total_q, n_heads, head_dim]`` /
        ``[total_k, n_kv_heads, head_dim]`` — already in the flash-attn
        layout (no batch dimension).
    cu_seqlens_q, cu_seqlens_k
        Cumulative sequence lengths, shape ``[B+1]``, on CUDA, ``torch.int32``.
    max_seqlen_q, max_seqlen_k
        Maximum sequence length in the batch (for kernel dispatch).
    causal
        Whether to apply a causal mask within each sequence.
    softmax_scale
        Optional custom scale.

    Returns
    -------
    torch.Tensor or None
        Output of shape ``[total_q, n_heads, head_dim]``, or ``None`` if
        flash-attn varlen is unavailable / fails.
    """
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore[import]
    except Exception:
        return None

    try:
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    except Exception:
        return None
    return out


# ---------------------------------------------------------------------------
#  Decode-step attention with KV cache
# ---------------------------------------------------------------------------

def flash_attention_with_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    cache_seqlens: Optional[torch.Tensor] = None,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> Optional[torch.Tensor]:
    """Compute decode-step attention using ``flash_attn_with_kvcache``.

    This function is designed for the single-token decode step in
    autoregressive generation.  ``k_cache`` / ``v_cache`` are pre-allocated
    tensors that hold previously computed keys/values; ``k_new`` / ``v_new``
    contain the newly computed key/value for the current token.

    Parameters
    ----------
    q
        Query tensor, shape ``[B, 1, n_heads, head_dim]`` (flash-attn layout).
    k_cache, v_cache
        Key/value cache tensors, shape
        ``[B, max_seq_len, n_kv_heads, head_dim]``.
    k_new, v_new
        New key/value for the current step, shape
        ``[B, 1, n_kv_heads, head_dim]``.
    cache_seqlens
        Per-batch sequence lengths in the cache, shape ``[B]``,
        ``torch.int32`` on CUDA.  Required by ``flash_attn_with_kvcache``.
    causal
        Always True for causal LMs; kept for API symmetry.
    softmax_scale
        Optional custom scale.

    Returns
    -------
    torch.Tensor or None
        Output of shape ``[B, 1, n_heads, head_dim]``, or ``None`` if
        ``flash_attn_with_kvcache`` is unavailable / fails.
    """
    try:
        from flash_attn import flash_attn_with_kvcache as _fa_kvcache  # type: ignore[import]
    except Exception:
        return None

    try:
        out = _fa_kvcache(
            q,
            k_cache,
            v_cache,
            k_new,
            v_new,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    except Exception:
        return None
    return out
