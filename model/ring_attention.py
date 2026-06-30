"""Pure-PyTorch blockwise ("Ring") attention fallback.

This module does **not** require ``flash-attn`` or Triton.  It implements the
online softmax algorithm from `From Online Softmax to FlashAttention`_ in
plain PyTorch so that very long sequences can be processed block-by-block.
The function is autograd-friendly and can be used as a drop-in replacement
for ``F.scaled_dot_product_attention`` when the optional FlashAttention
package is unavailable or when the user explicitly wants deterministic,
backend-independent attention.

.. _From Online Softmax to FlashAttention:
   https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def is_available() -> bool:
    """Always True: the blockwise fallback uses only PyTorch primitives."""
    return True


def blockwise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    block_size_q: Optional[int] = None,
    block_size_kv: Optional[int] = None,
) -> torch.Tensor:
    """Compute causal self-attention in query/kv blocks.

    Parameters
    ----------
    q
        Query tensor of shape ``[B, n_q_heads, T, head_dim]``.
    k, v
        Key/value tensors of shape ``[B, n_kv_heads, T, head_dim]``.  When
        ``n_kv_heads < n_q_heads`` (GQA), the KV heads are repeated
        block-by-block so that the output has ``n_q_heads`` heads.
    causal
        Whether to apply a lower-triangular causal mask.
    softmax_scale
        Optional custom scale.  Defaults to ``1 / sqrt(head_dim)``.
    block_size_q
        Number of query positions processed together.
    block_size_kv
        Number of key/value positions processed together.

    Returns
    -------
    torch.Tensor
        Attention output of shape ``[B, n_q_heads, T, head_dim]``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("blockwise_attention expects 4-D inputs [B,H,T,D]")

    B, Hq, Tq, D = q.shape
    _, Hkv, Tkv, _ = k.shape
    if k.shape != (B, Hkv, Tkv, D):
        raise ValueError(f"k shape {k.shape} is incompatible with q shape {q.shape}")
    if v.shape != (B, Hkv, Tkv, D):
        raise ValueError(f"v shape {v.shape} is incompatible with q shape {q.shape}")
    if Hq % Hkv != 0:
        raise ValueError(
            f"Number of query heads ({Hq}) must be divisible by number of KV heads ({Hkv})"
        )
    n_rep = Hq // Hkv

    if Tq != Tkv:
        raise ValueError(
            f"blockwise_attention currently requires Tq == Tkv, got {Tq} and {Tkv}"
        )

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    block_size_q = block_size_q or min(64, max(1, Tq))
    block_size_kv = block_size_kv or block_size_q
    if block_size_q <= 0 or block_size_kv <= 0:
        raise ValueError("block sizes must be positive")

    # Pad sequence lengths to multiples of the block sizes.
    pad_q = (block_size_q - Tq % block_size_q) % block_size_q
    pad_kv = (block_size_kv - Tkv % block_size_kv) % block_size_kv
    if pad_q or pad_kv:
        q = F.pad(q, (0, 0, 0, pad_q))
        k = F.pad(k, (0, 0, 0, pad_kv))
        v = F.pad(v, (0, 0, 0, pad_kv))

    Tq_pad = q.shape[2]
    Tkv_pad = k.shape[2]
    nq = Tq_pad // block_size_q
    nkv = Tkv_pad // block_size_kv

    out = torch.zeros_like(q)
    for i in range(nq):
        q_i = q[:, :, i * block_size_q : (i + 1) * block_size_q, :]
        q_start = i * block_size_q
        q_end = q_start + block_size_q

        m = torch.full(
            (B, Hq, block_size_q, 1),
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )
        l = torch.zeros((B, Hq, block_size_q, 1), device=q.device, dtype=q.dtype)
        acc = torch.zeros(
            (B, Hq, block_size_q, D),
            device=q.device,
            dtype=q.dtype,
        )

        for j in range(nkv):
            k_j = k[:, :, j * block_size_kv : (j + 1) * block_size_kv, :]
            v_j = v[:, :, j * block_size_kv : (j + 1) * block_size_kv, :]

            # Handle GQA by repeating KV heads locally within each block.
            if n_rep > 1:
                k_j = k_j.repeat_interleave(n_rep, dim=1)
                v_j = v_j.repeat_interleave(n_rep, dim=1)

            s_ij = torch.matmul(q_i, k_j.transpose(-2, -1)) * softmax_scale

            k_start = j * block_size_kv
            k_end = k_start + block_size_kv
            q_pos = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)
            k_pos = torch.arange(k_start, k_end, device=q.device).unsqueeze(0)
            mask = torch.ones(
                block_size_q, block_size_kv, dtype=torch.bool, device=q.device
            )
            if causal:
                mask = mask & (k_pos <= q_pos)
            # KV padding positions are beyond the original Tkv and must never
            # be attended to, even for non-causal attention.
            if k_end > Tkv:
                mask = mask & (k_pos < Tkv)
            if not mask.all():
                mask_b = mask.view(1, 1, block_size_q, block_size_kv).expand(
                    B, Hq, -1, -1
                )
                s_ij = s_ij.masked_fill(~mask_b, float("-inf"))

            m_ij = torch.max(s_ij, dim=-1, keepdim=True).values
            m_new = torch.maximum(m, m_ij)

            # Replace -inf with 0 for the exponential so we do not produce NaNs
            # for rows that are entirely masked in this block.
            m_new_safe = torch.where(
                torch.isneginf(m_new), torch.zeros_like(m_new), m_new
            )
            exp_scale = torch.exp(m - m_new_safe)
            exp_ij = torch.exp(s_ij - m_new_safe)

            l = l * exp_scale + exp_ij.sum(dim=-1, keepdim=True)
            acc = acc * exp_scale + torch.matmul(exp_ij, v_j)
            m = m_new

        # Rows that received no valid KV tokens would divide by 0; set l to 1
        # so their output remains 0 (these rows belong to padding queries).
        l = torch.where(l == 0, torch.ones_like(l), l)
        out[:, :, i * block_size_q : (i + 1) * block_size_q, :] = acc / l

    if pad_q:
        out = out[:, :, :Tq, :]
    return out
