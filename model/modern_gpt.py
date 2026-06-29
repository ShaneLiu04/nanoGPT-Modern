"""
ModernGPT: RMSNorm + SwiGLU FFN + RoPE (Rotary Position Embedding).
Supports KV Cache for efficient autoregressive generation.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import flash_attention as _flash_attention
from . import ring_attention as _ring_attention
from .kv_cache_utils import build_sliding_window_mask


class RMSNorm(nn.Module):
    def __init__(self, ndim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use PyTorch's fused RMSNorm when available (PyTorch >= 2.4).
        if hasattr(F, "rms_norm") and not torch.jit.is_scripting():
            return F.rms_norm(x, x.shape[-1:], self.weight, self.eps)
        # Fallback: numerically stable manual implementation in fp32.
        x_fp32 = x.float()
        norm = x_fp32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE with lazy-updating cache for inference efficiency.

    During token-by-token decode the sequence length grows one at a time.
    Instead of re-computing ``cos``/``sin`` for every forward call, we
    pre-compute up to ``max_seq_len`` on the first use and cache the result.
    Subsequent calls with shorter ``seq_len`` are cheap index slices.

    Supports NTK-aware length extrapolation: when ``rope_scaling`` is
    ``{"type": "ntk", "factor": s}`` the base frequency is adjusted so that
    the model can attend to positions up to ``s * train_length`` without
    fine-tuning on those lengths.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_scaling: dict = rope_scaling or {}

        # NTK-aware scaling: increase the RoPE base so that the same set of
        # wavelengths covers a longer interval.
        theta = base
        if self.rope_scaling.get("type") == "ntk":
            factor = self.rope_scaling.get("factor", 1.0)
            if factor != 1.0:
                theta = base * (factor ** (dim / (dim - 2)))
        self.base = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # Lazily filled on first forward
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None

    def _ensure_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        """Build the full cos/sin table up to ``seq_len`` if not already cached."""
        cache_len = self._cos_cached.shape[0] if self._cos_cached is not None else 0
        if cache_len >= seq_len:
            return
        t = torch.arange(self.max_seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = x.shape[-3]
        self._ensure_cache(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        assert self._cos_cached is not None and self._sin_cached is not None
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.

    Equivalent to ``torch.cat([-x2, x1], dim=-1)`` where ``x1, x2 = x.chunk(2, -1)``
    but avoids the intermediate chunk allocations by reshaping to ``[..., 2, D/2]``,
    flipping the pair dimension, negating, and reshaping back.  This reduces peak
    memory by one full hidden-dim tensor per call.
    """
    # Reshape [..., D] -> [..., 2, D//2], flip pair dim, negate first half, reshape back
    x_pair = x.unflatten(-1, (2, -1))          # [..., 2, D//2]
    x_rot = x_pair.flip(-2)                     # swap the two halves
    x_rot = torch.stack([-x_rot[..., 0, :], x_rot[..., 1, :]], dim=-2)  # neg first half
    return x_rot.flatten(-2)                     # [..., D]


def apply_rotary_pos_emb_single(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: [batch, n_heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


def _probe_sdpa_gqa_support() -> bool:
    """Check whether the current PyTorch/SDPA stack supports *raw* GQA
    broadcasting (Q/KV with different head counts passed directly).

    .. deprecated::
        Kept for backward compatibility.  Prefer
        :func:`model.attention_utils.probe_gqa_sdpa_support`, which reports both
        raw and grouped broadcast capabilities.  ``CausalSelfAttention`` now
        uses the grouped-broadcast reshape which works on every SDPA backend
        and never copies KV — see :meth:`CausalSelfAttention._gqa_grouped_sdpa`.
    """
    return False


def _gqa_grouped_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_kv_head: int,
    n_rep: int,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute SDPA for Grouped Query Attention without copying KV heads.

    Instead of ``k.repeat_interleave(n_rep, dim=1)`` (which materialises KV to
    the full query-head count and erases the GQA memory saving), this helper
    reshapes Q into ``[B, n_kv, rep, T, D]`` and unsqueezes KV into
    ``[B, n_kv, 1, S, D]``.  ``F.scaled_dot_product_attention`` then broadcasts
    the singleton ``rep`` dimension natively, producing a result that is
    bit-wise identical to ``repeat_interleave`` while keeping KV at its native
    size (an ``expand`` view, zero extra allocation).

    This works on every SDPA backend we have tested (flash / memory-efficient
    / math / cuDNN) because the broadcast happens over a *real* dimension
    rather than a head-count mismatch.

    Parameters
    ----------
    q : torch.Tensor
        ``[B, n_head, T, D]`` query tensor.
    k, v : torch.Tensor
        ``[B, n_kv_head, S, D]`` key/value tensors.
    n_kv_head, n_rep : int
        Number of KV heads and the query-heads-per-KV-head ratio
        (``n_head // n_kv_head``).
    attn_mask, is_causal, dropout_p, scale
        Forwarded to ``F.scaled_dot_product_attention``.

    Returns
    -------
    torch.Tensor
        Attention output of shape ``[B, n_head, T, D]``.
    """
    B, n_head, T, D = q.shape
    # Group Q heads: [B, n_kv, rep, T, D]; KV: [B, n_kv, 1, S, D].
    q_g = q.view(B, n_kv_head, n_rep, T, D)
    k_g = k.unsqueeze(2)
    v_g = v.unsqueeze(2)
    # Broadcast attn_mask over the new ``rep`` axis when present.  An incoming
    # mask is shaped for the un-grouped attention ([B, 1, T, S] or
    # [B, 1, 1, S] or [B, 1, T, T]); inserting the head dims at positions 1-2
    # keeps it broadcastable to [B, n_kv, rep, T, S].
    if attn_mask is not None and attn_mask.ndim == 4:
        # [B, 1, X, Y] -> [B, 1, 1, X, Y]
        attn_mask = attn_mask.unsqueeze(1)
    out_g = F.scaled_dot_product_attention(
        q_g, k_g, v_g,
        attn_mask=attn_mask,
        is_causal=is_causal,
        dropout_p=dropout_p,
        scale=scale,
    )
    # Merge the grouped head dims back: [B, n_kv, rep, T, D] -> [B, n_head, T, D]
    return out_g.reshape(B, n_head, T, D)

class CausalSelfAttention(nn.Module):
    """Multi-Head / Grouped Query Attention with RoPE and KV Cache.

    When `config.n_kv_head < config.n_head`, GQA is active: K and V are
    projected to fewer heads.  Instead of always copying KV to the full
    query-head count via ``repeat_interleave`` (which erases the GQA memory
    saving), the attention computation uses one of three broadcast strategies:

    * **raw** — pass different head counts directly to SDPA (fastest, only
      available on some fused backends / newer PyTorch builds).
    * **grouped** (default when available) — reshape Q to
      ``[B, n_kv, rep, T, D]`` and KV to ``[B, n_kv, 1, S, D]`` so SDPA
      broadcasts the singleton ``rep`` dimension.  Numerically identical to
      ``repeat_interleave`` but **zero KV copy** — the original KV tensor is
      used directly via an ``expand`` view.
    * **repeat** — fall back to ``repeat_interleave`` (legacy / eager path).

    The strategy is resolved lazily on the first forward call by probing the
    actual device/dtype, unless the config field ``gqa_broadcast`` explicitly
    forces a specific mode.
    """

    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_rep
        self.use_gqa = self.n_kv_head < self.n_head  # True when GQA is active

        kv_dim = self.n_kv_head * self.head_dim  # narrower than q_dim when GQA
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # QK-Norm: applied per-head before RoPE to stabilize large-model training.
        self.use_qk_norm = getattr(config, "qk_norm", False)
        rmsnorm_eps = getattr(config, "rmsnorm_eps", 1e-6)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rmsnorm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rmsnorm_eps)

        # Attention temperature: scales attention logits independently of head_dim.
        # Default 1.0 preserves the standard 1/sqrt(head_dim) scaling.
        self.attn_temperature = getattr(config, "attn_temperature", 1.0)

        # RoPE operates on per-head dimension (shared by Q and K heads).
        # Allow NTK-aware extrapolation by scaling the maximum cached length.
        rope_factor = 1.0
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        if rope_scaling.get("type") == "ntk":
            rope_factor = rope_scaling.get("factor", 1.0)
        max_seq_len = int(config.block_size * max(2.0, rope_factor))
        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=max_seq_len,
            base=getattr(config, "rope_theta", 10000.0),
            rope_scaling=rope_scaling,
        )
        # GQA broadcast strategy: determines how KV heads are expanded to match
        # Q heads without copying.  One of:
        #   "auto"    — probe at runtime (default), resolves to raw/grouped/repeat
        #   "raw"     — pass different head counts directly to SDPA (fastest, rare)
        #   "grouped" — grouped-broadcast reshape (no KV copy, works everywhere)
        #   "repeat"  — fall back to repeat_interleave (legacy / eager path)
        self.gqa_broadcast: str = getattr(config, "gqa_broadcast", "auto")
        if self.gqa_broadcast in ("raw", "grouped", "repeat"):
            self._gqa_mode: str = self.gqa_broadcast
        else:
            self._gqa_mode = "repeat"  # resolved lazily on first forward

        # Optional third-party flash-attention backend (separate from SDPA).
        self.use_flash_attn = getattr(config, "use_flash_attn", False)

        # Optional pure-PyTorch blockwise (Ring) attention fallback.
        self.use_ring_attention = getattr(config, "use_ring_attention", False)
        self.ring_block_size_q = getattr(config, "ring_block_size_q", 64)
        self.ring_block_size_kv = getattr(config, "ring_block_size_kv", 64)

        # Sliding window attention: only attend to the last W tokens.
        self.sliding_window_size = getattr(config, "sliding_window_size", None)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Any = None,
        use_cache: bool = False,
        start_pos: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.size()
        # attention_mask: optional [B, T] or [B, 1, T] bool tensor.  True = attend,
        # False = mask out.  Used for left-padded batch generation/GRPO.
        # document_ids: optional [B, T] long tensor.  When provided, position i is
        # only allowed to attend to positions j <= i within the same document id.
        # This enables packed sequences with multiple documents per sample.

        # --- project to Q / K / V ---
        q = self.q_proj(x)          # [B, T, n_embd]
        k = self.k_proj(x)          # [B, T, n_kv_head * head_dim]  -- narrower when GQA
        v = self.v_proj(x)          # [B, T, n_kv_head * head_dim]

        # --- reshape to multi-head ---
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # [B, n_head,   T, hd]
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # [B, n_kv_head,T, hd]
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # [B, n_kv_head,T, hd]

        # --- QK-Norm (per-head, before RoPE) ---
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # --- RoPE with absolute position awareness ---
        if past_kv is None:
            past_len = 0
        elif isinstance(past_kv, list):
            past_len = sum(bk.shape[2] for bk, _ in past_kv)
        else:
            past_len = past_kv[0].shape[2]
        total_len = past_len + T
        cos_full, sin_full = self.rope(q, seq_len=start_pos + total_len)

        q_cos = cos_full[start_pos + past_len : start_pos + total_len]
        q_sin = sin_full[start_pos + past_len : start_pos + total_len]
        q_embed = apply_rotary_pos_emb_single(q, q_cos, q_sin)

        if past_len > 0:
            k_cos = cos_full[start_pos + past_len : start_pos + total_len]
            k_sin = sin_full[start_pos + past_len : start_pos + total_len]
        else:
            k_cos = cos_full[start_pos : start_pos + total_len]
            k_sin = sin_full[start_pos : start_pos + total_len]
        k_embed = apply_rotary_pos_emb_single(k, k_cos, k_sin)

        # --- KV cache: concatenate with past ---
        if past_kv is not None:
            # Sliding-window cache: keep only the last W tokens so the KV cache
            # memory stays bounded by the window size during autoregressive decode.
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                W = self.sliding_window_size
                if isinstance(past_kv, list):
                    trimmed: List[Tuple[torch.Tensor, torch.Tensor]] = []
                    remaining = W
                    for pk, pv in reversed(past_kv):
                        L = pk.shape[2]
                        if L >= remaining:
                            trimmed.insert(0, (pk[:, :, -remaining:, :], pv[:, :, -remaining:, :]))
                            remaining = 0
                            break
                        else:
                            trimmed.insert(0, (pk, pv))
                            remaining -= L
                    past_kv = trimmed
                else:
                    past_k, past_v = past_kv
                    if past_k.shape[2] > W:
                        past_kv = (past_k[:, :, -W:, :], past_v[:, :, -W:, :])

            if isinstance(past_kv, list):
                # Paged / ring-buffer cache: concatenate all past blocks plus the
                # newly computed K/V in one shot.
                k_embed = torch.cat([pk for pk, _ in past_kv] + [k_embed], dim=2)
                v = torch.cat([pv for _, pv in past_kv] + [v], dim=2)
            else:
                past_k, past_v = past_kv
                k_embed = torch.cat([past_k, k_embed], dim=2)
                v = torch.cat([past_v, v], dim=2)

        # Return only the K/V corresponding to the current input tokens so that
        # callers can append them to a KV cache without duplicating past tokens.
        present_kv = (k_embed[:, :, past_len:, :], v[:, :, past_len:, :]) if use_cache else None

        # --- attention ---
        attn_scale = None
        if self.attn_temperature != 1.0:
            attn_scale = self.attn_temperature / math.sqrt(self.head_dim)

        # Determine whether sliding-window attention is active for this step.
        window_active = (
            self.sliding_window_size is not None
            and self.sliding_window_size > 0
            and past_kv is None
            and T > 1
        )

        # Try the optional third-party flash-attention backend first.  It natively
        # supports GQA and is often faster than SDPA when a wheel is available.
        y: Optional[torch.Tensor] = None
        can_use_ring = (
            self.use_ring_attention
            and _ring_attention.is_available()
            and past_kv is None
            and T > 1
            and attention_mask is None
            and document_ids is None
            and not window_active
        )
        if can_use_ring:
            y = _ring_attention.blockwise_attention(
                q_embed,
                k_embed,
                v,
                causal=True,
                softmax_scale=attn_scale,
                block_size_q=self.ring_block_size_q,
                block_size_kv=self.ring_block_size_kv,
            )

        if y is None:
            can_use_flash = (
                self.use_flash_attn
                and _flash_attention.is_available()
                and past_kv is None
                and T > 1
                and attention_mask is None
                and document_ids is None
                and not window_active
            )
            if can_use_flash:
                y = _flash_attention.flash_attention(
                    q_embed, k_embed, v, causal=True, softmax_scale=attn_scale
                )

            if hasattr(F, "scaled_dot_product_attention"):
                # When using cache, q positions are always >= k positions, so no causal mask needed.
                is_causal = (past_kv is None and T > 1 and document_ids is None and not window_active)
                # Build an additive mask that combines causal + padding + document boundary + sliding window masking.
                attn_mask = None
                if window_active:
                    attn_mask = build_sliding_window_mask(T, self.sliding_window_size, x.device).to(q_embed.dtype)
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
                if attention_mask is not None or document_ids is not None:
                    if document_ids is not None:
                        # Cross-document attention mask: query i may only attend to positions j
                        # if both belong to the same document and j <= i (causal inside doc).
                        doc_ids = document_ids.unsqueeze(2)  # [B, T, 1]
                        same_doc = doc_ids == doc_ids.transpose(1, 2)  # [B, T, T]
                        causal_doc = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                        doc_mask = same_doc & causal_doc  # [B, T, T]
                        additive_mask = torch.zeros((B, 1, T, T), device=x.device, dtype=q_embed.dtype)
                        additive_mask.masked_fill_(~doc_mask.unsqueeze(1), float("-inf"))
                        if attention_mask is not None:
                            # attention_mask: True = attend, False = ignore.
                            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                            additive_mask.masked_fill_(~pad_mask, float("-inf"))
                        if window_active:
                            additive_mask = additive_mask + attn_mask
                        attn_mask = additive_mask
                        is_causal = False
                    elif attention_mask is not None:
                        # Padding-only mask: use the smallest broadcastable shape
                        # [B, 1, 1, T] instead of [B, 1, T, T] to save memory.
                        # This reduces memory from O(B*T^2) to O(B*T), critical for
                        # long sequences (e.g. T=8192 dense mask = 256MB per sample).
                        pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                        additive_mask = torch.zeros(
                            (B, 1, 1, T), device=x.device, dtype=q_embed.dtype
                        )
                        additive_mask.masked_fill_(~pad_mask, float("-inf"))
                        if window_active:
                            additive_mask = additive_mask.expand(-1, -1, T, -1) + attn_mask
                        attn_mask = additive_mask

                # GQA: resolve broadcast strategy.
                # ``_gqa_mode`` is one of "raw" / "grouped" / "repeat".
                # - "raw": pass different head counts directly to SDPA (fastest,
                #   works on some fused backends in newer PyTorch).
                # - "grouped": reshape Q to [B, n_kv, rep, T, D] and KV to
                #   [B, n_kv, 1, S, D] so SDPA broadcasts (no KV copy,
                #   works everywhere, same numerical result as repeat_interleave).
                # - "repeat": fall back to repeat_interleave (legacy eager path).
                # The mode is resolved lazily on first forward when the device
                # and dtype are known, unless the user sets ``gqa_broadcast``
                # in the config to force a specific strategy.
                if self.use_gqa:
                    if self._gqa_mode == "repeat" and self.gqa_broadcast == "auto":
                        # Lazy probe: resolve once per instance.
                        from .attention_utils import probe_gqa_sdpa_support
                        raw_ok, grouped_ok = probe_gqa_sdpa_support(
                            q_embed.device, q_embed.dtype,
                            n_head=self.n_head, n_kv_head=self.n_kv_head,
                            seq_len=min(T, 16), head_dim=self.head_dim,
                        )
                        if raw_ok:
                            self._gqa_mode = "raw"
                        elif grouped_ok:
                            self._gqa_mode = "grouped"
                        # else stays "repeat"

                    if self._gqa_mode == "raw":
                        # Pass KV with fewer heads directly; SDPA handles it.
                        y = F.scaled_dot_product_attention(
                            q_embed, k_embed, v, attn_mask=attn_mask, is_causal=is_causal,
                            dropout_p=self.attn_dropout.p if self.training else 0.0,
                            scale=attn_scale,
                        )
                    elif self._gqa_mode == "grouped":
                        y = _gqa_grouped_sdpa(
                            q_embed, k_embed, v,
                            self.n_kv_head, self.n_rep,
                            attn_mask=attn_mask, is_causal=is_causal,
                            dropout_p=self.attn_dropout.p if self.training else 0.0,
                            scale=attn_scale,
                        )
                    else:
                        # "repeat" fallback: copy KV to match Q heads.
                        k_exp = k_embed.repeat_interleave(self.n_rep, dim=1)
                        v_exp = v.repeat_interleave(self.n_rep, dim=1)
                        y = F.scaled_dot_product_attention(
                            q_embed, k_exp, v_exp, attn_mask=attn_mask, is_causal=is_causal,
                            dropout_p=self.attn_dropout.p if self.training else 0.0,
                            scale=attn_scale,
                        )
                else:
                    y = F.scaled_dot_product_attention(
                        q_embed, k_embed, v, attn_mask=attn_mask, is_causal=is_causal,
                        dropout_p=self.attn_dropout.p if self.training else 0.0,
                        scale=attn_scale,
                    )
            else:
                # Eager / legacy path: manually repeat KV heads so matmul shapes match.
                k_eager, v_eager = k_embed, v
                if self.use_gqa:
                    k_eager = k_embed.repeat_interleave(self.n_rep, dim=1)
                    v_eager = v.repeat_interleave(self.n_rep, dim=1)
                att = (q_embed @ k_eager.transpose(-2, -1)) * (self.attn_temperature / math.sqrt(self.head_dim))
                if T > 1 and past_kv is None:
                    if window_active:
                        window_mask = build_sliding_window_mask(T, self.sliding_window_size, x.device)
                        att = att + window_mask.unsqueeze(0).unsqueeze(0)
                    elif document_ids is not None:
                        doc_ids = document_ids.unsqueeze(2)
                        same_doc = doc_ids == doc_ids.transpose(1, 2)
                        causal_doc = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
                        att = att.masked_fill(~(same_doc & causal_doc).unsqueeze(1), float("-inf"))
                    else:
                        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
                        att = att.masked_fill(causal_mask == 0, float("-inf"))
                if attention_mask is not None:
                    pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                    att = att.masked_fill(~pad_mask, float("-inf"))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v_eager

        assert y is not None

        # --- output projection ---
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        return y, present_kv


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network with optional MoE (Mixture of Experts).

    Parameters
    ----------
    config.intermediate_size : int
        Hidden dimension.  Defaults to `8d/3` rounded up to the nearest
        multiple of `multiple_of` (256) for GPU-friendly alignment.

    config.num_experts : int
        Number of experts for MoE.  Default 1 (dense SwiGLU).  When > 1,
        top-1 gating is used: each token routes to the single expert with
        the highest routing score.  Total forward FLOPs remain close to the
        dense case (only one expert active per token), but total parameter
        count scales with `num_experts`.
    """

    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        hidden = config.intermediate_size
        self.num_experts = getattr(config, "num_experts", 1)
        self.moe_aux_loss_factor = getattr(config, "moe_aux_loss_factor", 0.01)
        self.moe_capacity_factor = getattr(config, "moe_capacity_factor", 1.25)

        if self.num_experts <= 1:
            # dense SwiGLU
            self.gate_proj: nn.Module = nn.Linear(config.n_embd, hidden, bias=False)
            self.up_proj: nn.Module   = nn.Linear(config.n_embd, hidden, bias=False)
            self.down_proj: nn.Module = nn.Linear(hidden, config.n_embd, bias=False)
        else:
            # MoE: each expert has its own gate/up/down projection
            self.gate_proj = nn.ModuleList([
                nn.Linear(config.n_embd, hidden, bias=False) for _ in range(self.num_experts)
            ])
            self.up_proj = nn.ModuleList([
                nn.Linear(config.n_embd, hidden, bias=False) for _ in range(self.num_experts)
            ])
            self.down_proj = nn.ModuleList([
                nn.Linear(hidden, config.n_embd, bias=False) for _ in range(self.num_experts)
            ])
            # router: learns per-expert logits from the input
            self.router = nn.Linear(config.n_embd, self.num_experts, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_experts <= 1:
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            out = self.down_proj(gate * up)
            return self.dropout(out), torch.zeros((), device=x.device, dtype=x.dtype)

        # --- MoE: top-1 gating with load-balancing aux loss and capacity limit ---
        B, T, C = x.shape
        x_flat = x.view(B * T, C)                    # [B*T, C]

        # routing scores
        router_logits = self.router(x_flat)           # [B*T, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # top-1 selection: choose best expert per token
        _, selected = torch.topk(router_probs, 1, dim=-1)   # [B*T, 1]
        selected = selected.squeeze(-1)                      # [B*T]

        # Load-balancing aux loss: encourage uniform expert utilization.
        # f_i = fraction of tokens routed to expert i
        # P_i = mean router probability assigned to expert i
        f = torch.stack([(selected == e).float().mean() for e in range(self.num_experts)])
        P = router_probs.mean(dim=0)
        aux_loss = self.moe_aux_loss_factor * self.num_experts * (f * P).sum()

        # Capacity factor: each expert can process at most capacity tokens.
        total_tokens = B * T
        capacity = int(self.moe_capacity_factor * total_tokens / self.num_experts)

        out_flat = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            idx = (selected == e).nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            # Drop tokens beyond the per-expert capacity (common in Switch Transformer).
            if idx.numel() > capacity:
                idx = idx[:capacity]
            xe = x_flat.index_select(0, idx)                 # [n_e, C]
            ge = F.silu(self.gate_proj[e](xe))  # type: ignore[index]
            ue = self.up_proj[e](xe)              # type: ignore[index]
            de = self.down_proj[e](ge * ue)       # type: ignore[index]
            # weight by routing probability for gradient flow
            weight = router_probs.index_select(0, idx)[:, e].unsqueeze(-1)  # [n_e, 1]
            out_flat.index_copy_(0, idx, de * weight)

        out = out_flat.view(B, T, C)
        out = self.dropout(out)
        return out, aux_loss


class Block(nn.Module):
    """Transformer block with configurable Pre-Norm / Post-Norm.

    * `norm_position="pre"` (default, LLaMA-style):
      `x = x + Attn(Norm(x));  x = x + MLP(Norm(x))`
      Better final loss; requires warmup for stability.

    * `norm_position="post"` (GPT-2 original):
      `x = Norm(x + Attn(x));  x = Norm(x + MLP(x))`
      More stable without warmup, but slightly higher final loss.
    """

    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        self.config = config
        self.norm_pos = getattr(config, "norm_position", "pre")
        if self.norm_pos not in ("pre", "post"):
            raise ValueError(f"norm_position must be 'pre' or 'post', got '{self.norm_pos}'")

        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Any = None,
        use_cache: bool = False,
        start_pos: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]],
        Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    ]:
        gc_enabled = (
            self.training
            and getattr(self.config, "gradient_checkpointing", False)
        )

        aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        if self.norm_pos == "pre":
            if gc_enabled:
                # Gradient checkpointing is a training-only optimization.  We
                # checkpoint the attention and MLP sub-modules separately and do
                # not materialize KV-cache tensors during training.
                attn_out = checkpoint(
                    self.attn,
                    self.ln_1(x),
                    use_cache=False,
                    start_pos=start_pos,
                    attention_mask=attention_mask,
                    document_ids=document_ids,
                    use_reentrant=False,
                )[0]
                x = x + attn_out
                mlp_out, mlp_aux = checkpoint(
                    self.mlp, self.ln_2(x), use_reentrant=False
                )
                x = x + mlp_out
                aux_loss = aux_loss + mlp_aux
                if return_aux_loss:
                    return x, None, aux_loss
                return x, None

            attn_out, present_kv = self.attn(
                self.ln_1(x), past_kv=past_kv, use_cache=use_cache, start_pos=start_pos,
                attention_mask=attention_mask, document_ids=document_ids,
            )
            x = x + attn_out
            mlp_out, mlp_aux = self.mlp(self.ln_2(x))
            x = x + mlp_out
            aux_loss = aux_loss + mlp_aux
            if return_aux_loss:
                return x, present_kv, aux_loss
            return x, present_kv
        else:
            # post-norm: GPT-2 original style
            if gc_enabled:
                attn_out = checkpoint(
                    self.attn,
                    x,
                    use_cache=False,
                    start_pos=start_pos,
                    attention_mask=attention_mask,
                    document_ids=document_ids,
                    use_reentrant=False,
                )[0]
                x = self.ln_1(x + attn_out)
                mlp_out, mlp_aux = checkpoint(self.mlp, x, use_reentrant=False)
                x = self.ln_2(x + mlp_out)
                aux_loss = aux_loss + mlp_aux
                if return_aux_loss:
                    return x, None, aux_loss
                return x, None

            residual = x
            attn_out, present_kv = self.attn(
                x, past_kv=past_kv, use_cache=use_cache, start_pos=start_pos,
                attention_mask=attention_mask, document_ids=document_ids,
            )
            x = self.ln_1(residual + attn_out)
            mlp_out, mlp_aux = self.mlp(x)
            x = self.ln_2(x + mlp_out)
            aux_loss = aux_loss + mlp_aux
            if return_aux_loss:
                return x, present_kv, aux_loss
            return x, present_kv


class ModernGPTConfig:
    """Configuration for ModernGPT with Grouped Query Attention (GQA) support.

    GQA reduces KV-cache memory by sharing K/V heads across multiple Q heads.
    When `n_kv_head == n_head` this degenerates to standard Multi-Head Attention.
    When `n_kv_head < n_head` the KV cache per token is reduced by the ratio
    `n_kv_head / n_head`.

    Parameters
    ----------
    n_kv_head : int or None
        Number of key/value heads.  Must divide `n_head`.  Defaults to `n_head`
        (MHA).  Set to a smaller divisor (e.g. 2 or 4) to enable GQA.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        block_size: int = 1024,
        n_layer: int = 12,
        n_head: int = 8,
        n_embd: int = 512,
        n_kv_head: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        norm_position: str = "pre",
        num_experts: int = 1,
        gradient_checkpointing: bool = False,
        qk_norm: bool = False,
        attn_temperature: float = 1.0,
        rmsnorm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        moe_aux_loss_factor: float = 0.01,
        moe_capacity_factor: float = 1.25,
        use_flash_attn: bool = False,
        use_ring_attention: bool = False,
        ring_block_size_q: int = 64,
        ring_block_size_kv: int = 64,
        n_future: int = 0,
        mtp_weight: float = 1.0,
        sliding_window_size: Optional[int] = None,
        use_paged_kv_cache: bool = False,
        kv_cache_block_size: int = 16,
        gqa_broadcast: str = "auto",
    ):
        # --- architecture ---
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        # --- GQA (Grouped Query Attention) ---
        if n_kv_head is None:
            n_kv_head = n_head  # default: full MHA
        if n_head % n_kv_head != 0:
            raise ValueError(
                f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
            )
        if n_kv_head > n_head:
            raise ValueError(
                f"n_kv_head ({n_kv_head}) cannot exceed n_head ({n_head})"
            )
        self.n_kv_head = n_kv_head
        self._n_rep = n_head // n_kv_head

        # --- SwiGLU (with multiple-of alignment for GPU efficiency) ---
        if intermediate_size is None:
            raw = int(8 / 3 * n_embd)
            multiple_of = 128  # LLaMA-style alignment
            intermediate_size = ((raw + multiple_of - 1) // multiple_of) * multiple_of
        self.intermediate_size = intermediate_size

        # --- normalization ---
        if norm_position not in ("pre", "post"):
            raise ValueError(
                f"norm_position must be 'pre' or 'post', got '{norm_position}'"
            )
        self.norm_position = norm_position

        # --- MoE (experimental) ---
        self.num_experts = num_experts
        self.moe_aux_loss_factor = moe_aux_loss_factor
        self.moe_capacity_factor = moe_capacity_factor

        # --- attention backend ---
        self.use_flash_attn = use_flash_attn
        self.use_ring_attention = use_ring_attention
        self.ring_block_size_q = ring_block_size_q
        self.ring_block_size_kv = ring_block_size_kv

        # --- Multi-Token Prediction (MTP) ---
        self.n_future = n_future
        self.mtp_weight = mtp_weight

        # --- Sliding Window Attention ---
        self.sliding_window_size = sliding_window_size

        # --- KV Cache backend ---
        self.use_paged_kv_cache = use_paged_kv_cache
        self.kv_cache_block_size = kv_cache_block_size

        # --- GQA broadcast strategy ---
        if gqa_broadcast not in ("auto", "raw", "grouped", "repeat"):
            raise ValueError(
                f"gqa_broadcast must be 'auto', 'raw', 'grouped', or 'repeat'; "
                f"got '{gqa_broadcast}'"
            )
        self.gqa_broadcast = gqa_broadcast

        # --- memory-efficient training ---
        self.gradient_checkpointing = gradient_checkpointing  # set > 1 to enable MoE FFN (top-1 gating)

        # --- attention stabilization ---
        self.qk_norm = qk_norm
        self.attn_temperature = attn_temperature
        self.rmsnorm_eps = rmsnorm_eps

        # --- RoPE / position encoding ---
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # --- regularization ---
        self.dropout = dropout

    @property
    def n_rep(self) -> int:
        """Query heads per KV head.  Property allows old checkpoints without
        the stored attribute to compute it on demand."""
        return getattr(self, "_n_rep", self.n_head // self.n_kv_head)

    @n_rep.setter
    def n_rep(self, value):
        self._n_rep = value

    def to_dict(self) -> dict:
        """Serialize for checkpoint / logging (shallow dict).

        Uses ``getattr`` with defaults so old config objects that lack newer
        fields (e.g. ``norm_position`` or ``num_experts``) can still be saved.
        """
        return {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "n_kv_head": self.n_kv_head,
            "n_rep": self.n_rep,
            "intermediate_size": getattr(self, "intermediate_size", int(8 / 3 * self.n_embd)),
            "dropout": self.dropout,
            "norm_position": getattr(self, "norm_position", "pre"),
            "num_experts": getattr(self, "num_experts", 1),
            "gradient_checkpointing": getattr(self, "gradient_checkpointing", False),
            "qk_norm": getattr(self, "qk_norm", False),
            "attn_temperature": getattr(self, "attn_temperature", 1.0),
            "rmsnorm_eps": getattr(self, "rmsnorm_eps", 1e-6),
            "rope_theta": getattr(self, "rope_theta", 10000.0),
            "rope_scaling": getattr(self, "rope_scaling", None),
            "moe_aux_loss_factor": getattr(self, "moe_aux_loss_factor", 0.01),
            "moe_capacity_factor": getattr(self, "moe_capacity_factor", 1.25),
            "use_flash_attn": getattr(self, "use_flash_attn", False),
            "use_ring_attention": getattr(self, "use_ring_attention", False),
            "ring_block_size_q": getattr(self, "ring_block_size_q", 64),
            "ring_block_size_kv": getattr(self, "ring_block_size_kv", 64),
            "n_future": getattr(self, "n_future", 0),
            "mtp_weight": getattr(self, "mtp_weight", 1.0),
            "sliding_window_size": getattr(self, "sliding_window_size", None),
            "use_paged_kv_cache": getattr(self, "use_paged_kv_cache", False),
            "kv_cache_block_size": getattr(self, "kv_cache_block_size", 16),
            "gqa_broadcast": getattr(self, "gqa_broadcast", "auto"),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ModernGPTConfig:
        """Deserialize from dict.  Backward-compatible with old checkpoints that
        may not include `n_kv_head`."""
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})

    @property
    def gqa_ratio(self) -> float:
        """Fraction of KV heads vs query heads (1.0 = full MHA)."""
        return self.n_kv_head / self.n_head

    @property
    def effective_block_size(self):
        """Maximum sequence length allowed for forward passes.

        Equals ``block_size`` by default.  With NTK-aware RoPE scaling the
        model can extrapolate to ``block_size * factor`` without additional
        training.
        """
        factor = 1.0
        rope_scaling = getattr(self, "rope_scaling", None) or {}
        if rope_scaling.get("type") == "ntk":
            factor = rope_scaling.get("factor", 1.0)
        return int(self.block_size * factor)

    def estimate_kv_cache_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """Estimate KV-cache memory (bytes) per generated token.

        Parameters
        ----------
        dtype_bytes : int
            Bytes per element (2 for bf16/fp16, 4 for fp32).

        Returns
        -------
        int
            Per-token KV-cache size in bytes.
        """
        head_dim = self.n_embd // self.n_head
        per_layer = 2 * self.n_kv_head * head_dim * dtype_bytes  # K + V
        return per_layer * self.n_layer

    def describe(self) -> str:
        """Return a human-readable summary string."""
        gqa_tag = "MHA" if self.n_kv_head == self.n_head else f"GQA-{self.n_kv_head}KV"
        kv_bytes = self.estimate_kv_cache_bytes_per_token()
        return (
            f"ModernGPTConfig({gqa_tag}, {self.n_layer}L/{self.n_head}H/{self.n_embd}D, "
            f"ffn={self.intermediate_size}, ctx={self.block_size}, "
            f"kv_cache={kv_bytes}B/tok)"
        )

class ModernGPT(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Multi-Token Prediction heads: each predicts the (k+1)-th future token.
        self.n_future = config.n_future
        self.mtp_weight = config.mtp_weight
        if self.n_future > 0:
            self.future_heads = nn.ModuleList([
                nn.Linear(config.n_embd, config.vocab_size, bias=False)
                for _ in range(self.n_future)
            ])
            # Tie future head weights with the main lm_head to limit param growth.
            for head in self.future_heads:
                head.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # --- log attention backend ---
        self._log_attention_backend()
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _log_attention_backend():
        """Log the PyTorch SDPA backend being used.

        Prints one line showing which kernel PyTorch will select:
        `FlashAttention` / `MemoryEfficient` / `Math`.
        In training scripts the preferred approach is to call this after
        the model is moved to GPU; here it fires at model creation time
        (backend selection is device-independent in PyTorch >= 2.0).
        """
        try:
            # PyTorch >= 2.4 exposes the preferred query API in torch.nn.attention.
            from torch.nn.attention import (
                flash_sdp_enabled,
                math_sdp_enabled,
                mem_efficient_sdp_enabled,
            )
            ctx_info = {
                "enable_flash": flash_sdp_enabled(),
                "enable_math": math_sdp_enabled(),
                "enable_mem_efficient": mem_efficient_sdp_enabled(),
            }
        except Exception:
            # Fall back to the legacy cuda-specific helper.
            if not hasattr(torch.backends.cuda, "sdp_kernel"):
                return  # CPU-only or PyTorch < 2.0
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    ctx = torch.backends.cuda.sdp_kernel()
                ctx_info = {
                    "enable_flash": ctx.enable_flash,
                    "enable_math": ctx.enable_math,
                    "enable_mem_efficient": ctx.enable_mem_efficient,
                }
            except Exception:
                return  # silent fallback: non-critical diagnostic

        try:
            # Determine which backend will be used by priority:
            # Flash > MemEfficient > Math
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            backend = "FlashAttention" if ctx_info["enable_flash"] else (
                "MemEfficient" if ctx_info["enable_mem_efficient"] else "Math"
            )
            print(f"[Attention] backend={backend} | device={device_name} | {ctx_info}")
        except Exception:
            pass  # silent fallback: non-critical diagnostic

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs: Any = None,
        use_cache: bool = False,
        start_pos: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        return_aux_loss: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor], Any],
        Tuple[torch.Tensor, Optional[torch.Tensor], Any, torch.Tensor],
    ]:
        device = idx.device
        b, t = idx.size()
        max_len = self.config.effective_block_size
        assert t <= max_len, f"Cannot forward sequence of length {t}, effective block size is {max_len}"

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)

        new_past_kvs: Any = [] if use_cache else None
        total_aux_loss = torch.zeros((), device=device, dtype=torch.float32)
        moe_enabled = self.config.num_experts > 1
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            if moe_enabled:
                x, present_kv, block_aux = block(
                    x, past_kv=past_kv, use_cache=use_cache, start_pos=start_pos,
                    attention_mask=attention_mask, document_ids=document_ids,
                    return_aux_loss=True,
                )
                total_aux_loss = total_aux_loss + block_aux
            else:
                x, present_kv = block(
                    x, past_kv=past_kv, use_cache=use_cache, start_pos=start_pos,
                    attention_mask=attention_mask, document_ids=document_ids,
                )
            if use_cache:
                assert new_past_kvs is not None
                new_past_kvs.append(present_kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Multi-Token Prediction: compute future-token logits and optional losses.
        future_logits: List[torch.Tensor] = []
        if self.n_future > 0 and not use_cache:
            for head in self.future_heads:
                future_logits.append(head(x))

        loss = None
        mtp_loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if future_logits:
                mtp = torch.zeros((), device=device, dtype=torch.float32)
                T = logits.shape[1]
                for k, f_logits in enumerate(future_logits, start=1):
                    # Position i predicts token targets[i + k] (0-indexed).
                    # Valid positions: i <= T - k - 1.
                    if T <= k:
                        continue
                    pred = f_logits[:, : T - k, :].reshape(-1, f_logits.size(-1))
                    tgt = targets[:, k:].reshape(-1)
                    mtp = mtp + F.cross_entropy(pred, tgt, ignore_index=-1)
                mtp_loss = self.mtp_weight * mtp
                loss = loss + mtp_loss

        if return_aux_loss:
            return logits, loss, new_past_kvs, total_aux_loss
        return logits, loss, new_past_kvs

    def crop_block_size(self, block_size: int) -> None:
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float], device_type: str
    ) -> torch.optim.Optimizer:
        # Deduplicate parameters by tensor id.  wte and lm_head share the same
        # weight tensor, so without deduplication AdamW would update it twice.
        param_dict = {id(p): (n, p) for n, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        import inspect
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        use_cache: bool = False,
        eos_token_id: Optional[int] = None,
        compile: Union[bool, str] = False,
        draft_model: Optional[nn.Module] = None,
        draft_tokens: int = 4,
        draft_temperature: float = 0.0,
        draft_top_k: Optional[int] = 1,
    ) -> torch.Tensor:
        """Autoregressive generation with KV Cache, batch support, and early stopping.

        Parameters
        ----------
        idx : torch.LongTensor [B, T] or [1, T]
            Prompt token ids.  Full batch dimension is supported.
        max_new_tokens : int
            Maximum tokens to generate per sequence.
        temperature : float
            Softmax temperature (1.0=identity, <1.0 sharper, >1.0 flatter).
        top_k : int or None
            Keep only the top-k highest-probability tokens.
        top_p : float or None
            Nucleus sampling: keep tokens until cumulative prob >= ``top_p``.
            Applied after ``top_k``.
        repetition_penalty : float or None
            Divide logits of tokens already seen in *each sequence's own*
            history by this value (>1.0 = discourage repetition).  Applied
            per-sequence independently in batch mode.
        use_cache : bool
            Enable KV-cache for efficient autoregressive decoding.
        eos_token_id : int or None
            If set, each sequence stops as soon as it emits this token.
            Finished slots are padded with ``eos_token_id`` and generation
            terminates early when **all** batch items have finished.
        compile : bool or str
            If ``True``, compile the forward pass with ``torch.compile``
            (``mode="reduce-overhead", fullgraph=False``) to reduce Python-
            side kernel launch overhead in the token-by-token loop.
            If ``"fullgraph"`` (or ``"full"``), compile a single-token decode
            step with ``fullgraph=True`` and replace early termination by a
            per-sequence mask operation.  This is only supported on CUDA,
            with ``use_cache=True``, and when the total length fits in the
            context window without ring-buffer wrap-around.
        draft_model : nn.Module or None
            Optional smaller model for speculative decoding.  When provided,
            ``generate()`` runs a draft-then-verify loop: the draft model
            autoregressively proposes up to ``draft_tokens`` tokens, and the
            target model (``self``) verifies them in a single forward pass.
            Currently only batch size 1 is supported.
        draft_tokens : int
            Number of tokens to draft per speculative step (``gamma``).
        draft_temperature : float
            Sampling temperature for the draft model.  Default ``0.0`` means
            greedy draft (argmax), which is the most common and stable setting.
        draft_top_k : int or None
            Top-k sampling for the draft model.

        Returns
        -------
        idx_out : torch.LongTensor [B, T + generated_len]
            Full sequences (prompt + generated), right-padded with
            ``eos_token_id`` for sequences that finished before the limit.
        """
        B = idx.size(0)
        max_cache_len = self.config.block_size

        # ---- speculative decoding path ----
        if draft_model is not None:
            return self._generate_speculative(
                idx,
                max_new_tokens=max_new_tokens,
                draft_model=draft_model,
                draft_tokens=draft_tokens,
                draft_temperature=draft_temperature,
                draft_top_k=draft_top_k,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
            )

        # ---- normalize compile flag ----
        compile_fullgraph = False
        if isinstance(compile, str):
            compile = compile.strip().lower()
            if compile in ("fullgraph", "full"):
                compile_fullgraph = True
                compile_enabled = True
            else:
                raise ValueError(
                    f"compile must be True, False, or 'fullgraph'; got {compile!r}"
                )
        else:
            compile_enabled = bool(compile)

        # CUDA Graph style optimization: compile is only meaningful on CUDA.
        if compile_enabled and idx.device.type != "cuda":
            import warnings
            warnings.warn(
                "generate(compile=...) is only supported on CUDA; "
                "falling back to eager mode."
            )
            compile_enabled = False
            compile_fullgraph = False

        # ---- fullgraph decode step: compile a single-token forward + sample ----
        if compile_fullgraph:
            can_fullgraph = (
                use_cache
                and repetition_penalty is None
                and getattr(self.config, "sliding_window_size", None) is None
                and not getattr(self.config, "use_paged_kv_cache", False)
                and idx.shape[1] + max_new_tokens <= max_cache_len - 1
                and B >= 1
            )
            if can_fullgraph:
                return self._generate_fullgraph(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=eos_token_id,
                )
            import warnings
            warnings.warn(
                "generate(compile='fullgraph') requirements not met "
                "(use_cache=True, no repetition_penalty/sliding-window/paged-cache, "
                "and prompt+max_new_tokens <= block_size-1); "
                "falling back to reduce-overhead compile."
            )
            compile_enabled = True
            compile_fullgraph = False

        # ---- optional torch.compile graph-mode forward ----
        if compile_enabled:
            if getattr(self, "_compiled_forward", None) is None:
                try:
                    self._compiled_forward = torch.compile(
                        self, mode="reduce-overhead", fullgraph=False, dynamic=True
                    )
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"torch.compile failed for generate() ({e}); "
                        "falling back to eager mode."
                    )
                    self._compiled_forward = self
            mod = self._compiled_forward
        else:
            mod = self

        def _forward(*args, **kwargs):
            """Call ``mod`` and fall back to eager if compilation fails."""
            nonlocal mod
            try:
                return mod(*args, **kwargs)
            except Exception as e:
                if compile_enabled and mod is not self:
                    import warnings
                    warnings.warn(
                        f"Compiled forward failed at generate() step ({e}); "
                        "falling back to eager mode for the remaining tokens."
                    )
                    mod = self
                    self._compiled_forward = self
                    return self(*args, **kwargs)
                raise

        # ---- per-sequence finished mask ----
        if eos_token_id is not None:
            finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        def _all_done():
            return eos_token_id is not None and finished.all().item()

        # ---- helper: token sampling ----
        def _sample(logits):
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k, dim=-1)
                logits[logits < v[:, [-1]]] = -float("Inf")
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                sorted_logits[sorted_mask] = -float("Inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        # ---- helper: per-sequence repetition penalty ----
        def _rep_penalty(logits, history):
            if repetition_penalty is None or repetition_penalty == 1.0:
                return
            for b in range(B):
                for tid in history[b].tolist():
                    logits[b, tid] /= repetition_penalty

        # ---- helper: overwrite finished slots ----
        def _apply_finish(tokens_2d):
            if eos_token_id is None:
                return tokens_2d
            flat = tokens_2d.squeeze(-1)
            flat = torch.where(finished, torch.full_like(flat, eos_token_id), flat)
            return flat.unsqueeze(-1)

        # ---- helper: mark newly finished sequences ----
        def _mark_finished(tokens_2d):
            if eos_token_id is not None:
                nonlocal finished
                finished = finished | (tokens_2d.squeeze(-1) == eos_token_id)

        # ============================================================
        #  No-cache path
        # ============================================================
        if not use_cache:
            idx_out = idx
            for _ in range(max_new_tokens):
                if _all_done():
                    break
                idx_cond = idx_out if idx_out.size(1) <= max_cache_len else idx_out[:, -max_cache_len:]
                # Use absolute positions so the cropped context matches the
                # KV-cache path, which keeps absolute positions across evictions.
                start_pos = max(0, idx_out.size(1) - max_cache_len)
                logits, _, _ = _forward(idx_cond, use_cache=False, start_pos=start_pos)
                logits = logits[:, -1, :]
                _rep_penalty(logits, idx_out)
                idx_next = _sample(logits)
                _mark_finished(idx_next)
                idx_next = _apply_finish(idx_next)
                idx_out = torch.cat([idx_out, idx_next], dim=1)
            return idx_out

        # ============================================================
        #  Cache-enabled path  (prefill + decode)
        # ============================================================
        head_dim = self.config.n_embd // self.config.n_head
        cache: Any
        if getattr(self.config, "use_paged_kv_cache", False):
            from model.paged_kv_cache import PagedKVCacheManager
            cache = PagedKVCacheManager(
                n_layers=self.config.n_layer,
                n_heads=self.config.n_head,
                n_kv_heads=self.config.n_kv_head,
                head_dim=head_dim,
                max_cache_len=max_cache_len,
                block_size=getattr(self.config, "kv_cache_block_size", 16),
            )
        else:
            from model.kv_cache_utils import KVCacheManager
            cache = KVCacheManager(
                n_layers=self.config.n_layer,
                n_heads=self.config.n_head,
                n_kv_heads=self.config.n_kv_head,
                head_dim=head_dim,
                max_cache_len=max_cache_len,
            )
        # KV cache should use the model's activation dtype, not the token-id dtype.
        cache_dtype = next(self.parameters()).dtype
        cache.init_cache(B, idx.device, cache_dtype)

        # --- prefill: encode full prompt, store all K/V ---
        logits, _, raw_kvs = _forward(idx, use_cache=True, start_pos=cache.start_pos)
        for li in range(self.config.n_layer):
            cache.update(li, raw_kvs[li][0], raw_kvs[li][1])
        cache.advance(idx.shape[1])

        # --- first generated token ---
        logits = logits[:, -1, :]
        _rep_penalty(logits, idx)
        idx_next = _sample(logits)                      # [B, 1]
        _mark_finished(idx_next)
        idx_out = torch.cat([idx, idx_next], dim=1)     # [B, prompt_len + 1]

        # --- decode loop ---
        for _ in range(max_new_tokens - 1):
            if _all_done():
                break

            # Feed eos_token to finished sequences so KV-cache stays
            # aligned across the batch (avoids expensive reshape).
            inp = idx_next
            if eos_token_id is not None:
                inp = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(inp, eos_token_id),
                    inp,
                )

            logits, _, next_kv = _forward(
                inp, past_kvs=cache.get_cache(), use_cache=True, start_pos=cache.start_pos
            )
            for li in range(self.config.n_layer):
                cache.update(li, next_kv[li][0], next_kv[li][1])
            cache.advance(1)

            logits = logits[:, -1, :]
            _rep_penalty(logits, idx_out)
            idx_next = _sample(logits)                  # [B, 1]
            _mark_finished(idx_next)
            idx_next = _apply_finish(idx_next)

            idx_out = torch.cat([idx_out, idx_next], dim=1)

        return idx_out

    @torch.no_grad()
    def _generate_fullgraph(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        """Single-token fullgraph compile path used by ``generate(compile='fullgraph')``.

        This path is intentionally narrow: it assumes ``use_cache=True``, no
        repetition penalty, no sliding window / paged cache, and that the whole
        sequence fits in the ring buffer without wrap-around.  Under these
        conditions the decode step (forward + sampling + finished mask) can be
        compiled with ``fullgraph=True`` and executed in a Python loop without
        data-dependent ``break`` statements.
        """
        import warnings

        B = idx.size(0)
        device = idx.device
        max_cache_len = self.config.block_size

        # ---- sampling helper (captured as constants by the compiled step) ----
        def _sample(logits: torch.Tensor) -> torch.Tensor:
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k, dim=-1)
                logits[logits < v[:, [-1]]] = -float("Inf")
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                sorted_logits[sorted_mask] = -float("Inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        eos_tensor: Optional[torch.Tensor] = None
        if eos_token_id is not None:
            eos_tensor = torch.tensor(eos_token_id, dtype=torch.long, device=device)

        def _decode_step(
            x: torch.Tensor,
            past_kvs: List[Tuple[torch.Tensor, torch.Tensor]],
            start_pos: int,
            finished: torch.Tensor,
        ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
            logits, _, new_past = self(
                x,
                past_kvs=past_kvs,
                use_cache=True,
                start_pos=start_pos,
            )
            tok = _sample(logits[:, -1, :])
            # Mask finished sequences so they keep emitting eos_token_id.
            if eos_token_id is not None:
                tok = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(tok, eos_token_id),
                    tok,
                )
            return tok, new_past

        # ---- compile the decode step; on failure fall back to eager ----
        compiled_step: Any = None
        try:
            compiled_step = torch.compile(
                _decode_step,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=True,
            )
        except Exception as e:
            warnings.warn(
                f"torch.compile(fullgraph=True) failed ({e}); "
                "falling back to eager decode step."
            )
            compiled_step = None

        step_fn = compiled_step if compiled_step is not None else _decode_step

        # ---- initialize KV cache ----
        from model.kv_cache_utils import KVCacheManager

        head_dim = self.config.n_embd // self.config.n_head
        cache = KVCacheManager(
            n_layers=self.config.n_layer,
            n_heads=self.config.n_head,
            n_kv_heads=self.config.n_kv_head,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
        )
        cache_dtype = next(self.parameters()).dtype
        cache.init_cache(B, device, cache_dtype)

        # ---- prefill: encode prompt and store K/V ----
        logits, _, raw_kvs = self(idx, use_cache=True, start_pos=cache.start_pos)
        for li in range(self.config.n_layer):
            cache.update(li, raw_kvs[li][0], raw_kvs[li][1])
        cache.advance(idx.shape[1])

        # ---- first token (eager, so the compiled step always sees [B, 1]) ----
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        idx_next = _sample(logits[:, -1, :])
        if eos_token_id is not None:
            finished = finished | (idx_next.squeeze(-1) == eos_token_id)
            idx_next = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(idx_next, eos_token_id),
                idx_next,
            )
        idx_out = torch.cat([idx, idx_next], dim=1)

        # ---- fullgraph-friendly decode loop: no data-dependent break ----
        for _ in range(max_new_tokens - 1):
            # For finished sequences feed eos_token_id so the KV cache stays
            # aligned (the compiled step will also mask the sampled token).
            inp = idx_next
            if eos_token_id is not None:
                inp = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(inp, eos_token_id),
                    inp,
                )

            try:
                tok, next_kv = step_fn(
                    inp,
                    cache.get_cache(),
                    cache.start_pos,
                    finished,
                )
            except Exception as e:
                # Compilation may fail lazily on the first call (e.g. missing
                # Triton on Windows).  Fall back to the eager step function.
                if step_fn is not _decode_step:
                    warnings.warn(
                        f"Compiled decode step failed at runtime ({e}); "
                        "falling back to eager step."
                    )
                    step_fn = _decode_step
                    tok, next_kv = step_fn(
                        inp,
                        cache.get_cache(),
                        cache.start_pos,
                        finished,
                    )
                else:
                    raise

            for li in range(self.config.n_layer):
                cache.update(li, next_kv[li][0], next_kv[li][1])
            cache.advance(1)

            if eos_token_id is not None:
                finished = finished | (tok.squeeze(-1) == eos_token_id)
            idx_out = torch.cat([idx_out, tok], dim=1)

        return idx_out

    @torch.no_grad()
    def _generate_speculative(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        draft_model: nn.Module,
        draft_tokens: int,
        draft_temperature: float,
        draft_top_k: Optional[int],
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        """Speculative decoding (draft-then-verify) for batch size 1.

        The draft model proposes up to ``draft_tokens`` tokens autoregressively;
        the target model (``self``) verifies them with a single forward pass and
        accepts or rejects each proposal using the standard acceptance ratio.
        When all proposals are accepted, one additional target token is sampled.
        """
        from model.kv_cache_utils import KVCacheManager

        B = idx.size(0)
        if B != 1:
            raise ValueError(
                f"Speculative decoding currently supports batch size 1, got {B}"
            )
        device = idx.device
        prompt_len = idx.size(1)
        max_cache_len = self.config.block_size
        if prompt_len + max_new_tokens > max_cache_len - 1:
            raise ValueError(
                "Speculative decoding requires prompt_len + max_new_tokens <= block_size - 1"
            )

        def _sample(logits: torch.Tensor, temp: float, k: Optional[int], p: Optional[float]) -> torch.Tensor:
            if temp > 0 and temp != 1.0:
                logits = logits / temp
            if k is not None and k > 0:
                kk = min(k, logits.size(-1))
                v, _ = torch.topk(logits, kk, dim=-1)
                logits[logits < v[:, [-1]]] = -float("Inf")
            if p is not None and p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cum_probs > p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                sorted_logits[sorted_mask] = -float("Inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        def _draft_sample(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Greedy or sampled draft token + its probability under the draft model.

            The returned probability is taken from the *original* softmax distribution
            (before top-k masking) so that the speculative acceptance ratio is
            mathematically correct.
            """
            if draft_temperature is not None and draft_temperature > 0 and draft_temperature != 1.0:
                logits = logits / draft_temperature
            # Original probabilities used for the acceptance ratio.
            orig_probs = F.softmax(logits, dim=-1)
            # Top-k mask is applied only for sampling.
            sample_logits = logits
            if draft_top_k is not None and draft_top_k > 0:
                sample_logits = logits.clone()
                kk = min(draft_top_k, sample_logits.size(-1))
                v, _ = torch.topk(sample_logits, kk, dim=-1)
                sample_logits[sample_logits < v[:, [-1]]] = -float("Inf")
            sample_probs = F.softmax(sample_logits, dim=-1)
            if draft_temperature == 0.0:
                tok = sample_probs.argmax(dim=-1, keepdim=True)
            else:
                tok = torch.multinomial(sample_probs, num_samples=1)
            p_tok = orig_probs.gather(-1, tok).squeeze(-1)
            return tok, p_tok

        head_dim = self.config.n_embd // self.config.n_head
        dtype = next(self.parameters()).dtype

        target_cache = KVCacheManager(
            self.config.n_layer, self.config.n_head, self.config.n_kv_head, head_dim, max_cache_len
        )
        draft_cache = KVCacheManager(
            draft_model.config.n_layer,
            draft_model.config.n_head,
            draft_model.config.n_kv_head,
            draft_model.config.n_embd // draft_model.config.n_head,
            max_cache_len,
        )
        target_cache.init_cache(B, device, dtype)
        draft_cache.init_cache(B, device, dtype)

        # ---- prefill both models with the prompt ----
        logits_t_prefill, _, raw_kvs_t = self(idx, use_cache=True, start_pos=0)
        logits_d_prefill, _, raw_kvs_d = draft_model(idx, use_cache=True, start_pos=0)
        for li in range(self.config.n_layer):
            target_cache.update(li, raw_kvs_t[li][0], raw_kvs_t[li][1])
        for li in range(draft_model.config.n_layer):
            draft_cache.update(li, raw_kvs_d[li][0], raw_kvs_d[li][1])
        target_cache.advance(prompt_len)
        draft_cache.advance(prompt_len)

        # ``current_*_logits`` is the distribution for the *next* token at the
        # current context boundary, produced by the last forward call.
        current_target_logits = logits_t_prefill[:, -1, :]  # [B, V]
        current_draft_logits = logits_d_prefill[:, -1, :]   # [B, V]

        idx_out = idx
        generated = 0

        while generated < max_new_tokens:
            context_len_before = idx_out.size(1)

            # ---- draft phase: generate up to draft_tokens tokens ----
            draft_chunk: List[torch.Tensor] = []
            draft_probs: List[torch.Tensor] = []
            gamma = min(draft_tokens, max_new_tokens - generated)
            hit_eos = False
            logits_for_next = current_draft_logits
            for i in range(gamma):
                tok, p_tok = _draft_sample(logits_for_next)
                draft_chunk.append(tok)
                draft_probs.append(p_tok)
                # Run the draft model with the sampled token to obtain its KV
                # and the logits for the following position.
                logits_d, _, next_kv_d = draft_model(
                    tok,
                    past_kvs=draft_cache.get_cache(),
                    use_cache=True,
                    start_pos=draft_cache.start_pos,
                )
                for li in range(draft_model.config.n_layer):
                    draft_cache.update(li, next_kv_d[li][0], next_kv_d[li][1])
                draft_cache.advance(1)
                logits_for_next = logits_d[:, -1, :]
                if eos_token_id is not None and tok.item() == eos_token_id:
                    hit_eos = True
                    break
            gamma = len(draft_chunk)
            if gamma == 0:
                break
            draft_ids = torch.cat(draft_chunk, dim=1)  # [B, gamma]

            # ---- verification phase: target model scores all draft tokens ----
            logits_t, _, next_kv_t = self(
                draft_ids,
                past_kvs=target_cache.get_cache(),
                use_cache=True,
                start_pos=target_cache.start_pos,
            )
            # logits_t[:, i, :] is p_t(· | context + d_1..d_{i+1}), i.e. it
            # predicts d_{i+2}.  Therefore p_t(d_i | context + d_1..d_{i-1})
            # is current_target_logits for i==0 and logits_t[:, i-1, :] for i>=1.

            # ---- acceptance loop ----
            accepted_tokens: List[torch.Tensor] = []
            n_accepted = 0
            rejected = False
            for i in range(gamma):
                tok_i = draft_ids[:, i]  # [B]
                if i == 0:
                    logits_for_tok = current_target_logits
                else:
                    logits_for_tok = logits_t[:, i - 1, :]
                p_t = F.softmax(logits_for_tok, dim=-1)[:, tok_i]
                p_d = draft_probs[i]
                ratio = torch.clamp(p_t / p_d, max=1.0)
                accept = torch.rand(B, device=device) < ratio
                if accept.all().item():
                    accepted_tokens.append(tok_i.unsqueeze(-1))
                    n_accepted += 1
                    if hit_eos and i == gamma - 1:
                        # Draft hit eos at its last generated position and the
                        # target agrees; generation ends after appending eos.
                        break
                else:
                    # Rejection: sample from q(x) = normalize(max(0, p_target(x) - p_draft(x)))
                    p_t_dist = F.softmax(logits_for_tok[0, :], dim=-1)  # [V]
                    p_d_dist = torch.zeros_like(p_t_dist)
                    p_d_dist[tok_i[0]] = p_d[0]
                    q = torch.relu(p_t_dist - p_d_dist)
                    q = q / q.sum()
                    replace = torch.multinomial(q, num_samples=1).unsqueeze(0)  # [1, 1]
                    accepted_tokens.append(replace)
                    rejected = True
                    break
            else:
                # All draft tokens accepted.  Sample one extra token only if we
                # have not already reached the generation budget.
                if generated + gamma < max_new_tokens:
                    extra = _sample(logits_t[:, -1, :], temperature, top_k, top_p)
                    accepted_tokens.append(extra)

            # ---- append accepted tokens to output ----
            for tok_t in accepted_tokens:
                idx_out = torch.cat([idx_out, tok_t], dim=1)
                generated += 1
                if eos_token_id is not None and tok_t.item() == eos_token_id:
                    return idx_out

            # ---- update target KV cache ----
            if rejected:
                if n_accepted > 0:
                    for li in range(self.config.n_layer):
                        k = next_kv_t[li][0][:, :, :n_accepted, :]
                        v = next_kv_t[li][1][:, :, :n_accepted, :]
                        target_cache.update(li, k, v)
                    target_cache.advance(n_accepted)
                replace_tok = accepted_tokens[n_accepted]
                logits_replace, _, replace_kv_t = self(
                    replace_tok,
                    past_kvs=target_cache.get_cache(),
                    use_cache=True,
                    start_pos=target_cache.start_pos,
                )
                for li in range(self.config.n_layer):
                    target_cache.update(li, replace_kv_t[li][0], replace_kv_t[li][1])
                target_cache.advance(1)
                current_target_logits = logits_replace[:, -1, :]
            else:
                # All gamma accepted, plus one extra token.
                for li in range(self.config.n_layer):
                    target_cache.update(li, next_kv_t[li][0], next_kv_t[li][1])
                target_cache.advance(gamma)
                extra_tok = accepted_tokens[-1]
                logits_extra, _, extra_kv_t = self(
                    extra_tok,
                    past_kvs=target_cache.get_cache(),
                    use_cache=True,
                    start_pos=target_cache.start_pos,
                )
                for li in range(self.config.n_layer):
                    target_cache.update(li, extra_kv_t[li][0], extra_kv_t[li][1])
                target_cache.advance(1)
                current_target_logits = logits_extra[:, -1, :]

            # ---- update draft KV cache to match the new context ----
            if rejected:
                draft_cache.truncate(context_len_before + n_accepted)
                replace_tok = accepted_tokens[n_accepted]
                logits_d_replace, _, replace_kv_d = draft_model(
                    replace_tok,
                    past_kvs=draft_cache.get_cache(),
                    use_cache=True,
                    start_pos=draft_cache.start_pos,
                )
                for li in range(draft_model.config.n_layer):
                    draft_cache.update(li, replace_kv_d[li][0], replace_kv_d[li][1])
                draft_cache.advance(1)
                current_draft_logits = logits_d_replace[:, -1, :]
            else:
                extra_tok = accepted_tokens[-1]
                logits_d_extra, _, extra_kv_d = draft_model(
                    extra_tok,
                    past_kvs=draft_cache.get_cache(),
                    use_cache=True,
                    start_pos=draft_cache.start_pos,
                )
                for li in range(draft_model.config.n_layer):
                    draft_cache.update(li, extra_kv_d[li][0], extra_kv_d[li][1])
                draft_cache.advance(1)
                current_draft_logits = logits_d_extra[:, -1, :]

        return idx_out

    # ------------------------------------------------------------------
    #  EMA (Exponential Moving Average)
    # ------------------------------------------------------------------
    def init_ema(self, decay=0.999):
        """Initialise shadow weights for EMA.
        Call once before training. After each optimizer step, call
        ``update_ema()`` to maintain the shadow copy.
        """
        self.ema_decay = decay
        self.ema_shadow = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.ema_shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update_ema(self):
        """Update EMA shadow weights after an optimizer step."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.ema_shadow[name].mul_(self.ema_decay).add_(
                    param.data, alpha=1.0 - self.ema_decay
                )

    def apply_ema_weights(self):
        """Swap current weights with EMA shadow (for evaluation)."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                param.data, self.ema_shadow[name] = (
                    self.ema_shadow[name],
                    param.data,
                )

    def restore_ema_weights(self):
        """Swap back after EMA evaluation."""
        self.apply_ema_weights()

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
