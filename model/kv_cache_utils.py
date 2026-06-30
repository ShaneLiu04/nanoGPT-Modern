"""KV Cache utilities for autoregressive generation with optional quantization.

This module provides a static, ring-buffer-based KV cache manager.  Instead of
concatenating new keys/values onto the cache at every step (``O(cache_len)``
copy cost and memory fragmentation), the cache is pre-allocated as
``[B, n_kv_heads, max_cache_len, head_dim]`` and written via position pointers.
Data movement only happens inside :meth:`KVCacheManager.get_cache` when the
logical sequence wraps around the physical buffer, in which case two slices are
concatenated.

Quantization Support
--------------------
The manager supports ``cache_dtype`` parameter that controls the storage dtype
of the KV cache tensors:

* ``fp16`` / ``bf16`` — store in the model's native half-precision (default).
* ``int8`` — 8-bit per-channel (head_dim) quantization with per-channel scale.
* ``fp8`` — 8-bit floating-point storage (E4M3 / E5M2) when PyTorch supports it.

When a quantized dtype is selected, :meth:`update` transparently quantizes the
incoming K/V tensors before writing them into the ring buffer, and
:meth:`get_cache` dequantizes them back to the model's compute dtype on the fly.
This reduces peak memory by roughly 50 % (INT8) with minimal impact on
generation quality for most models.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import torch


class KVCacheManager:
    """Ring-buffer-based KV cache manager with sliding-window eviction.

    Each layer owns one pre-allocated tensor pair ``(K, V)`` of shape
    ``[B, n_kv_heads, max_cache_len, head_dim]``.  New keys/values are written
    via slicing at the current write pointer; when the buffer fills up the
    oldest tokens are logically evicted by advancing ``start_pos`` and the
    write pointer wraps around.  No per-step ``torch.cat`` is performed in
    :meth:`update` / :meth:`advance`.

    Parameters
    ----------
    n_layers : int
    n_heads : int
        Number of query heads (informational / validation only).
    n_kv_heads : int
        Number of key/value heads.  May be smaller than ``n_heads`` for GQA.
    head_dim : int
    max_cache_len : int
        Maximum number of tokens to keep in the cache.  Older tokens are
        dropped when the cache is full.
    cache_dtype : str
        One of ``fp16``, ``bf16``, ``int8``, ``fp8``.  ``fp16`` and ``bf16``
        store the cache in the model's native compute dtype (no quantization).
        ``int8`` uses per-channel INT8 quantization with a scale factor per
        head and token.  ``fp8`` uses 8-bit float storage when supported.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_cache_len: int = 1024,
        cache_dtype: str = "bf16",
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len
        self.cache_dtype = cache_dtype.lower()

        self._k: Optional[List[torch.Tensor]] = None
        self._v: Optional[List[torch.Tensor]] = None
        self._k_scale: Optional[List[torch.Tensor]] = None
        self._v_scale: Optional[List[torch.Tensor]] = None
        self._batch_size: Optional[int] = None
        self._device: Optional[torch.device] = None
        self._compute_dtype: Optional[torch.dtype] = None

        self._cache_len = 0        # logical number of tokens in cache
        self._write_pos = 0        # next physical position to write (0 .. max_cache_len)
        self.start_pos = 0         # absolute position of logical first token

    @classmethod
    def from_config(cls, config, cache_dtype: str = "bf16"):
        """Factory: build a KVCacheManager from a ModernGPTConfig."""
        head_dim = config.n_embd // config.n_head
        return cls(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            n_kv_heads=config.n_kv_head,
            head_dim=head_dim,
            max_cache_len=getattr(config, "effective_block_size", config.block_size),
            cache_dtype=cache_dtype,
        )

    def _quantization_enabled(self) -> bool:
        return self.cache_dtype in ("int8", "fp8")

    def _storage_dtype(self) -> torch.dtype:
        if self.cache_dtype == "int8":
            return torch.int8
        if self.cache_dtype == "fp8":
            # Prefer E4M3 when available (PyTorch 2.1+), fall back to E5M2.
            if hasattr(torch, "float8_e4m3fn"):
                return torch.float8_e4m3fn
            if hasattr(torch, "float8_e5m2"):
                return torch.float8_e5m2
            return torch.int8  # graceful fallback
        if self.cache_dtype in ("fp16", "float16"):
            return torch.float16
        return torch.bfloat16

    def _quantize_cache(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize a K or V tensor to the configured storage dtype.

        For ``int8`` we use per-channel (per ``head_dim``) scaling:

        .. math::
            scale = max(|tensor|) / 127
            q = clamp(round(tensor / scale), -128, 127)

        The scale tensor has shape ``[B, n_kv_heads, S, 1]`` so it can be stored
        alongside the quantized cache and reused during dequantization.

        Parameters
        ----------
        tensor : torch.Tensor
            K or V of shape ``[B, n_kv_heads, S, head_dim]``.

        Returns
        -------
        q : torch.Tensor
            Quantized tensor, same rank but dtype ``int8``.
        scale : torch.Tensor
            Per-channel scale, shape ``[B, n_kv_heads, S, 1]``, dtype fp32.
        """
        if self.cache_dtype == "int8":
            # Per-channel (head_dim) symmetric quantization.
            # Compute max abs over head_dim, keep dims for broadcasting.
            abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-6)
            scale = abs_max / 127.0
            q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
            return q, scale
        if self.cache_dtype == "fp8":
            # Use PyTorch's native float8 conversion if available.
            if hasattr(torch, "float8_e4m3fn"):
                q = tensor.to(torch.float8_e4m3fn)
                # Scale is identity for native fp8; we keep a placeholder
                # so the API is uniform.
                scale = torch.ones_like(tensor[:, :, :, :1], dtype=torch.float32)
                return q, scale
            # Graceful fallback to int8.
            return self._quantize_cache_int8_fallback(tensor)
        # No quantization for fp16/bf16.
        return tensor, torch.ones_like(tensor[:, :, :, :1], dtype=torch.float32)

    def _quantize_cache_int8_fallback(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback int8 quantization when fp8 is requested but unsupported."""
        abs_max = tensor.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-6)
        scale = abs_max / 127.0
        q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
        return q, scale

    def _dequantize_cache(
        self, q: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize a cached K or V tensor back to the compute dtype.

        Parameters
        ----------
        q : torch.Tensor
            Quantized tensor, dtype ``int8`` or ``float8``.
        scale : torch.Tensor
            Per-channel scale from :meth:`_quantize_cache`.

        Returns
        -------
        tensor : torch.Tensor
            Dequantized tensor in ``self._compute_dtype``.
        """
        if q.dtype == torch.int8:
            return q.to(self._compute_dtype) * scale.to(self._compute_dtype)
        if "float8" in str(q.dtype):
            return q.to(self._compute_dtype)
        return q.to(self._compute_dtype)

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """Allocate empty cache tensors for each layer."""
        self._batch_size = batch_size
        self._device = device
        self._compute_dtype = dtype
        is_quantized = self._quantization_enabled()
        # For non-quantized paths, use the compute dtype directly so that
        # dtype round-trips (e.g. float32 -> float32) remain bit-exact and
        # backward-compatible with the original KVCacheManager behaviour.
        storage_dtype = self._storage_dtype() if is_quantized else dtype

        self._k = [
            torch.zeros(
                batch_size, self.n_kv_heads, self.max_cache_len, self.head_dim,
                device=device, dtype=storage_dtype,
            )
            for _ in range(self.n_layers)
        ]
        self._v = [
            torch.zeros(
                batch_size, self.n_kv_heads, self.max_cache_len, self.head_dim,
                device=device, dtype=storage_dtype,
            )
            for _ in range(self.n_layers)
        ]
        if is_quantized:
            self._k_scale = [
                torch.ones(
                    batch_size, self.n_kv_heads, self.max_cache_len, 1,
                    device=device, dtype=torch.float32,
                )
                for _ in range(self.n_layers)
            ]
            self._v_scale = [
                torch.ones(
                    batch_size, self.n_kv_heads, self.max_cache_len, 1,
                    device=device, dtype=torch.float32,
                )
                for _ in range(self.n_layers)
            ]
        else:
            self._k_scale = None
            self._v_scale = None

        self._cache_len = 0
        self._write_pos = 0
        self.start_pos = 0

    def reset_cache(self) -> None:
        """Clear all cached keys and values."""
        if self._k is not None:
            assert self._v is not None
            for i in range(self.n_layers):
                self._k[i].zero_()
                self._v[i].zero_()
                if self._k_scale is not None:
                    assert self._v_scale is not None
                    self._k_scale[i].fill_(1.0)
                    self._v_scale[i].fill_(1.0)
        self._cache_len = 0
        self._write_pos = 0
        self.start_pos = 0

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Write new K/V at the current logical end of the cache.

        ``new_k`` and ``new_v`` must have shape ``[B, n_kv_heads, S, head_dim]``.
        This method does **not** advance the logical length; call
        :meth:`advance` after all layers have been updated.

        The write uses position pointers and wraps around the physical buffer
        without ``torch.cat``.  If quantization is enabled, incoming tensors are
        quantized before storage and dequantized on read in :meth:`get_cache`.
        """
        if self._k is None:
            raise RuntimeError("Cache not initialized. Call init_cache first.")
        assert self._v is not None

        new_len = new_k.shape[2]
        if new_len == 0:
            return

        if new_len > self.max_cache_len:
            # The incoming chunk is larger than the whole cache.  Keep only the
            # last ``max_cache_len`` tokens and reset the logical state so that
            # advance() can place them correctly.
            new_k = new_k[:, :, -self.max_cache_len:, :]
            new_v = new_v[:, :, -self.max_cache_len:, :]
            new_len = self.max_cache_len
            # Reset logical state; advance() will recompute start_pos/write_pos.
            self._cache_len = 0
            self._write_pos = 0
            self.start_pos = 0

        # Quantize before writing when enabled.
        if self._quantization_enabled():
            q_k, s_k = self._quantize_cache(new_k)
            q_v, s_v = self._quantize_cache(new_v)
        else:
            q_k, q_v = new_k, new_v
            s_k = None
            s_v = None

        # Write with wrap-around.
        end_pos = self._write_pos + new_len
        if end_pos <= self.max_cache_len:
            self._k[layer_idx][:, :, self._write_pos:end_pos, :] = q_k
            self._v[layer_idx][:, :, self._write_pos:end_pos, :] = q_v
            if s_k is not None:
                assert self._k_scale is not None and self._v_scale is not None
                assert s_v is not None
                self._k_scale[layer_idx][:, :, self._write_pos:end_pos, :] = s_k
                self._v_scale[layer_idx][:, :, self._write_pos:end_pos, :] = s_v
        else:
            first_part = self.max_cache_len - self._write_pos
            self._k[layer_idx][:, :, self._write_pos:, :] = q_k[:, :, :first_part, :]
            self._k[layer_idx][:, :, :end_pos - self.max_cache_len, :] = q_k[:, :, first_part:, :]
            self._v[layer_idx][:, :, self._write_pos:, :] = q_v[:, :, :first_part, :]
            self._v[layer_idx][:, :, :end_pos - self.max_cache_len, :] = q_v[:, :, first_part:, :]
            if s_k is not None:
                assert self._k_scale is not None and self._v_scale is not None
                assert s_v is not None
                self._k_scale[layer_idx][:, :, self._write_pos:, :] = s_k[:, :, :first_part, :]
                self._k_scale[layer_idx][:, :, :end_pos - self.max_cache_len, :] = s_k[:, :, first_part:, :]
                self._v_scale[layer_idx][:, :, self._write_pos:, :] = s_v[:, :, :first_part, :]
                self._v_scale[layer_idx][:, :, :end_pos - self.max_cache_len, :] = s_v[:, :, first_part:, :]

    def advance(self, delta: int) -> None:
        """Advance the logical cache length after all layers are written.

        ``delta`` should match the sequence length of the K/V chunks passed to
        :meth:`update`.  If the new logical length would exceed
        ``max_cache_len - 1``, the oldest tokens are evicted by updating
        ``start_pos`` (no data movement).

        We keep one slot free so that the next decode step can append a single
        new token without exceeding ``max_cache_len`` total context.  This makes
        the cached path identical to the no-cache path that crops to the last
        ``max_cache_len`` tokens.
        """
        if delta <= 0:
            return

        new_total = self._cache_len + delta
        # Reserve one slot for the next single-token decode step.
        capacity = self.max_cache_len - 1
        if new_total > capacity:
            evicted = new_total - capacity
            self.start_pos += evicted
            self._cache_len = capacity
        else:
            self._cache_len = new_total

        self._write_pos = (self._write_pos + delta) % self.max_cache_len

    def get_cache(self) -> List[Any]:
        """Return cached K/V as a list of contiguous blocks.

        Returns a list of ``(k_block, v_block)`` tuples.  Most of the time the
        logical sequence is physically contiguous and the list contains a single
        block; when the ring buffer wraps around, two blocks are returned and
        the caller concatenates them.

        If quantization is enabled, blocks are dequantized on the fly back to
        the model's compute dtype so that downstream attention kernels receive
        the expected floating-point tensors.
        """
        if self._k is None or self._cache_len == 0:
            return []

        assert self._v is not None

        is_quantized = self._quantization_enabled()

        def _dequant_layer(layer_idx: int, start: int, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
            assert self._k is not None and self._v is not None
            k_q = self._k[layer_idx][:, :, start:end, :]
            v_q = self._v[layer_idx][:, :, start:end, :]
            if is_quantized and self._k_scale is not None:
                assert self._v_scale is not None
                s_k = self._k_scale[layer_idx][:, :, start:end, :]
                s_v = self._v_scale[layer_idx][:, :, start:end, :]
                k = self._dequantize_cache(k_q, s_k)
                v = self._dequantize_cache(v_q, s_v)
            else:
                k = k_q.to(self._compute_dtype)
                v = v_q.to(self._compute_dtype)
            return k, v

        # Logical sequence occupies ``_cache_len`` slots ending at ``_write_pos``.
        if self._cache_len <= self._write_pos:
            start = self._write_pos - self._cache_len
            return [
                (_dequant_layer(i, start, self._write_pos))
                for i in range(self.n_layers)
            ]
        else:
            # Wrapped: two contiguous blocks in physical memory that concatenate
            # to the logical sequence.
            tail_len = self._cache_len - self._write_pos
            return [
                [
                    _dequant_layer(i, -tail_len, self.max_cache_len),
                    _dequant_layer(i, 0, self._write_pos),
                ]
                for i in range(self.n_layers)
            ]

    def get_cache_contiguous(self) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Return cached K/V as single contiguous tensors per layer.

        Convenience wrapper around :meth:`get_cache` that concatenates wrapped
        blocks.  Useful for state saving or when the caller requires a single
        tensor per layer.
        """
        blocks = self.get_cache()
        if not blocks:
            return None
        return [
            (
                torch.cat([b[0] for b in layer_blocks], dim=2),
                torch.cat([b[1] for b in layer_blocks], dim=2),
            )
            for layer_blocks in blocks
        ]

    def set_cache(self, cache: Any) -> None:
        """Replace the cache contents (used for state restore/tests).

        ``cache`` must be a list of length ``n_layers``.  Each element is either
        a single ``(k, v)`` tuple or a list of blocks ``[(k, v), ...]`` as
        returned by :meth:`get_cache`.
        """
        if not cache:
            self.reset_cache()
            return

        assert self._k is not None and self._v is not None

        # Normalize each layer to a list of blocks.
        normalized = []
        for layer_cache in cache:
            if isinstance(layer_cache, tuple):
                normalized.append([layer_cache])
            else:
                normalized.append(list(layer_cache))

        # Trim from the front if the total length exceeds capacity.
        # We keep at most ``max_cache_len - 1`` tokens so that the next decode
        # step can append one new token without exceeding the context window.
        total_len = sum(b[0].shape[2] for b in normalized[0])
        capacity = self.max_cache_len - 1
        drop = 0
        if total_len > capacity:
            drop = total_len - capacity
            trimmed: List[List[Tuple[torch.Tensor, torch.Tensor]]] = [[] for _ in normalized]
            remaining = drop
            for block_idx, block in enumerate(normalized[0]):
                L = block[0].shape[2]
                if L <= remaining:
                    remaining -= L
                else:
                    skip = remaining
                    for i in range(self.n_layers):
                        k = normalized[i][block_idx][0]
                        v = normalized[i][block_idx][1]
                        trimmed[i].append((k[:, :, skip:, :], v[:, :, skip:, :]))
                    remaining = 0
                if remaining == 0:
                    # Append remaining blocks unchanged.
                    for j in range(block_idx + 1, len(normalized[0])):
                        for i in range(self.n_layers):
                            trimmed[i].append(normalized[i][j])
                    break
            normalized = trimmed
            total_len = capacity

        # Write blocks sequentially starting at physical position 0.
        self._cache_len = 0
        self._write_pos = 0
        self.start_pos = drop
        for block_idx in range(len(normalized[0])):
            L = normalized[0][block_idx][0].shape[2]
            for i in range(self.n_layers):
                k_block = normalized[i][block_idx][0]
                v_block = normalized[i][block_idx][1]
                # Quantize on-the-fly during restore if needed.
                if self._quantization_enabled():
                    q_k, s_k = self._quantize_cache(k_block)
                    q_v, s_v = self._quantize_cache(v_block)
                    self._k[i][:, :, self._write_pos:self._write_pos + L, :] = q_k
                    self._v[i][:, :, self._write_pos:self._write_pos + L, :] = q_v
                    if self._k_scale is not None:
                        assert self._v_scale is not None
                        self._k_scale[i][:, :, self._write_pos:self._write_pos + L, :] = s_k
                        self._v_scale[i][:, :, self._write_pos:self._write_pos + L, :] = s_v
                else:
                    self._k[i][:, :, self._write_pos:self._write_pos + L, :] = k_block.to(
                        self._k[i].dtype
                    )
                    self._v[i][:, :, self._write_pos:self._write_pos + L, :] = v_block.to(
                        self._v[i].dtype
                    )
            self._write_pos += L
        self._cache_len = total_len

    @property
    def cache_len(self) -> int:
        return self._cache_len

    def truncate(self, new_cache_len: int) -> None:
        """Truncate the logical cache to ``new_cache_len`` tokens.

        This is a helper for speculative decoding: after a rejection we want
        to keep only the accepted prefix of the draft cache.  It is only
        valid before the ring buffer wraps around (``start_pos == 0``).
        """
        if new_cache_len < 0 or new_cache_len > self._cache_len:
            raise ValueError(
                f"new_cache_len ({new_cache_len}) must be in [0, {self._cache_len}]"
            )
        if self.start_pos != 0:
            raise RuntimeError("truncate() is only supported before ring-buffer wrap-around")
        self._cache_len = new_cache_len
        self._write_pos = new_cache_len % self.max_cache_len


class QuantizedKVCacheManager(KVCacheManager):
    """Drop-in subclass that forces INT8 quantization for the KV cache.

    This is a convenience alias that keeps the exact same API as
    ``KVCacheManager`` but hard-codes ``cache_dtype="int8"``.  It can be used
    wherever ``KVCacheManager`` is accepted without changing call sites.

    Example
    -------
    >>> cache = QuantizedKVCacheManager(n_layers=12, n_heads=8, n_kv_heads=2,
    ...                                 head_dim=64, max_cache_len=2048)
    >>> cache.init_cache(batch_size=1, device="cuda", dtype=torch.bfloat16)
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_cache_len: int = 1024,
    ):
        super().__init__(
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_cache_len=max_cache_len,
            cache_dtype="int8",
        )

    @classmethod
    def from_config(cls, config):
        """Factory: build a QuantizedKVCacheManager from a ModernGPTConfig."""
        head_dim = config.n_embd // config.n_head
        return cls(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            n_kv_heads=config.n_kv_head,
            head_dim=head_dim,
            max_cache_len=getattr(config, "effective_block_size", config.block_size),
        )


def build_sliding_window_mask(seq_len: int, window_size: int, device: Union[str, torch.device]) -> torch.Tensor:
    """Build a causal + sliding window mask for training or inference."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start : i + 1] = 0.0
    return mask
