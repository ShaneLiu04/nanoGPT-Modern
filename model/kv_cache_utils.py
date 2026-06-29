"""
KV Cache utilities for autoregressive generation.

This module provides a static, ring-buffer-based KV cache manager.  Instead of
concatenating new keys/values onto the cache at every step (``O(cache_len)``
copy cost and memory fragmentation), the cache is pre-allocated as
``[B, n_kv_heads, max_cache_len, head_dim]`` and written via position pointers.
Data movement only happens inside :meth:`KVCacheManager.get_cache` when the
logical sequence wraps around the physical buffer, in which case two slices are
concatenated.

For callers that need a simpler contiguous cache, the manager also exposes
:meth:`get_cache_contiguous`.
"""
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

    Attributes
    ----------
    start_pos : int
        Absolute position of the first token currently stored in the cache.
    """

    def __init__(self, n_layers, n_heads, n_kv_heads, head_dim, max_cache_len=1024):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_cache_len = max_cache_len

        self._k = None
        self._v = None
        self._batch_size = None
        self._device = None
        self._dtype = None

        self._cache_len = 0        # logical number of tokens in cache
        self._write_pos = 0        # next physical position to write (0 .. max_cache_len)
        self.start_pos = 0         # absolute position of logical first token

    @classmethod
    def from_config(cls, config):
        """Factory: build a KVCacheManager from a ModernGPTConfig."""
        head_dim = config.n_embd // config.n_head
        return cls(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            n_kv_heads=config.n_kv_head,
            head_dim=head_dim,
            max_cache_len=getattr(config, "effective_block_size", config.block_size),
        )

    def init_cache(self, batch_size, device, dtype):
        """Allocate empty cache tensors for each layer."""
        self._batch_size = batch_size
        self._device = device
        self._dtype = dtype
        self._k = [
            torch.zeros(
                batch_size, self.n_kv_heads, self.max_cache_len, self.head_dim,
                device=device, dtype=dtype,
            )
            for _ in range(self.n_layers)
        ]
        self._v = [
            torch.zeros(
                batch_size, self.n_kv_heads, self.max_cache_len, self.head_dim,
                device=device, dtype=dtype,
            )
            for _ in range(self.n_layers)
        ]
        self._cache_len = 0
        self._write_pos = 0
        self.start_pos = 0

    def reset_cache(self):
        """Clear all cached keys and values."""
        if self._k is not None:
            for i in range(self.n_layers):
                self._k[i].zero_()
                self._v[i].zero_()
        self._cache_len = 0
        self._write_pos = 0
        self.start_pos = 0

    def update(self, layer_idx, new_k, new_v):
        """Write new K/V at the current logical end of the cache.

        ``new_k`` and ``new_v`` must have shape ``[B, n_kv_heads, S, head_dim]``.
        This method does **not** advance the logical length; call
        :meth:`advance` after all layers have been updated.

        The write uses position pointers and wraps around the physical buffer
        without ``torch.cat``.
        """
        if self._k is None:
            raise RuntimeError("Cache not initialized. Call init_cache first.")

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

        # Write with wrap-around.
        end_pos = self._write_pos + new_len
        if end_pos <= self.max_cache_len:
            self._k[layer_idx][:, :, self._write_pos:end_pos, :] = new_k
            self._v[layer_idx][:, :, self._write_pos:end_pos, :] = new_v
        else:
            first_part = self.max_cache_len - self._write_pos
            self._k[layer_idx][:, :, self._write_pos:, :] = new_k[:, :, :first_part, :]
            self._k[layer_idx][:, :, :end_pos - self.max_cache_len, :] = new_k[:, :, first_part:, :]
            self._v[layer_idx][:, :, self._write_pos:, :] = new_v[:, :, :first_part, :]
            self._v[layer_idx][:, :, :end_pos - self.max_cache_len, :] = new_v[:, :, first_part:, :]

    def advance(self, delta):
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

    def get_cache(self):
        """Return cached K/V as a list of contiguous blocks.

        Returns a list of ``(k_block, v_block)`` tuples.  Most of the time the
        logical sequence is physically contiguous and the list contains a single
        block; when the ring buffer wraps around, two blocks are returned and
        the caller concatenates them.
        """
        if self._k is None or self._cache_len == 0:
            return []

        # Logical sequence occupies ``_cache_len`` slots ending at ``_write_pos``.
        if self._cache_len <= self._write_pos:
            start = self._write_pos - self._cache_len
            return [
                (self._k[i][:, :, start:self._write_pos, :],
                 self._v[i][:, :, start:self._write_pos, :])
                for i in range(self.n_layers)
            ]
        else:
            # Wrapped: two contiguous blocks in physical memory that concatenate
            # to the logical sequence.
            tail_len = self._cache_len - self._write_pos
            return [
                [
                    (self._k[i][:, :, -tail_len:, :], self._v[i][:, :, -tail_len:, :]),
                    (self._k[i][:, :, :self._write_pos, :], self._v[i][:, :, :self._write_pos, :]),
                ]
                for i in range(self.n_layers)
            ]

    def get_cache_contiguous(self):
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

    def set_cache(self, cache):
        """Replace the cache contents (used for state restore/tests).

        ``cache`` must be a list of length ``n_layers``.  Each element is either
        a single ``(k, v)`` tuple or a list of blocks ``[(k, v), ...]`` as
        returned by :meth:`get_cache`.
        """
        if not cache:
            self.reset_cache()
            return

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
            trimmed = [[] for _ in normalized]
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
                self._k[i][:, :, self._write_pos:self._write_pos + L, :] = k_block
                self._v[i][:, :, self._write_pos:self._write_pos + L, :] = v_block
            self._write_pos += L
        self._cache_len = total_len

    @property
    def cache_len(self):
        return self._cache_len

    def truncate(self, new_cache_len):
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


def build_sliding_window_mask(seq_len, window_size, device):
    """Build a causal + sliding window mask for training or inference."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start : i + 1] = 0.0
    return mask
