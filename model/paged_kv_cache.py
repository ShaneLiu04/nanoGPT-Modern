"""Block-table-based KV cache manager (PagedAttention-style layout).

This module provides an alternative to ``KVCacheManager`` that organizes keys
and values in fixed-size blocks.  It is API-compatible with
``KVCacheManager``: callers still receive contiguous K/V tensors per layer via
``get_cache_contiguous()``.

The block-table layout is the foundation for future PagedAttention kernel
integration (continuous batching, prefix caching).  The current Python
implementation keeps K/V in contiguous per-sequence buffers internally and uses
block tables to track which logical positions are allocated, making it a safe,
bit-exact drop-in replacement for the ring-buffer manager.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class PagedKVCacheManager:
    """Block-table KV cache manager.

    Each sequence owns a contiguous buffer of shape
    ``[max_blocks_per_seq, n_kv_heads, block_size, head_dim]``.  A block table
    records which blocks are currently populated.  This makes the manager a
    stepping stone toward true PagedAttention kernels while remaining compatible
    with the existing ``past_kvs`` API.

    Parameters
    ----------
    n_layers : int
    n_heads : int
        Informational only.
    n_kv_heads : int
    head_dim : int
    max_cache_len : int
        Maximum logical tokens per sequence.  Rounded up to a multiple of
        ``block_size``.
    block_size : int
        Number of tokens per block.

    Attributes
    ----------
    start_pos : int
        Absolute position of the first token currently stored.  Kept for API
        compatibility with ``KVCacheManager``.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_cache_len: int = 1024,
        block_size: int = 16,
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_cache_len = (
            (max_cache_len + block_size - 1) // block_size
        ) * block_size
        self.max_blocks_per_seq = self.max_cache_len // block_size

        self._k: List[torch.Tensor] = []
        self._v: List[torch.Tensor] = []
        self._batch_size: Optional[int] = None
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

        # _block_tables[b] = list of populated block indices for sequence b.
        self._block_tables: List[List[int]] = []
        # Logical length per sequence.
        self._seq_len: List[int] = []
        self.start_pos = 0

    @classmethod
    def from_config(cls, config, block_size: int = 16):
        """Factory: build a PagedKVCacheManager from a ModernGPTConfig."""
        head_dim = config.n_embd // config.n_head
        max_len = getattr(config, "effective_block_size", config.block_size)
        return cls(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            n_kv_heads=config.n_kv_head,
            head_dim=head_dim,
            max_cache_len=max_len,
            block_size=block_size,
        )

    def init_cache(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Allocate per-layer per-sequence block buffers and reset state."""
        self._batch_size = batch_size
        self._device = device
        self._dtype = dtype
        self._k = [
            torch.zeros(
                batch_size,
                self.max_blocks_per_seq,
                self.n_kv_heads,
                self.block_size,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.n_layers)
        ]
        self._v = [
            torch.zeros(
                batch_size,
                self.max_blocks_per_seq,
                self.n_kv_heads,
                self.block_size,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.n_layers)
        ]
        self._block_tables = [[] for _ in range(batch_size)]
        self._seq_len = [0] * batch_size
        self.start_pos = 0

    def reset_cache(self) -> None:
        """Clear all cached keys/values and reset block tables."""
        for tensor in self._k:
            tensor.zero_()
        for tensor in self._v:
            tensor.zero_()
        self._block_tables = (
            [[] for _ in range(self._batch_size)] if self._batch_size else []
        )
        self._seq_len = [0] * self._batch_size if self._batch_size else []
        self.start_pos = 0

    def _allocate_block(self, batch_idx: int) -> int:
        """Return the next free block index for ``batch_idx``."""
        used = set(self._block_tables[batch_idx])
        for i in range(self.max_blocks_per_seq):
            if i not in used:
                return i
        raise RuntimeError(f"Paged KV cache out of blocks for sequence {batch_idx}")

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """Write new K/V at the current logical end of each sequence.

        ``new_k`` and ``new_v`` must have shape ``[B, n_kv_heads, S, head_dim]``.
        The logical length is **not** changed here; call :meth:`advance` after
        all layers have been updated.
        """
        if self._k is None:
            raise RuntimeError("Cache not initialized. Call init_cache first.")

        B, _, S, _ = new_k.shape
        for b in range(B):
            write_pos = self._seq_len[b]
            pos_in_block = write_pos % self.block_size
            remaining = S
            write_offset = 0

            while remaining > 0:
                if pos_in_block == 0:
                    # Start a new block at this sequence's block boundary.
                    self._block_tables[b].append(self._allocate_block(b))
                elif not self._block_tables[b]:
                    self._block_tables[b].append(self._allocate_block(b))

                block_idx = self._block_tables[b][-1]
                room = self.block_size - pos_in_block
                chunk = min(room, remaining)

                self._k[layer_idx][
                    b, block_idx, :, pos_in_block : pos_in_block + chunk, :
                ] = new_k[b, :, write_offset : write_offset + chunk, :]
                self._v[layer_idx][
                    b, block_idx, :, pos_in_block : pos_in_block + chunk, :
                ] = new_v[b, :, write_offset : write_offset + chunk, :]

                write_pos += chunk
                write_offset += chunk
                remaining -= chunk
                pos_in_block = write_pos % self.block_size

    def advance(self, delta: int) -> None:
        """Advance the logical cache length after all layers are written."""
        if delta <= 0:
            return
        batch_size = self._batch_size
        assert batch_size is not None
        for b in range(batch_size):
            self._seq_len[b] += delta

    def _get_layer_blocks(
        self, layer_idx: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return a list of (k_block, v_block) tuples covering all sequences.

        Each block is contiguous in sequence dimension for one or more sequences
        (all sequences are concatenated along the sequence dim).  This mirrors
        ``KVCacheManager.get_cache()``.
        """
        if not self._k or all(length == 0 for length in self._seq_len):
            return []

        batch_size = self._batch_size
        device = self._device
        dtype = self._dtype
        assert batch_size is not None and device is not None and dtype is not None

        # Materialize one contiguous tensor per layer [B, n_kv_heads, T_max, head_dim]
        # where T_max is the maximum logical length across the batch.
        max_len = max(self._seq_len)
        k_out = torch.zeros(
            batch_size,
            self.n_kv_heads,
            max_len,
            self.head_dim,
            device=device,
            dtype=dtype,
        )
        v_out = torch.zeros(
            batch_size,
            self.n_kv_heads,
            max_len,
            self.head_dim,
            device=device,
            dtype=dtype,
        )

        k_pool = self._k[layer_idx]
        v_pool = self._v[layer_idx]
        for b in range(batch_size):
            seq_len = self._seq_len[b]
            if seq_len == 0:
                continue
            written = 0
            for block_idx in self._block_tables[b]:
                block_len = min(self.block_size, seq_len - written)
                if block_len <= 0:
                    break
                k_out[b, :, written : written + block_len, :] = k_pool[
                    b, block_idx, :, :block_len, :
                ]
                v_out[b, :, written : written + block_len, :] = v_pool[
                    b, block_idx, :, :block_len, :
                ]
                written += block_len

        return [(k_out, v_out)]

    def get_cache(self):
        """Return cached K/V as a list of blocks per layer."""
        if not self._k or all(length == 0 for length in self._seq_len):
            return []
        layer_blocks = self._get_layer_blocks(0)
        return [layer_blocks for _ in range(self.n_layers)]

    def get_cache_contiguous(self):
        """Return cached K/V as single contiguous tensors per layer."""
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

    def set_cache(self, cache) -> None:
        """Replace the cache contents (used for state restore/tests).

        ``cache`` must be a list of length ``n_layers``.  Each element is either
        a single ``(k, v)`` tuple or a list of blocks ``[(k, v), ...]``.
        """
        if not cache:
            self.reset_cache()
            return

        normalized = []
        for layer_cache in cache:
            if isinstance(layer_cache, tuple):
                normalized.append([layer_cache])
            else:
                normalized.append(list(layer_cache))

        self.reset_cache()
        for layer_idx in range(self.n_layers):
            k_cat = torch.cat([b[0] for b in normalized[layer_idx]], dim=2)
            v_cat = torch.cat([b[1] for b in normalized[layer_idx]], dim=2)
            self.update(layer_idx, k_cat, v_cat)

    @property
    def cache_len(self) -> int:
        return max(self._seq_len) if self._seq_len else 0
