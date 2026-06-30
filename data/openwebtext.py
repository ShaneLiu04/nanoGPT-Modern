"""OpenWebText streaming dataloader with memory-efficient loading and dynamic batching.
Expects ~1M docs, ~1.13B tokens total.
"""

import os
import random
import warnings
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


def get_openwebtext_dataset(
    data_dir="data/openwebtext",
    split="train",
    block_size=1024,
    resume_offset=0,
    use_packing=False,
    shuffle_buffer=None,
    shuffle_idx_path=None,
):
    bin_path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"OpenWebText binary not found at {bin_path}. "
            "Please run data preparation (prepare.py) to tokenize and shard the dataset."
        )
    if use_packing:
        return PackingDataset(
            bin_path, block_size=block_size, resume_offset=resume_offset
        )
    return MemmapDataset(
        bin_path,
        block_size=block_size,
        resume_offset=resume_offset,
        shuffle_buffer=shuffle_buffer,
        shuffle_idx_path=shuffle_idx_path,
    )


def _detect_dtype_from_index(bin_path):
    """Read .idx metadata to determine the numpy dtype used for the binary."""
    idx_path = bin_path.replace(".bin", ".idx")
    dtype = np.uint16  # default for backward compat
    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            for line in f:
                if line.startswith("dtype="):
                    name = line.strip().split("=")[1]
                    if name == "uint32":
                        dtype = np.uint32
                    break
    return dtype


def _detect_eot_from_index(bin_path):
    """Read the EOT token id recorded in the index file."""
    idx_path = bin_path.replace(".bin", ".idx")
    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            for line in f:
                if line.startswith("eot_token="):
                    return int(line.strip().split("=")[1])
    return 50256  # GPT-2 default


def _detect_block_size_from_index(bin_path):
    """Read the block_size recorded in the index file."""
    idx_path = bin_path.replace(".bin", ".idx")
    block_size = 1024
    if os.path.exists(idx_path):
        with open(idx_path, "r") as f:
            for line in f:
                if line.startswith("block_size="):
                    block_size = int(line.strip().split("=")[1])
                    break
    return block_size


def generate_shuffle_index(bin_path, seed=1337, block_size=None):
    """Generate a global random chunk index file for *bin_path*.

    The output file is named ``<bin_path>.shuffle.idx`` and contains a
    NumPy array of shuffled chunk indices.  When *block_size* is not
    provided, the function attempts to read it from the companion ``.idx``
    metadata file.
    """
    if block_size is None:
        block_size = _detect_block_size_from_index(bin_path)
    dtype = _detect_dtype_from_index(bin_path)
    data = np.memmap(bin_path, dtype=dtype, mode="r")
    length = len(data) // (block_size + 1)
    rng = np.random.default_rng(seed)
    shuffle_idx = rng.permutation(length)
    idx_path = bin_path + ".shuffle.idx"
    with open(idx_path, "wb") as f:
        np.save(f, shuffle_idx)


class MemmapDataset(IterableDataset):
    def __init__(
        self,
        bin_path,
        block_size=1024,
        resume_offset=0,
        shuffle_buffer=None,
        shuffle_idx_path=None,
    ):
        super().__init__()
        self.block_size = block_size
        self.resume_offset = resume_offset
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_idx_path = shuffle_idx_path
        dtype = _detect_dtype_from_index(bin_path)
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self.length = len(self.data) // (block_size + 1)
        self._shuffle_idx = None
        if shuffle_idx_path is not None:
            if os.path.exists(shuffle_idx_path):
                with open(shuffle_idx_path, "rb") as f:
                    self._shuffle_idx = np.load(f)
                if len(self._shuffle_idx) != self.length:
                    warnings.warn(
                        f"Shuffle index length mismatch: {len(self._shuffle_idx)} != {self.length}"
                    )
                    self._shuffle_idx = None
            else:
                warnings.warn(f"Shuffle index not found: {shuffle_idx_path}")

    def state_dict(self):
        """Return a serializable state for resumable training."""
        return {"resume_offset": self.resume_offset}

    def load_state_dict(self, state):
        """Restore iteration offset from a checkpoint state."""
        self.resume_offset = int(state.get("resume_offset", 0))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start = self.resume_offset
            end = self.length
        else:
            per_worker = (self.length - self.resume_offset) // worker_info.num_workers
            start = self.resume_offset + worker_info.id * per_worker
            end = (
                start + per_worker
                if worker_info.id < worker_info.num_workers - 1
                else self.length
            )

        if self._shuffle_idx is not None:
            # Global shuffle: use pre-computed permutation for this worker slice.
            indices = self._shuffle_idx[start:end].tolist()
        else:
            indices = list(range(start, end))

        # Apply chunk-level shuffle only when no global shuffle index is available.
        if self._shuffle_idx is None:
            buffer_size = (
                self.shuffle_buffer
                if self.shuffle_buffer is not None
                else min(10000, end - start)
            )
            buffer_size = max(1, buffer_size)
            for buf_start in range(0, len(indices), buffer_size):
                buf_end = min(buf_start + buffer_size, len(indices))
                buf = indices[buf_start:buf_end]
                random.shuffle(buf)
                indices[buf_start:buf_end] = buf

        for i in indices:
            if i < 0 or i >= self.length:
                continue
            chunk = self.data[
                i * (self.block_size + 1) : (i + 1) * (self.block_size + 1)
            ]
            chunk_i64 = chunk.astype(np.int64)
            x = torch.from_numpy(chunk_i64[:-1])
            y = torch.from_numpy(chunk_i64[1:])
            yield x, y

    def __len__(self):
        return self.length


class DocBoundaryDataset(IterableDataset):
    """Like MemmapDataset but respects document boundaries.

    Documents are separated by EOT tokens (token id 50256 for GPT-2).
    When a chunk would span a document boundary, it is truncated to the
    boundary and the remainder starts a new chunk.  This prevents the
    model from attending across unrelated documents.

    .. note::
        ``resume_offset`` is interpreted as the number of already-yielded
        examples in the *current* worker partition.  For exact global resume
        semantics use a single worker (``num_workers=0``) or divide the global
        offset among workers externally.
    """

    def __init__(self, bin_path, block_size=1024, eot_token=50256, resume_offset=0):
        super().__init__()
        self.block_size = block_size
        self.eot_token = eot_token
        self.resume_offset = resume_offset
        dtype = _detect_dtype_from_index(bin_path)
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        # pre-compute EOT positions (sampling every 1000th position for speed)
        self._eot_positions = None  # lazy

    def state_dict(self):
        return {"resume_offset": self.resume_offset}

    def load_state_dict(self, state):
        self.resume_offset = int(state.get("resume_offset", 0))

    def _build_eot_index(self):
        """Find all EOT token positions (cached)."""
        if self._eot_positions is not None:
            return
        # scan every 256th position first for large datasets
        positions = []
        chunk = 256
        for i in range(0, len(self.data), chunk):
            end = min(i + chunk, len(self.data))
            segment = self.data[i:end]
            eot_indices = np.where(segment == self.eot_token)[0]
            for idx in eot_indices:
                positions.append(i + idx)
        self._eot_positions = np.array(positions, dtype=np.int64)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, len(self.data)
        else:
            per_worker = len(self.data) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = (
                start + per_worker
                if worker_info.id < worker_info.num_workers - 1
                else len(self.data)
            )

        self._build_eot_index()

        # Walk through tokens, producing chunks that stay within documents
        pos = start
        buffer = min(10000, max(1, (end - start) // self.block_size))
        segments = []
        yielded = 0

        while pos + self.block_size + 1 <= end:
            chunk = self.data[pos : pos + self.block_size + 1]
            eots_in_chunk = np.where(chunk[:-1] == self.eot_token)[0]
            if len(eots_in_chunk) > 0:
                # Truncate to first EOT to keep doc boundary clean
                split = eots_in_chunk[0] + 1
                if split >= 2:
                    chunk_i64 = chunk[:split].astype(np.int64)
                    x = torch.from_numpy(chunk_i64[:-1])
                    y = torch.from_numpy(chunk_i64[1:])
                    segments.append((x, y))
                pos += split
            else:
                chunk_i64 = chunk.astype(np.int64)
                x = torch.from_numpy(chunk_i64[:-1])
                y = torch.from_numpy(chunk_i64[1:])
                segments.append((x, y))
                pos += self.block_size

            if len(segments) >= buffer:
                random.shuffle(segments)
                for s in segments:
                    yielded += 1
                    if yielded > self.resume_offset:
                        yield s
                segments = []

        random.shuffle(segments)
        for s in segments:
            yielded += 1
            if yielded > self.resume_offset:
                yield s

    def __len__(self):
        return len(self.data) // self.block_size


class PackingDataset(IterableDataset):
    """Pack multiple short documents into block_size sequences.

    Each returned sample is a 3-tuple ``(x, y, document_ids)`` where
    ``document_ids`` marks the document boundary of every token.  The model
    can use this to prevent attention across documents, allowing efficient
    packing without introducing cross-document noise.

    The target ``y`` is set to ``-1`` for positions that predict the first
    token of a new document (or padding), so the loss is not computed on
    positions with no in-document context.
    """

    def __init__(
        self, bin_path, block_size=1024, eot_token=None, resume_offset=0, shuffle=True
    ):
        super().__init__()
        self.block_size = block_size
        self.eot_token = (
            eot_token if eot_token is not None else _detect_eot_from_index(bin_path)
        )
        self.resume_offset = resume_offset
        self.shuffle = shuffle
        dtype = _detect_dtype_from_index(bin_path)
        self.data = np.memmap(bin_path, dtype=dtype, mode="r")
        self._samples = None  # lazy build

    def _build_samples(self):
        """Greedily pack documents into fixed-length sequences."""
        if self._samples is not None:
            return

        data = self.data
        eot = self.eot_token
        block_size = self.block_size

        # Find document boundaries: start of each document.
        # Document 0 starts at 0; each EOT marks the end of a document, and
        # the next token starts a new document.
        doc_starts = [0]
        # Scan in chunks to handle large memmaps efficiently.
        scan_chunk = 1_000_000
        for start in range(0, len(data), scan_chunk):
            end = min(start + scan_chunk, len(data))
            segment = np.array(data[start:end])
            eot_positions = np.where(segment == eot)[0]
            for pos in eot_positions:
                nxt = start + pos + 1
                if nxt < len(data):
                    doc_starts.append(nxt)
        doc_starts = sorted(set(doc_starts))

        # Build documents as (start, end) intervals.
        docs = []
        for i in range(len(doc_starts)):
            s = doc_starts[i]
            e = doc_starts[i + 1] if i + 1 < len(doc_starts) else len(data)
            if e > s:
                docs.append((s, e))

        # Greedily pack documents into sequences of length block_size.
        samples = []
        seq_tokens = []
        seq_doc_ids = []
        cur_doc_id = 0

        def flush_seq():
            nonlocal cur_doc_id
            if not seq_tokens:
                return
            L = len(seq_tokens)
            pad = block_size - L
            x = seq_tokens + [eot] * pad
            y = x[1:] + [eot]
            doc_ids = seq_doc_ids + [-1] * pad

            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)
            doc_ids = torch.tensor(doc_ids, dtype=torch.long)

            # Mask out targets that predict the first token of a new document
            # or padding positions.
            for j in range(L):
                if seq_tokens[j] == eot or j >= L - 1:
                    y[j] = -1
            for j in range(L, block_size):
                y[j] = -1

            samples.append((x, y, doc_ids))
            seq_tokens.clear()
            seq_doc_ids.clear()
            cur_doc_id += 1

        doc_idx = 0
        for s, e in docs:
            length = e - s
            if length > block_size:
                # Long document: split into contiguous block_size chunks.
                # All chunks of the same document share doc_idx so that the
                # cross-document mask does not block attention between them.
                pos = s
                while pos < e:
                    end_pos = min(pos + block_size, e)
                    chunk = data[pos:end_pos].tolist()
                    chunk_doc_ids = [doc_idx] * len(chunk)
                    if len(chunk) < block_size:
                        # Tail of a long doc: pad and flush.
                        seq_tokens.extend(chunk)
                        seq_doc_ids.extend(chunk_doc_ids)
                        flush_seq()
                    else:
                        # Full block: the target at the last position is the
                        # next token in the *same* document, if available.
                        next_pos = end_pos
                        if next_pos < e:
                            next_token = int(data[next_pos])
                        else:
                            next_token = eot
                        x = chunk
                        y = chunk[1:] + [next_token]
                        # Mask EOT at the end of a document because it would
                        # predict the first token of the next document.
                        if next_token == eot:
                            y[-1] = -1
                        doc_ids = chunk_doc_ids
                        x = torch.tensor(x, dtype=torch.long)
                        y = torch.tensor(y, dtype=torch.long)
                        doc_ids = torch.tensor(doc_ids, dtype=torch.long)
                        samples.append((x, y, doc_ids))
                    pos += block_size
            else:
                if len(seq_tokens) + length > block_size:
                    flush_seq()
                seq_tokens.extend(data[s:e].tolist())
                seq_doc_ids.extend([doc_idx] * length)
            doc_idx += 1

        if seq_tokens:
            flush_seq()

        self._samples = samples

    def state_dict(self):
        return {"resume_offset": self.resume_offset}

    def load_state_dict(self, state):
        self.resume_offset = int(state.get("resume_offset", 0))

    def __iter__(self):
        self._build_samples()
        samples = list(self._samples)
        if self.shuffle:
            random.shuffle(samples)
        for i, sample in enumerate(samples):
            if i < self.resume_offset:
                continue
            yield sample

    def __len__(self):
        self._build_samples()
        return len(self._samples)


class GlobalShuffledDataset(IterableDataset):
    """Compatibility wrapper that always applies global shuffle.

    On first iteration, a ``.shuffle.idx`` file is generated if it does not
    already exist, and the underlying ``MemmapDataset`` reads from it.
    """

    def __init__(
        self, bin_path, block_size=1024, resume_offset=0, shuffle_buffer=None, seed=1337
    ):
        super().__init__()
        self.bin_path = bin_path
        self.block_size = block_size
        self.resume_offset = resume_offset
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        idx_path = bin_path + ".shuffle.idx"
        if not os.path.exists(idx_path):
            generate_shuffle_index(bin_path, seed=seed, block_size=block_size)
        self._dataset = MemmapDataset(
            bin_path,
            block_size=block_size,
            resume_offset=resume_offset,
            shuffle_buffer=shuffle_buffer,
            shuffle_idx_path=idx_path,
        )

    def __iter__(self):
        return iter(self._dataset)

    def __len__(self):
        return len(self._dataset)

    def state_dict(self):
        return self._dataset.state_dict()

    def load_state_dict(self, state):
        self._dataset.load_state_dict(state)


def get_dataloader(
    data_dir="data/openwebtext",
    split="train",
    batch_size=12,
    block_size=1024,
    num_workers=4,
    resume_offset=0,
    worker_init_fn=None,
    use_packing=False,
    shuffle_buffer=None,
    shuffle_idx_path=None,
):
    dataset = get_openwebtext_dataset(
        data_dir,
        split,
        block_size=block_size,
        resume_offset=resume_offset,
        use_packing=use_packing,
        shuffle_buffer=shuffle_buffer,
        shuffle_idx_path=shuffle_idx_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    return loader
