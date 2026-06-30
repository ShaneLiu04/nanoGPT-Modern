"""
Prepare OpenWebText dataset for nanoGPT-style training.

Downloads/uses HuggingFace datasets, tokenizes with tiktoken, and saves binary
shards.  Two modes are supported:

* **map mode (default)** : loads the source dataset into local memory/cache and
  applies ``datasets.map(..., batched=True, num_proc=...)`` for fast,
  multi-process tokenization.  This is suitable for the full ~1.13B token
  OpenWebText split.
* **streaming mode** (--streaming) : never materializes all documents in RAM;
  tokenizes in batched chunks and writes fixed-size shards as tokens are
  produced.  Useful for quick tests or disk-constrained environments.

Both modes:

  * auto-select ``uint16``/``uint32`` based on the vocabulary size;
  * record dtype, vocab size, and document-boundary metadata in the index file;
  * produce a canonical ``{split}.bin`` plus per-shard files for resumable
    loading.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/openwebtext")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Max docs to process (default: 1M for train, 10K for val)",
    )
    parser.add_argument(
        "--shard_size", type=int, default=100_000_000, help="Target tokens per shard"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of documents per tokenization batch",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for datasets.map (default: os.cpu_count())",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode instead of cached map mode",
    )
    parser.add_argument(
        "--eot_token", type=int, default=50256, help="GPT-2 EOT token id"
    )

    # Data-quality pipeline options
    parser.add_argument(
        "--min_doc_chars",
        type=int,
        default=None,
        help="Drop documents shorter than this",
    )
    parser.add_argument(
        "--max_doc_chars",
        type=int,
        default=None,
        help="Drop documents longer than this",
    )
    parser.add_argument(
        "--max_repetition_ratio",
        type=float,
        default=None,
        help="Drop documents whose most common char n-gram exceeds this ratio",
    )
    parser.add_argument(
        "--repetition_n",
        type=int,
        default=10,
        help="n-gram size used by the repetition filter",
    )
    parser.add_argument(
        "--require_regex",
        type=str,
        default=None,
        help="Keep only documents matching this regex",
    )
    parser.add_argument(
        "--reject_regex",
        type=str,
        default=None,
        help="Drop documents matching this regex",
    )
    parser.add_argument(
        "--fasttext_model",
        type=str,
        default=None,
        help="Path to a fasttext .bin model for quality/language filtering",
    )
    parser.add_argument(
        "--fasttext_threshold",
        type=float,
        default=0.5,
        help="Minimum fasttext score to keep a document",
    )
    parser.add_argument(
        "--fasttext_label",
        type=str,
        default=None,
        help="Required fasttext top-1 label (e.g. __label__en)",
    )
    parser.add_argument(
        "--dedup_threshold",
        type=float,
        default=0.0,
        help="MinHash Jaccard threshold for near-duplicate removal (0=disabled)",
    )
    parser.add_argument(
        "--dedup_ngram", type=int, default=5, help="Character n-gram size for MinHash"
    )
    parser.add_argument(
        "--dedup_num_hashes", type=int, default=128, help="MinHash signature length"
    )
    parser.add_argument(
        "--dedup_num_bands", type=int, default=8, help="Number of LSH bands"
    )
    parser.add_argument(
        "--dedup_rows_per_band",
        type=int,
        default=16,
        help="Number of rows per LSH band",
    )
    parser.add_argument(
        "--mixture_config",
        type=str,
        default=None,
        help="JSON config for mixing multiple source datasets",
    )
    return parser.parse_args()


def _build_quality_filter(args):
    """Build the document-level quality filter from CLI args."""
    from data.filter import (
        CompositeFilter,
        LengthFilter,
        RegexFilter,
        RepetitionFilter,
        QualityFilter,
    )

    try:
        from data.filter import FastTextQualityFilter
    except Exception:  # pragma: no cover
        FastTextQualityFilter = None

    filters: list[QualityFilter] = []
    if args.min_doc_chars is not None or args.max_doc_chars is not None:
        filters.append(LengthFilter(args.min_doc_chars, args.max_doc_chars))
    if args.max_repetition_ratio is not None:
        filters.append(
            RepetitionFilter(
                n=args.repetition_n,
                max_repetition_ratio=args.max_repetition_ratio,
            )
        )
    if args.require_regex is not None or args.reject_regex is not None:
        filters.append(RegexFilter(args.require_regex, args.reject_regex))

    fasttext_model = getattr(args, "fasttext_model", None)
    if fasttext_model is not None:
        if FastTextQualityFilter is None:
            raise ImportError(
                "FastTextQualityFilter is not available; "
                "install fasttext to use --fasttext_model."
            )
        filters.append(
            FastTextQualityFilter(
                model_path=fasttext_model,
                threshold=getattr(args, "fasttext_threshold", 0.5),
                label=getattr(args, "fasttext_label", None),
            )
        )

    if not filters:
        return None
    return CompositeFilter(filters)


def _build_dedup(args, streaming: bool = False):
    """Build the deduplicator from CLI args."""
    if args.dedup_threshold <= 0.0:
        return None
    if streaming:
        from data.dedup import StreamingDuplicateDetector

        return StreamingDuplicateDetector(
            threshold=args.dedup_threshold,
            num_hashes=args.dedup_num_hashes,
            ngram_size=args.dedup_ngram,
            num_bands=args.dedup_num_bands,
            rows_per_band=args.dedup_rows_per_band,
        )
    from data.dedup import MinHashDeduplicator

    return MinHashDeduplicator(
        threshold=args.dedup_threshold,
        num_hashes=args.dedup_num_hashes,
        ngram_size=args.dedup_ngram,
        num_bands=args.dedup_num_bands,
        rows_per_band=args.dedup_rows_per_band,
    )


def _load_mixed_dataset(args, streaming: bool = False):
    """Load a single dataset or a mixture configured by ``args.mixture_config``."""
    from datasets import load_dataset  # type: ignore[import-untyped]
    from data.mixer import load_mixture_config, mix_datasets

    if args.mixture_config is None:
        name = "openwebtext"
        try:
            return load_dataset(name, split=args.split, streaming=streaming)
        except Exception:
            print("Falling back to Skylion007/openwebtext...")
            return load_dataset(
                "Skylion007/openwebtext",
                split=args.split,
                streaming=streaming,
                trust_remote_code=True,
            )

    cfg = load_mixture_config(args.mixture_config)
    sources = {}
    for key, spec in cfg["datasets"].items():
        ds_name = spec["name"]
        split_name = spec.get("split", args.split)
        trust = spec.get("trust_remote_code", False)
        try:
            sources[key] = load_dataset(
                ds_name,
                split=split_name,
                streaming=streaming,
                trust_remote_code=trust,
            )
        except Exception as exc:
            print(f"Failed to load {ds_name}: {exc}")
            raise
    return mix_datasets(
        sources,
        cfg["weights"],
        temperature=cfg.get("temperature", 1.0),
        seed=cfg.get("seed", 0),
    )


def _select_dtype(vocab_size):
    """Select uint16 for vocab <= 65535, otherwise uint32."""
    return np.uint16 if vocab_size <= 65535 else np.uint32


def _concatenate_shards_to_bin(
    shard_dir, output_dir, split, dtype, chunk_tokens=10_000_000
):
    """Stream all shard files into a single ``{split}.bin`` without loading it all into RAM."""
    import glob

    shard_files = sorted(glob.glob(os.path.join(shard_dir, f"{split}_*.bin")))
    if not shard_files:
        return
    main_path = os.path.join(output_dir, f"{split}.bin")
    print(f"Concatenating {len(shard_files)} shards into {main_path}...", flush=True)
    with open(main_path, "wb") as out_f:
        for shard_path in shard_files:
            arr = np.memmap(shard_path, dtype=dtype, mode="r")
            for start in range(0, len(arr), chunk_tokens):
                end = min(start + chunk_tokens, len(arr))
                out_f.write(arr[start:end].tobytes())
    print(f"Saved single-file binary: {main_path}", flush=True)


class _ShardWriter:
    """Accumulates token arrays and flushes fixed-size binary shards to disk."""

    def __init__(self, output_dir, split, dtype, shard_size, vocab_size, eot_token):
        self.output_dir = output_dir
        self.split = split
        self.dtype = dtype
        self.shard_size = shard_size
        self.vocab_size = vocab_size
        self.eot_token = eot_token
        self.shard_dir = os.path.join(output_dir, f"{split}_shards")
        os.makedirs(self.shard_dir, exist_ok=True)
        # Remove stale shards from previous runs so they cannot be concatenated
        # into the new {split}.bin.
        for stale in Path(self.shard_dir).glob(f"{split}_*.bin"):
            stale.unlink()

        self._buffer = np.empty(shard_size, dtype=dtype)
        self._used = 0
        self._shard_idx = 0
        self._total_tokens = 0
        self._doc_boundaries = [0]

    def append(self, tokens):
        """Append a 1-D numpy array of tokens (must already include EOT)."""
        n = len(tokens)
        if n == 0:
            return
        offset = 0
        while offset < n:
            room = self.shard_size - self._used
            take = min(room, n - offset)
            self._buffer[self._used : self._used + take] = tokens[
                offset : offset + take
            ]
            self._used += take
            offset += take
            self._total_tokens += take

            if self._used >= self.shard_size:
                self._flush()

    def record_doc_boundary(self):
        """Record the global token offset where a new document starts."""
        self._doc_boundaries.append(self._total_tokens)

    def close(self):
        """Flush any remaining tokens and return metadata."""
        if self._used > 0:
            self._flush(partial=True)
        return {
            "total_tokens": self._total_tokens,
            "num_docs": len(self._doc_boundaries) - 1,
            "n_shards": self._shard_idx,
        }

    def _flush(self, partial=False):
        data = self._buffer[: self._used].copy()
        if partial:
            # Also write the tail as the canonical single-file bin for convenience.
            main_path = os.path.join(self.output_dir, f"{self.split}.bin")
            data.tofile(main_path)
            print(f"Saved {self._used:,} tokens to {main_path}")

        shard_path = os.path.join(
            self.shard_dir, f"{self.split}_{self._shard_idx:05d}.bin"
        )
        data.tofile(shard_path)
        print(f"  shard {self._shard_idx:05d}: {self._used:,} tokens -> {shard_path}")

        self._shard_idx += 1
        self._used = 0

    def write_index(self):
        idx_path = os.path.join(self.output_dir, f"{self.split}.idx")
        with open(idx_path, "w") as f:
            f.write(f"dtype={self.dtype.__name__}\n")
            f.write(f"vocab_size={self.vocab_size}\n")
            f.write(f"total_tokens={self._total_tokens}\n")
            f.write(f"num_docs={len(self._doc_boundaries) - 1}\n")
            f.write(f"shard_size={self.shard_size}\n")
            f.write(f"eot_token={self.eot_token}\n")
            # Write boundaries sparsely to keep the index small.
            boundary_stride = max(1, len(self._doc_boundaries) // 10_000)
            for i in range(0, len(self._doc_boundaries), boundary_stride):
                f.write(f"doc_boundary={self._doc_boundaries[i]}\n")
        print(f"Index saved to {idx_path}")


def _tokenize_batch(batch, tokenizer, eot_token):
    """Batched tokenization helper used with ``datasets.map``."""
    texts = batch["text"]
    encoded = tokenizer.encode_ordinary_batch(texts)
    return {
        "input_ids": [toks + [eot_token] for toks in encoded],
        "len": [len(toks) + 1 for toks in encoded],
    }


def _load_streaming_dataset(split):
    from datasets import load_dataset

    print(f"Loading OpenWebText ({split}) via streaming...")
    try:
        return load_dataset(
            "openwebtext", split=split, streaming=True, trust_remote_code=True
        )
    except Exception:
        print(
            "Failed to load openwebtext via streaming; trying Skylion007/openwebtext subset..."
        )
        return load_dataset(
            "Skylion007/openwebtext",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )


def _prepare_map_mode(args, tokenizer, dtype):
    """Fast path: cache the split locally and use datasets.map with num_proc."""
    quality_filter = _build_quality_filter(args)
    dedup = _build_dedup(args, streaming=False)

    print(
        f"Loading source dataset ({args.split}) into local cache for map-mode tokenization..."
    )
    ds = _load_mixed_dataset(args, streaming=False)

    default_max = 1_000_000 if args.split == "train" else 10_000
    max_docs = args.max_docs if args.max_docs is not None else default_max
    if max_docs and hasattr(ds, "__len__") and len(ds) > max_docs:  # type: ignore[arg-type]
        ds = ds.select(range(max_docs))

    if quality_filter is not None:
        print(f"Applying quality filter: {quality_filter.describe()}")
        ds = ds.filter(
            lambda example: quality_filter(example.get("text", "")),
            batched=False,
            desc="Filtering",
        )

    if dedup is not None:
        print(f"Applying MinHash deduplication (threshold={args.dedup_threshold})...")
        texts = ds["text"]
        keep_mask = dedup.fit(texts).duplicate_mask()
        keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
        ds = ds.select(keep_indices)
        print(f"  kept {len(keep_indices):,} / {len(texts):,} documents")

    num_proc = args.num_proc or os.cpu_count()
    print(f"Tokenizing with batch_size={args.batch_size}, num_proc={num_proc}...")
    ds = ds.map(
        lambda batch: _tokenize_batch(batch, tokenizer, args.eot_token),
        batched=True,
        batch_size=args.batch_size,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    writer = _ShardWriter(
        output_dir=args.output_dir,
        split=args.split,
        dtype=dtype,
        shard_size=args.shard_size,
        vocab_size=tokenizer.n_vocab,
        eot_token=args.eot_token,
    )

    # Stream tokens out of the mapped dataset without materializing the full
    # token array in memory.
    buffer = np.empty(args.shard_size, dtype=dtype)
    used = 0
    doc_count = 0
    for row in ds:
        tokens = row["input_ids"]
        n = len(tokens)
        offset = 0
        while offset < n:
            room = args.shard_size - used
            take = min(room, n - offset)
            buffer[used : used + take] = tokens[offset : offset + take]
            used += take
            offset += take
            if used >= args.shard_size:
                writer.append(buffer)
                used = 0
        writer.record_doc_boundary()
        doc_count += 1
        if doc_count % 10_000 == 0:
            print(
                f"  written {doc_count:,} docs, {writer._total_tokens:,} tokens...",
                flush=True,
            )

    if used > 0:
        writer.append(buffer[:used])

    meta = writer.close()
    return meta, writer


def _prepare_streaming_mode(args, tokenizer, dtype):
    """Memory-cheap path: tokenize streaming batches and write shards on the fly."""
    quality_filter = _build_quality_filter(args)
    dedup = _build_dedup(args, streaming=True)

    ds = _load_mixed_dataset(args, streaming=True)

    default_max = 1_000_000 if args.split == "train" else 10_000
    max_docs = args.max_docs if args.max_docs is not None else default_max
    print(
        f"Target: up to {max_docs:,} documents, shards of ~{args.shard_size:,} tokens"
    )

    writer = _ShardWriter(
        output_dir=args.output_dir,
        split=args.split,
        dtype=dtype,
        shard_size=args.shard_size,
        vocab_size=tokenizer.n_vocab,
        eot_token=args.eot_token,
    )

    batch = []
    doc_count = 0

    def _flush_batch():
        nonlocal batch
        if not batch:
            return
        filtered = batch
        if quality_filter is not None:
            filtered = [t for t in filtered if quality_filter(t)]
        if dedup is not None:
            filtered = [t for t in filtered if not dedup.is_duplicate(t)]
        if not filtered:
            batch = []
            return
        encoded = tokenizer.encode_ordinary_batch(filtered)
        for toks in encoded:
            toks.append(args.eot_token)
            writer.append(np.array(toks, dtype=dtype))
            writer.record_doc_boundary()
        batch = []

    for doc in ds:
        if doc_count >= max_docs:
            break
        batch.append(doc.get("text", ""))
        doc_count += 1
        if len(batch) >= args.batch_size:
            _flush_batch()
        if doc_count % 10_000 == 0:
            print(
                f"  processed {doc_count:,} docs, {writer._total_tokens:,} tokens...",
                flush=True,
            )

    _flush_batch()
    meta = writer.close()
    return meta, writer


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import tiktoken
        from datasets import load_dataset
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please run: pip install tiktoken datasets")
        sys.exit(1)

    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    dtype = _select_dtype(vocab_size)
    print(f"Vocab size: {vocab_size}, using dtype: {dtype.__name__}")

    if args.streaming:
        meta, writer = _prepare_streaming_mode(args, tokenizer, dtype)
    else:
        meta, writer = _prepare_map_mode(args, tokenizer, dtype)

    # Reconstruct the canonical single-file binary from shards so existing loaders
    # can keep using ``{split}.bin`` without knowing about sharding.
    if meta["n_shards"] > 0:
        _concatenate_shards_to_bin(writer.shard_dir, args.output_dir, args.split, dtype)

    writer.write_index()

    print(
        f"\nDone. Docs: {meta['num_docs']:,}, tokens: {meta['total_tokens']:,}, shards: {meta['n_shards']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
