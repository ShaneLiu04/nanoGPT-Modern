"""Document-level near-duplicate detection with MinHash + LSH.

The implementation is intentionally dependency-free so it works on Windows and
in minimal environments.  For very large corpora consider swapping in
``datasketch``.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _char_ngrams(text: str, n: int) -> Set[str]:
    """Return the set of character n-grams."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _stable_hash(value: str, seed: int) -> int:
    """Return a deterministic 32-bit integer hash."""
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


class MinHash:
    """Compute a MinHash signature for a string.

    Parameters
    ----------
    num_hashes:
        Length of the signature.  More hashes give better recall at the cost
        of CPU/memory.
    ngram_size:
        Character n-gram size used to represent the document.
    seed:
        Base seed for the hash functions.
    """

    def __init__(self, num_hashes: int = 128, ngram_size: int = 5, seed: int = 0):
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")
        if ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self._seeds = [seed + i for i in range(num_hashes)]

    def ngrams(self, text: str) -> Set[str]:
        return _char_ngrams(text, self.ngram_size)

    def signature(self, text: str) -> Tuple[int, ...]:
        """Return the MinHash signature as a tuple of ints."""
        ngrams = self.ngrams(text)
        if not ngrams:
            return tuple([0] * self.num_hashes)
        sig: List[int] = []
        for s in self._seeds:
            min_val = min(_stable_hash(ng, s) for ng in ngrams)
            sig.append(min_val)
        return tuple(sig)


class LocalitySensitiveHashing:
    """Band-based LSH for MinHash signatures.

    A pair of documents is a *candidate* duplicate if any band of their
    signatures collides in the same bucket.
    """

    def __init__(self, num_bands: int = 8, rows_per_band: int = 16):
        if num_bands <= 0 or rows_per_band <= 0:
            raise ValueError("num_bands and rows_per_band must be positive")
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self._buckets: List[Dict[int, Set[int]]] = [
            defaultdict(set) for _ in range(num_bands)
        ]

    @property
    def num_hashes(self) -> int:
        return self.num_bands * self.rows_per_band

    def fit(self, signatures: Sequence[Tuple[int, ...]]) -> "LocalitySensitiveHashing":
        """Index a batch of signatures and return self."""
        for idx, sig in enumerate(signatures):
            self.add(sig, idx)
        return self

    def add(self, signature: Tuple[int, ...], idx: int) -> None:
        """Index a single signature."""
        if len(signature) != self.num_hashes:
            raise ValueError(
                f"Signature length {len(signature)} != {self.num_hashes} "
                "(num_bands*rows_per_band)"
            )
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band = signature[start : start + self.rows_per_band]
            bucket = hash(band)
            self._buckets[band_idx][bucket].add(idx)

    def query(self, signature: Tuple[int, ...]) -> Set[int]:
        """Return candidate indices for a signature."""
        candidates: Set[int] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            band = signature[start : start + self.rows_per_band]
            bucket = hash(band)
            candidates.update(self._buckets[band_idx].get(bucket, set()))
        return candidates


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class MinHashDeduplicator:
    """End-to-end MinHash + LSH deduplicator.

    The algorithm:

    1. Compute a MinHash signature for every document.
    2. Index the signatures with LSH bands.
    3. For each candidate pair, verify exact Jaccard similarity on n-grams.
    4. Mark later-occurring duplicates with similarity >= ``threshold``.

    Parameters
    ----------
    threshold:
        Jaccard similarity above which a document is considered a duplicate.
    num_hashes:
        Signature length.  Should equal ``num_bands * rows_per_band``.
    ngram_size:
        Character n-gram size.
    num_bands:
        Number of LSH bands.
    rows_per_band:
        Number of rows per LSH band.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        num_hashes: int = 128,
        ngram_size: int = 5,
        num_bands: int = 8,
        rows_per_band: int = 16,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if num_bands * rows_per_band != num_hashes:
            raise ValueError("num_hashes must equal num_bands * rows_per_band")
        self.threshold = threshold
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self._minhash = MinHash(num_hashes, ngram_size)
        self._lsh = LocalitySensitiveHashing(num_bands, rows_per_band)
        self._duplicates: Optional[Set[int]] = None
        self._size: Optional[int] = None

    def fit(self, texts: Sequence[str]) -> "MinHashDeduplicator":
        """Compute signatures, build LSH index, and find duplicates."""
        self._size = len(texts)
        signatures = [self._minhash.signature(t) for t in texts]
        self._lsh.fit(signatures)

        ngrams = [self._minhash.ngrams(t) for t in texts]
        duplicate: Set[int] = set()

        for i, sig in enumerate(signatures):
            if i in duplicate:
                continue
            candidates = self._lsh.query(sig)
            candidates.discard(i)
            for j in candidates:
                if j in duplicate or j <= i:
                    continue
                if _jaccard(ngrams[i], ngrams[j]) >= self.threshold:
                    duplicate.add(j)

        self._duplicates = duplicate
        return self

    def duplicate_mask(self, texts: Optional[Sequence[str]] = None) -> List[bool]:
        """Return a list where ``True`` means "keep" and ``False" means "drop".

        If ``texts`` is provided, re-fit on it; otherwise use the result from the
        last ``fit`` call.
        """
        if texts is not None:
            self.fit(texts)
        if self._duplicates is None:
            raise RuntimeError("fit() must be called before duplicate_mask()")
        return [i not in self._duplicates for i in range(self._expected_size(texts))]

    def _expected_size(self, texts: Optional[Sequence[str]]) -> int:
        if texts is not None:
            return len(texts)
        if self._size is None:
            raise RuntimeError("fit() must be called before duplicate_mask()")
        return self._size

    def duplicate_indices(self) -> Set[int]:
        """Return indices of documents marked as duplicates."""
        if self._duplicates is None:
            raise RuntimeError("fit() must be called first")
        return self._duplicates.copy()


class StreamingDuplicateDetector:
    """Single-pass approximate duplicate detector for streaming data.

    Maintains an LSH index of all previously seen documents and rejects any
    new document whose Jaccard similarity to a previous document is at least
    ``threshold``.  Memory grows linearly with the number of unique documents
    seen so far.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        num_hashes: int = 128,
        ngram_size: int = 5,
        num_bands: int = 8,
        rows_per_band: int = 16,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if num_bands * rows_per_band != num_hashes:
            raise ValueError("num_hashes must equal num_bands * rows_per_band")
        self.threshold = threshold
        self._minhash = MinHash(num_hashes, ngram_size)
        self._lsh = LocalitySensitiveHashing(num_bands, rows_per_band)
        self._ngrams: List[Set[str]] = []
        self._seen = 0

    def is_duplicate(self, text: str) -> bool:
        """Return ``True`` if ``text`` is a near-duplicate of a previous document."""
        sig = self._minhash.signature(text)
        candidates = self._lsh.query(sig)
        ngrams = self._minhash.ngrams(text)
        for j in candidates:
            if _jaccard(ngrams, self._ngrams[j]) >= self.threshold:
                return True
        idx = self._seen
        self._lsh.add(sig, idx)
        self._ngrams.append(ngrams)
        self._seen += 1
        return False
