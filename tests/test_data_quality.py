"""Tests for data quality pipeline: filtering, dedup, mixing."""
import argparse
import os
import tempfile

import numpy as np
import pytest

from data.filter import (
    CompositeFilter,
    LengthFilter,
    RegexFilter,
    RepetitionFilter,
)
from data.dedup import MinHash, LocalitySensitiveHashing, MinHashDeduplicator, _jaccard
from data.mixer import MixtureStrategy, MixedIterableDataset, mix_datasets


def test_length_filter():
    f = LengthFilter(min_chars=5, max_chars=10)
    assert f("hello") is True
    assert f("hi") is False
    assert f("this is too long") is False


def test_repetition_filter():
    f = RepetitionFilter(n=2, max_repetition_ratio=0.3)
    assert f("abcabcabc") is False  # high 2-gram repetition
    assert f("abcdef") is True


def test_regex_filter():
    f = RegexFilter(require=r"\d+", reject=None)
    assert f("hello 123") is True
    assert f("hello") is False

    f2 = RegexFilter(require=None, reject=r"bad")
    assert f2("good text") is True
    assert f2("bad text") is False


def test_composite_filter_counts():
    f = CompositeFilter([
        LengthFilter(min_chars=5),
        RepetitionFilter(n=2, max_repetition_ratio=0.5),
    ])
    assert f("hello world") is True
    assert f("hi") is False
    texts = ["hello world", "hi", "foo bar baz", "abcabcabc"]
    stats = f.stats(texts)
    assert any(v > 0 for v in stats.values())


def test_minhash_signature_stability():
    m = MinHash(num_hashes=64)
    a = m.signature("hello world")
    b = m.signature("hello world")
    assert a == b
    assert len(a) == 64


def test_minhash_jaccard_consistency():
    m = MinHash(num_hashes=64)
    text1 = "the quick brown fox"
    text2 = "the quick brown fox jumps"
    text3 = "completely different content here"
    j12 = _jaccard(m.ngrams(text1), m.ngrams(text2))
    j13 = _jaccard(m.ngrams(text1), m.ngrams(text3))
    assert 0.0 < j12 < 1.0
    assert j13 < j12


def test_lsh_finds_candidates():
    lsh = LocalitySensitiveHashing(num_bands=8, rows_per_band=8)
    m = MinHash(num_hashes=64)
    sig = m.signature("duplicate text")
    lsh.add(sig, 0)
    lsh.add(sig, 1)
    candidates = lsh.query(sig)
    assert 0 in candidates or 1 in candidates


def test_minhash_deduplicator():
    docs = ["hello world", "hello world", "completely different"]
    dedup = MinHashDeduplicator(threshold=0.85, num_hashes=64, num_bands=8, rows_per_band=8)
    mask = dedup.fit(docs).duplicate_mask()
    assert mask[0] is True
    assert mask[1] is False
    assert mask[2] is True


def test_mixture_strategy_sampling():
    weights = {"a": 0.7, "b": 0.3}
    strategy = MixtureStrategy(weights, temperature=1.0)
    rng = np.random.default_rng(0)
    counts = {"a": 0, "b": 0}
    for _ in range(1000):
        counts[strategy.sample_source(rng)] += 1
    assert counts["a"] > counts["b"]


def test_mixed_iterable_dataset():
    sources = {
        "a": iter([1, 2, 3]),
        "b": iter([10, 20, 30]),
    }
    ds = MixedIterableDataset(sources, {"a": 0.5, "b": 0.5}, total_examples=10, seed=0)
    items = list(ds)
    # Sources only provide 6 examples total.
    assert len(items) == 6
    assert all((it in (1, 2, 3) or it in (10, 20, 30)) for it in items)


def test_mixed_iterable_dataset_state_dict():
    # Use lists as sources so each __iter__ gets a fresh iterator.
    sources = {
        "a": [1, 2, 3, 4, 5],
        "b": [10, 20, 30, 40, 50],
    }
    ds = MixedIterableDataset(sources, {"a": 0.5, "b": 0.5}, total_examples=10, seed=42)
    # Consume a few items.
    it = iter(ds)
    first = [next(it) for _ in range(3)]
    state = ds.state_dict()
    assert state["yielded"] == 3
    assert "rng_state" in state

    # Resume and verify the remaining examples are produced.
    ds2 = MixedIterableDataset(sources, {"a": 0.5, "b": 0.5}, total_examples=10, seed=42)
    ds2.load_state_dict(state)
    resumed = list(ds2)

    ds3 = MixedIterableDataset(sources, {"a": 0.5, "b": 0.5}, total_examples=10, seed=42)
    full = list(ds3)
    # The resumed stream must contain exactly the examples not seen in ``first``.
    assert sorted(resumed) == sorted(full[3:])
    assert len(resumed) == len(full) - 3


def test_mix_datasets_fallback():
    sources = {"a": iter([1, 2]), "b": iter([3, 4])}
    mixed = mix_datasets(sources, {"a": 0.5, "b": 0.5}, seed=0)
    assert isinstance(mixed, MixedIterableDataset)


# ---------------------------------------------------------------------------
# Integration with prepare.py helpers
# ---------------------------------------------------------------------------


def test_build_quality_filter_from_args():
    from data.prepare import _build_quality_filter

    args = argparse.Namespace(
        min_doc_chars=10,
        max_doc_chars=100,
        max_repetition_ratio=0.3,
        repetition_n=5,
        require_regex=None,
        reject_regex=None,
        fasttext_model=None,
    )
    f = _build_quality_filter(args)
    assert f is not None
    assert f("short") is False
    assert f("the quick brown fox jumps over the lazy dog") is True


def test_build_dedup_disabled():
    from data.prepare import _build_dedup

    args = argparse.Namespace(dedup_threshold=0.0)
    assert _build_dedup(args, streaming=False) is None
    assert _build_dedup(args, streaming=True) is None


def test_build_dedup_enabled():
    from data.prepare import _build_dedup

    args = argparse.Namespace(
        dedup_threshold=0.85,
        dedup_ngram=5,
        dedup_num_hashes=64,
        dedup_num_bands=8,
        dedup_rows_per_band=8,
    )
    dedup = _build_dedup(args, streaming=False)
    assert dedup is not None


def test_shard_writer_cleans_stale_shards():
    """_ShardWriter must not include shards left over from previous runs."""
    from data.prepare import _ShardWriter, _concatenate_shards_to_bin

    tmpdir = tempfile.mkdtemp()
    try:
        shard_dir = os.path.join(tmpdir, "train_shards")
        os.makedirs(shard_dir, exist_ok=True)
        # Stale shard from a previous run.
        stale = os.path.join(shard_dir, "train_99999.bin")
        np.array([99], dtype=np.uint16).tofile(stale)

        writer = _ShardWriter(
            output_dir=tmpdir,
            split="train",
            dtype=np.uint16,
            shard_size=100,
            vocab_size=50257,
            eot_token=50256,
        )
        writer.append(np.array([1, 2, 3, 4, 5], dtype=np.uint16))
        writer._flush()
        writer.write_index()

        _concatenate_shards_to_bin(shard_dir, tmpdir, "train", np.uint16)
        out_path = os.path.join(tmpdir, "train.bin")
        arr = np.memmap(out_path, dtype=np.uint16, mode="r")
        assert arr.tolist() == [1, 2, 3, 4, 5]
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
