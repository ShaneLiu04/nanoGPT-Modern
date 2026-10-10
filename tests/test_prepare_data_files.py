"""Tests for data/prepare.py ``--data_files`` local parquet source (T006)."""

from types import SimpleNamespace

import pytest

from data.prepare import _load_mixed_dataset, get_args


def _tiny_parquet(path, texts):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"text": texts})
    pq.write_table(table, str(path))


def test_get_args_accepts_data_files_flag(tmp_path):
    args = get_args(["--data_files", str(tmp_path / "*.parquet")])
    assert args.data_files == str(tmp_path / "*.parquet")
    args = get_args([])
    assert args.data_files is None


def test_load_mixed_dataset_data_files_no_match_raises(tmp_path):
    args = SimpleNamespace(
        data_files=str(tmp_path / "no-such-*.parquet"),
        mixture_config=None,
        split="train",
    )
    with pytest.raises(FileNotFoundError):
        _load_mixed_dataset(args)


def test_load_mixed_dataset_data_files_loads_local_parquet(tmp_path):
    p1 = tmp_path / "a.parquet"
    p2 = tmp_path / "b.parquet"
    _tiny_parquet(p1, ["hello world one", "hello world two"])
    _tiny_parquet(p2, ["hello world three"])

    args = SimpleNamespace(
        data_files=f"{tmp_path}/*.parquet",
        mixture_config=None,
        split="train",
    )
    ds = _load_mixed_dataset(args)
    assert len(ds) == 3
    texts = sorted(ds["text"])
    assert texts == [
        "hello world one",
        "hello world three",
        "hello world two",
    ]


def test_load_mixed_dataset_data_files_takes_priority(tmp_path):
    """data_files is used directly; no hub access is attempted."""
    p = tmp_path / "only.parquet"
    _tiny_parquet(p, ["local doc"])

    args = SimpleNamespace(
        data_files=str(p),
        mixture_config=None,
        split="train",
    )
    ds = _load_mixed_dataset(args)
    assert ds["text"] == ["local doc"]


# ---------------------------------------------------------------------------
# Migrated from tests/test_bugfixes.py (FR11b split)
# ---------------------------------------------------------------------------

import os
import tempfile

import numpy as np


def test_prepare_shard_writer_dtype_and_index():
    """_ShardWriter must produce readable shards and a dtype-aware .idx."""
    from data.openwebtext import _detect_dtype_from_index
    from data.prepare import _concatenate_shards_to_bin, _ShardWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = _ShardWriter(
            output_dir=tmpdir,
            split="train",
            dtype=np.uint32,
            shard_size=100,
            vocab_size=70000,
            eot_token=50256,
        )
        tokens = np.arange(1000, 1100, dtype=np.uint32)
        writer.append(tokens)
        writer.record_doc_boundary()
        writer.append(tokens + 100)
        writer.record_doc_boundary()
        meta = writer.close()
        writer.write_index()

        assert meta["total_tokens"] == 200
        assert meta["num_docs"] == 2

        _concatenate_shards_to_bin(writer.shard_dir, tmpdir, "train", np.uint32)
        bin_path = os.path.join(tmpdir, "train.bin")
        idx_path = os.path.join(tmpdir, "train.idx")
        assert os.path.exists(bin_path)
        assert os.path.exists(idx_path)

        detected = _detect_dtype_from_index(bin_path)
        assert detected == np.uint32

        arr = np.memmap(bin_path, dtype=np.uint32, mode="r")
        try:
            assert len(arr) == 200
            assert arr[0] == 1000
            assert arr[100] == 1100
        finally:
            del arr  # close memmap so Windows can clean up the temp dir
        print("[OK] Shard writer + dtype-aware index works")


def test_prepare_dtype_selection():
    """_select_dtype must pick uint16 for small vocab and uint32 for large vocab."""
    from data.prepare import _select_dtype

    assert _select_dtype(50000) == np.uint16
    assert _select_dtype(65535) == np.uint16
    assert _select_dtype(65536) == np.uint32
    assert _select_dtype(100000) == np.uint32
    print("[OK] prepare.py dtype selection works")
