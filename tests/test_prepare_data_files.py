"""Tests for data/prepare.py ``--data_files`` local parquet source (T006)."""

from types import SimpleNamespace

import pytest

from data.prepare import get_args, _load_mixed_dataset


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
