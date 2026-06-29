"""Regression tests for utils.config."""
import argparse
import os
import tempfile
from pathlib import Path

import pytest


from utils.config import (
    load_yaml_config,
    parse_args_with_config,
    NestedNamespace,
    flatten,
    unflatten,
    to_dict,
    validate_keys,
)


def test_load_yaml_config_env_expansion():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.yaml"
        path.write_text("out_dir: ${TMPDIR_TEST}/run\nvalue: 42\n", encoding="utf-8")
        os.environ["TMPDIR_TEST"] = tmpdir
        try:
            cfg = load_yaml_config(str(path))
        finally:
            os.environ.pop("TMPDIR_TEST", None)
        assert Path(cfg["out_dir"]) == Path(tmpdir) / "run"
        assert cfg["value"] == 42


def test_load_yaml_config_strips_bom():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.yaml"
        path.write_bytes(b"\xef\xbb\xbfvalue: 1\n")
        cfg = load_yaml_config(str(path))
        assert cfg["value"] == 1


def test_parse_args_with_config_flat_yaml_and_cli_override():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.yaml"
        path.write_text("batch_size: 8\nlearning_rate: 0.001\n", encoding="utf-8")

        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=1e-5)

        args = parse_args_with_config(parser, ["--config", str(path), "--batch_size", "16"])
        assert args.batch_size == 16
        assert args.learning_rate == 0.001


def test_parse_args_with_config_nested_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.yaml"
        path.write_text("optimizer:\n  lr: 0.01\n  beta: 0.9\nseed: 7\n", encoding="utf-8")

        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=0)

        args = parse_args_with_config(parser, ["--config", str(path)])
        assert args.seed == 7
        assert args.optimizer.lr == 0.01
        assert args.optimizer.beta == 0.9
        assert args.to_dict()["optimizer"]["lr"] == 0.01


def test_parse_args_with_config_nested_cli_override():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "cfg.yaml"
        path.write_text("optimizer:\n  lr: 0.01\n  beta: 0.9\nseed: 7\n", encoding="utf-8")

        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=0)

        args = parse_args_with_config(
            parser,
            ["--config", str(path), "--optimizer.lr", "0.005", "--seed=42"],
        )
        assert args.seed == 42
        assert args.optimizer.lr == 0.005
        assert args.optimizer.beta == 0.9


def test_nested_namespace_get():
    ns = NestedNamespace({"a": {"b": {"c": 123}}, "d": "hello"})
    assert ns.get("a.b.c") == 123
    assert ns.get("a.b") is not None
    assert ns.get("missing", default="x") == "x"
    assert ns.get("d") == "hello"


def test_flatten_unflatten_roundtrip():
    nested = {"a": {"b": 1, "c": {"d": 2}}, "e": [1, 2, 3]}
    flat = flatten(nested)
    assert flat["a.b"] == 1
    assert flat["a.c.d"] == 2
    assert flat["e"] == [1, 2, 3]
    assert unflatten(flat) == nested


def test_to_dict_namespace():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=1)
    args = parser.parse_args(["--x", "2"])
    assert to_dict(args) == {"x": 2}


def test_to_dict_nested():
    ns = NestedNamespace({"x": 1, "y": NestedNamespace({"z": 2})})
    assert to_dict(ns) == {"x": 1, "y": {"z": 2}}


def test_validate_keys_ok():
    ns = NestedNamespace({"a": 1, "b": NestedNamespace({"c": 2})})
    validate_keys(ns, ["a", "b.c"])


def test_validate_keys_missing():
    ns = NestedNamespace({"a": 1, "b": NestedNamespace({"c": None})})
    with pytest.raises(ValueError, match="Missing required config keys"):
        validate_keys(ns, ["a", "b.c", "missing"])
