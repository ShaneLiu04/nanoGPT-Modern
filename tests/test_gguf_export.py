"""Tests for the GGUF export path and the in-tree GGUF writer."""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch


from model.gguf_utils import (
    GGMLQuantizationType,
    GGUFWriter,
    dequantize_q8_0,
    quantize_q8_0,
    read_gguf_header,
    read_gguf_tensor_data,
    read_gguf_tensor_info,
)
from model.modern_gpt import ModernGPT, ModernGPTConfig


def _tiny_config():
    return ModernGPTConfig(
        n_layer=1, n_head=2, n_embd=16, block_size=32, vocab_size=50, dropout=0.0
    )


def test_q8_0_roundtrip():
    """Q8_0 quantization/dequantization must be close to the original tensor."""
    w = torch.randn(8, 16)
    raw = quantize_q8_0(w)
    rec = dequantize_q8_0(raw, tuple(w.shape))
    mse = np.mean((w.numpy() - rec) ** 2)
    assert mse < 1e-4


def test_q8_0_zero_block():
    """A block of all zeros must round-trip without NaNs."""
    w = torch.zeros(32)
    raw = quantize_q8_0(w)
    rec = dequantize_q8_0(raw, (32,))
    assert np.allclose(rec, 0.0)


def test_gguf_writer_f32_roundtrip():
    """F32 tensors must round-trip through the minimal GGUF writer."""
    w = torch.randn(5, 7)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.gguf")
        writer = GGUFWriter(path, "testarch")
        writer.add_tensor("w", w, GGMLQuantizationType.F32)
        writer.write()

        header = read_gguf_header(path)
        assert header["version"] == 3
        assert header["tensor_count"] == 1

        infos = read_gguf_tensor_info(path)
        assert infos[0]["name"] == "w"
        assert infos[0]["shape"] == tuple(w.shape)
        assert infos[0]["dtype"] == GGMLQuantizationType.F32

        rec = read_gguf_tensor_data(path, infos[0])
        assert np.allclose(w.numpy(), rec)


def test_gguf_writer_q8_0_roundtrip():
    """Q8_0 tensors must round-trip through the minimal GGUF writer."""
    w = torch.randn(5, 7)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.gguf")
        writer = GGUFWriter(path, "testarch")
        writer.add_tensor("w", w, GGMLQuantizationType.Q8_0)
        writer.write()

        infos = read_gguf_tensor_info(path)
        assert infos[0]["dtype"] == GGMLQuantizationType.Q8_0

        rec = read_gguf_tensor_data(path, infos[0])
        assert np.mean((w.numpy() - rec) ** 2) < 1e-4


def test_export_gguf_cli_q8_0():
    """``export_gguf.py`` must produce a valid GGUF file for a tiny checkpoint."""
    cfg = _tiny_config()
    model = ModernGPT(cfg).eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "ckpt.pt")
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, ckpt)
        out = os.path.join(tmpdir, "model.q8_0.gguf")

        result = subprocess.run(
            [
                sys.executable,
                "export_gguf.py",
                "--checkpoint",
                ckpt,
                "--out",
                out,
                "--quant",
                "q8_0",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        header = read_gguf_header(out)
        assert header["tensor_count"] > 0

        infos = read_gguf_tensor_info(out)
        names = {info["name"] for info in infos}
        assert "transformer.h.0.attn.q_proj.weight" in names


def test_export_gguf_cli_f16():
    """CLI export with --quant f16 must store tensors as F16."""
    cfg = _tiny_config()
    model = ModernGPT(cfg).eval()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = os.path.join(tmpdir, "ckpt.pt")
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, ckpt)
        out = os.path.join(tmpdir, "model.f16.gguf")

        result = subprocess.run(
            [
                sys.executable,
                "export_gguf.py",
                "--checkpoint",
                ckpt,
                "--out",
                out,
                "--quant",
                "f16",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        infos = read_gguf_tensor_info(out)
        for info in infos:
            assert info["dtype"] == GGMLQuantizationType.F16
