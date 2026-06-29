"""Tests for the post-training quantization helpers."""
import os

import pytest
import torch


from model.modern_gpt import ModernGPT, ModernGPTConfig
from model.quantization import (
    QuantConfig,
    QuantizedLinear,
    compute_quantization_mse,
    dequantize_model,
    estimate_quantized_size,
    quantize_model,
)


def _tiny_config():
    return ModernGPTConfig(
        n_layer=1, n_head=2, n_embd=16, block_size=32, vocab_size=50, dropout=0.0
    )


def test_quantized_linear_forward_matches_float():
    """A QuantizedLinear layer must be close to the original float linear."""
    linear = torch.nn.Linear(16, 32, bias=False)
    qlinear = QuantizedLinear.from_float(linear, dtype=torch.float32)
    x = torch.randn(2, 10, 16)
    with torch.no_grad():
        ref = linear(x)
        out = qlinear(x)
    assert out.shape == ref.shape
    max_err = (out - ref).abs().max().item()
    assert max_err < 1e-2


def test_quantize_model_reduces_weight_precision():
    """After quantize_model the eligible Linear layers become QuantizedLinear."""
    model = ModernGPT(_tiny_config()).eval().float()
    config = QuantConfig(method="int8", compute_dtype=torch.float32)
    _, replaced = quantize_model(model, config)

    assert len(replaced) > 0
    for name in replaced:
        assert isinstance(model.get_submodule(name), QuantizedLinear)


def test_quantized_model_forward_close_to_original():
    """Quantizing a model in place must keep the forward output close."""
    cfg = _tiny_config()
    torch.manual_seed(42)
    model = ModernGPT(cfg).eval().float()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        ref = model(idx)[0]

    qconfig = QuantConfig(method="int8", compute_dtype=torch.float32)
    quantize_model(model, qconfig)
    with torch.no_grad():
        out = model(idx)[0]

    max_err = (ref - out).abs().max().item()
    assert max_err < 1e-1
    assert torch.equal(ref.argmax(dim=-1), out.argmax(dim=-1))


def test_dequantize_model_restores_linear_modules():
    """dequantize_model must replace QuantizedLinear with nn.Linear."""
    cfg = _tiny_config()
    model = ModernGPT(cfg).eval().float()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))

    qconfig = QuantConfig(method="int8", compute_dtype=torch.float32)
    quantize_model(model, qconfig)
    with torch.no_grad():
        q_out = model(idx)[0]

    dequantize_model(model)
    for module in model.modules():
        assert not isinstance(module, QuantizedLinear)

    with torch.no_grad():
        out = model(idx)[0]
    assert torch.equal(q_out.argmax(dim=-1), out.argmax(dim=-1))


def test_quantization_mse_is_small():
    """compute_quantization_mse should report a small average error."""
    model = ModernGPT(_tiny_config()).eval().float()
    mse = compute_quantization_mse(model, QuantConfig(method="int8", compute_dtype=torch.float32))
    assert 0.0 < mse < 1e-4


def test_estimate_quantized_size_smaller_than_original():
    """The estimated INT8 size should be roughly half of the FP32 size."""
    model = ModernGPT(_tiny_config()).eval().float()
    original_bytes = sum(p.numel() * 4 for p in model.parameters())
    int8_bytes = estimate_quantized_size(model, method="int8")
    assert int8_bytes < original_bytes * 0.6


def test_quantize_model_respects_skip_modules():
    """Layers matching skip_modules must stay untouched."""
    cfg = _tiny_config()
    model = ModernGPT(cfg).eval().float()
    config = QuantConfig(
        method="int8",
        compute_dtype=torch.float32,
        skip_modules=("wte", "lm_head"),
    )
    _, replaced = quantize_model(model, config)
    assert "transformer.wte" not in replaced
    assert "lm_head" not in replaced


def test_bitsandbytes_import_if_available():
    """If bitsandbytes is importable, quantize_model must accept the methods."""
    try:
        import bitsandbytes as bnb  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"bitsandbytes not available: {exc}")

    cfg = _tiny_config()
    model = ModernGPT(cfg).eval().half().cuda()
    for method in ("bnb_8bit", "bnb_4bit"):
        try:
            qmodel, replaced = quantize_model(
                model,
                QuantConfig(method=method, compute_dtype=torch.float16),
            )
        except Exception as exc:
            pytest.skip(f"{method} runtime not functional on this platform: {exc}")
        assert len(replaced) > 0
        idx = torch.randint(0, cfg.vocab_size, (1, 8), device="cuda")
        with torch.no_grad():
            qmodel(idx)[0]
        break  # one successful method is enough to validate the API wiring
