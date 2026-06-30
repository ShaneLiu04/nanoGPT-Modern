"""Tests for HuggingFace Transformers compatibility wrapper."""

import tempfile

import pytest
import torch

pytest.importorskip("transformers")

from model.hf_model import NanoGPTModernConfig, NanoGPTModernForCausalLM
from model.modern_gpt import ModernGPT, ModernGPTConfig


def test_hf_config_round_trip():
    """``NanoGPTModernConfig`` must round-trip to ``ModernGPTConfig``."""
    original = ModernGPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=64,
        vocab_size=100,
        dropout=0.0,
        rope_scaling={"type": "ntk", "factor": 2.0},
        qk_norm=True,
    )
    hf_config = NanoGPTModernConfig.from_nanogpt_config(original)
    recovered = hf_config.to_nanogpt_config()

    for key in [
        "vocab_size",
        "block_size",
        "n_layer",
        "n_head",
        "n_embd",
        "n_kv_head",
        "dropout",
        "qk_norm",
        "rope_scaling",
    ]:
        assert getattr(recovered, key) == getattr(original, key), key


def test_hf_save_load_state_dict():
    """Saving and loading via HF format must preserve weights."""
    config = ModernGPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=64,
        vocab_size=100,
        dropout=0.0,
    )
    model = ModernGPT(config).eval()
    original_sd = model.state_dict()

    hf_config = NanoGPTModernConfig.from_nanogpt_config(config)
    wrapper = NanoGPTModernForCausalLM(hf_config)
    wrapper.model.load_state_dict(original_sd)

    with tempfile.TemporaryDirectory() as tmpdir:
        wrapper.save_pretrained(tmpdir, safe_serialization=True)
        reloaded = NanoGPTModernForCausalLM.from_pretrained(tmpdir)

    reloaded_sd = reloaded.model.state_dict()
    assert set(original_sd.keys()) == set(reloaded_sd.keys())
    for key in original_sd:
        assert torch.equal(original_sd[key], reloaded_sd[key]), key


def test_hf_forward_matches_native():
    """The HF wrapper forward must match the native model forward."""
    config = ModernGPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=64,
        vocab_size=100,
        dropout=0.0,
    )
    model = ModernGPT(config).eval()
    hf_config = NanoGPTModernConfig.from_nanogpt_config(config)
    wrapper = NanoGPTModernForCausalLM(hf_config)
    wrapper.model.load_state_dict(model.state_dict())

    idx = torch.randint(0, config.vocab_size, (1, 8))
    targets = torch.randint(0, config.vocab_size, (1, 8))

    with torch.no_grad():
        logits_native, loss_native, _ = model(idx, targets=targets)
        out = wrapper(idx, labels=targets)

    assert torch.equal(logits_native, out.logits)
    assert torch.isfinite(out.loss)
    assert abs(out.loss.item() - loss_native.item()) < 1e-6


def test_hf_generate_matches_native():
    """``generate`` through the wrapper must match native ``generate``."""
    config = ModernGPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=64,
        vocab_size=100,
        dropout=0.0,
    )
    model = ModernGPT(config).eval()
    hf_config = NanoGPTModernConfig.from_nanogpt_config(config)
    wrapper = NanoGPTModernForCausalLM(hf_config)
    wrapper.model.load_state_dict(model.state_dict())

    idx = torch.randint(0, config.vocab_size, (1, 5))

    with torch.no_grad():
        native_out = model.generate(idx.clone(), max_new_tokens=8, top_k=1)
        hf_out = wrapper.generate(idx.clone(), max_new_tokens=8, top_k=1)

    assert torch.equal(native_out, hf_out)
