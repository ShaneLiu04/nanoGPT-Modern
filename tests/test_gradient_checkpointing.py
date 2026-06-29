"""Regression tests for gradient checkpointing."""
import pytest
import torch

from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from model.modern_gpt import ModernGPT, ModernGPTConfig


@pytest.mark.parametrize("model_cls, config_cls", [
    (ModernGPT, ModernGPTConfig),
    (BaselineGPT, BaselineGPTConfig),
])
def test_gradient_checkpointing_runs(model_cls, config_cls):
    """Gradient checkpointing must produce a valid backward pass."""
    config = config_cls(
        n_layer=2, n_head=4, n_embd=128, block_size=64,
        vocab_size=100, dropout=0.0, gradient_checkpointing=True,
    )
    model = model_cls(config).train()
    optimizer = model.configure_optimizers(0.0, 1e-3, (0.9, 0.95), "cpu")

    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    targets = torch.randint(0, config.vocab_size, (2, 32))

    out = model(input_ids, targets=targets)
    loss = out[1]
    assert loss.requires_grad

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # All trainable parameters should receive gradients.
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no gradient"


def test_gradient_checkpointing_no_cache_in_training():
    """When gradient checkpointing is on, training forward must not return KV cache."""
    config = ModernGPTConfig(
        n_layer=2, n_head=4, n_embd=128, block_size=64,
        vocab_size=100, dropout=0.0, gradient_checkpointing=True, n_kv_head=2,
    )
    model = ModernGPT(config).train()
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    targets = torch.randint(0, config.vocab_size, (1, 16))

    logits, loss, past_kvs = model(input_ids, targets=targets, use_cache=True)
    assert past_kvs is None or all(pk is None for pk in past_kvs)


def test_gradient_checkpointing_disabled_by_default():
    """Default config must not enable gradient checkpointing."""
    config = ModernGPTConfig()
    assert not config.gradient_checkpointing
