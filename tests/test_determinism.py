"""Determinism regression tests for training."""

import pytest
import torch

from model.modern_gpt import ModernGPT, ModernGPTConfig
from model.attention_utils import reset_gqa_probe_cache
from training.trainer_base import set_seed


def _build_tiny_model():
    config = ModernGPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=128,
        block_size=64,
        vocab_size=100,
        dropout=0.0,
        n_kv_head=2,
        gqa_broadcast="repeat",  # avoid lazy probe side-effects in determinism test
    )
    return ModernGPT(config)


def _run_training_steps(seed, device="cpu", steps=5):
    reset_gqa_probe_cache()
    set_seed(seed)
    model = _build_tiny_model().to(device)
    optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), device_type=device)

    losses = []
    for _ in range(steps):
        input_ids = torch.randint(0, model.config.vocab_size, (2, 32), device=device)
        targets = torch.randint(0, model.config.vocab_size, (2, 32), device=device)
        logits, loss, _ = model(input_ids, targets=targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def test_training_loss_deterministic():
    """Two runs with the same seed must produce identical loss curves."""
    losses_a = _run_training_steps(seed=1337)
    losses_b = _run_training_steps(seed=1337)
    assert losses_a == pytest.approx(losses_b, rel=1e-6, abs=1e-6)


def test_different_seeds_produce_different_losses():
    """Different seeds should generally produce different loss curves."""
    losses_a = _run_training_steps(seed=1337)
    losses_b = _run_training_steps(seed=42)
    assert losses_a != pytest.approx(losses_b, rel=1e-6, abs=1e-6)
