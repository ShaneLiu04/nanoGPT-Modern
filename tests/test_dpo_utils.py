"""Tests for DPO/IPO/KTO preference-learning utilities."""
import pytest
import torch

from utils.dpo_utils import (
    compute_dpo_loss,
    compute_ipo_loss,
    compute_kto_loss,
    compute_sequence_logprob,
)


def test_sequence_logprob_basic():
    logits = torch.randn(2, 5, 10)
    tokens = torch.randint(0, 10, (2, 5))
    logp = compute_sequence_logprob(logits, tokens)
    assert logp.shape == (2,)
    assert torch.isfinite(logp).all()


def test_sequence_logprob_with_mask():
    logits = torch.randn(2, 5, 10)
    tokens = torch.randint(0, 10, (2, 5))
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    logp = compute_sequence_logprob(logits, tokens, mask)
    assert logp.shape == (2,)


def test_dpo_loss_decreases_when_policy_prefers_chosen():
    """If policy assigns much higher prob to chosen, DPO loss should be low."""
    policy_chosen = torch.tensor([0.0])
    policy_rejected = torch.tensor([-10.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    loss, metrics = compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )
    assert loss.item() < 0.4
    assert metrics["margin"].item() > 0.0


def test_dpo_loss_increases_when_policy_prefers_rejected():
    """If policy assigns higher prob to rejected, DPO loss should be high."""
    policy_chosen = torch.tensor([-10.0])
    policy_rejected = torch.tensor([0.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])
    loss, _ = compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )
    assert loss.item() > 0.5


def test_ipo_loss_is_finite():
    policy_chosen = torch.tensor([0.0, -1.0])
    policy_rejected = torch.tensor([-1.0, 0.0])
    ref_chosen = torch.tensor([0.0, 0.0])
    ref_rejected = torch.tensor([0.0, 0.0])
    loss, metrics = compute_ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)
    assert torch.isfinite(loss)
    assert "logits" in metrics


def test_kto_loss_is_finite():
    policy_logp = torch.tensor([0.0, -1.0])
    ref_logp = torch.tensor([0.0, 0.0])
    is_desirable = torch.tensor([True, False])
    loss, metrics = compute_kto_loss(policy_logp, ref_logp, is_desirable, beta=0.1)
    assert torch.isfinite(loss)
    assert "kl" in metrics


def test_dpo_loss_gradients_flow():
    policy_chosen = torch.tensor([0.0], requires_grad=True)
    policy_rejected = torch.tensor([-1.0], requires_grad=True)
    ref_chosen = torch.tensor([0.0]).detach()
    ref_rejected = torch.tensor([0.0]).detach()
    loss, _ = compute_dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    loss.backward()
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None
