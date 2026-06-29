"""Tests for shared RL utilities."""
import pytest
import torch

from utils.rl_utils import clip_logprob, compute_kl_divergence, compute_token_logprobs


def test_clip_logprob():
    x = torch.tensor([-30.0, -10.0, 1.0])
    out = clip_logprob(x)
    assert out[0].item() == -20.0
    assert out[1].item() == -10.0
    assert out[2].item() == 0.0


def test_compute_token_logprobs_shape_and_clamp():
    B, T, V = 2, 4, 8
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)

    logp = compute_token_logprobs(logits, targets, mask=mask)
    assert logp.shape == (B, T)
    assert (logp <= 0.0).all()
    assert (logp >= -20.0).all()
    # Masked positions should be zeroed.
    assert (logp[~mask] == 0.0).all()


def test_compute_kl_divergence_reverse_kl():
    ref = torch.tensor([[-1.0, -2.0, -3.0]])
    policy = torch.tensor([[-1.5, -1.5, -1.5]])
    kl = compute_kl_divergence(ref, policy, reduction="mean")
    expected = (ref - policy).mean()
    assert kl.item() == pytest.approx(expected.item(), rel=1e-6)


def test_compute_kl_divergence_with_mask():
    ref = torch.tensor([[-1.0, -2.0, -3.0]])
    policy = torch.tensor([[-1.5, -1.5, -1.5]])
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    kl = compute_kl_divergence(ref, policy, mask=mask, reduction="mean")
    expected = ((ref - policy) * mask.float()).sum() / mask.sum()
    assert kl.item() == pytest.approx(expected.item(), rel=1e-6)
