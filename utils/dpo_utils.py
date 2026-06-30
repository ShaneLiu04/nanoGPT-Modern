"""Direct Preference Optimization (DPO) and related preference-learning losses.

This module provides numerically stable helpers for DPO, IPO, and a simplified
KTO-style loss.  They operate on per-sequence log-probabilities computed under
both the policy model and the reference model.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def compute_sequence_logprob(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the average log-probability of a sequence under a model.

    Parameters
    ----------
    logits : torch.Tensor [B, T, V]
        Unnormalized model logits.
    tokens : torch.Tensor [B, T]
        Target token ids.  The log-prob at position ``i`` is computed for
        ``tokens[:, i]``.
    mask : torch.Tensor [B, T] or None
        Boolean mask selecting positions to include.

    Returns
    -------
    logprobs : torch.Tensor [B]
        Mean token log-probability for each sequence.
    """
    logp = F.log_softmax(logits, dim=-1)
    token_logp = logp.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
    if mask is None:
        return token_logp.mean(dim=-1)
    masked = token_logp * mask.float()
    total = masked.sum(dim=-1)
    count = mask.float().sum(dim=-1).clamp_min(1.0)
    return total / count


def compute_dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute the DPO loss (Bradley-Terry preference model).

    Loss = -log σ(β * (chosen_margin - rejected_margin))

    where ``margin = policy_logp - ref_logp``.

    Parameters
    ----------
    policy_*_logp, ref_*_logp : torch.Tensor [B]
        Sequence log-probs under policy and reference models.
    beta : float
        Temperature controlling the divergence penalty.
    label_smoothing : float
        Mixes the ground-truth label with a neutral label (0.5) to regularize.

    Returns
    -------
    loss : torch.Tensor scalar
    metrics : dict
        Contains ``loss``, ``chosen_rewards``, ``rejected_rewards``, ``margin``.
    """
    policy_ratio = policy_chosen_logp - policy_rejected_logp
    ref_ratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_ratio - ref_ratio)
    # Standard BCE on the preference probability; label_smoothing blends labels.
    labels = torch.ones_like(logits) * (1.0 - label_smoothing) + 0.5 * label_smoothing
    loss = F.binary_cross_entropy_with_logits(logits, labels)

    chosen_rewards = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_rewards = beta * (policy_rejected_logp - ref_rejected_logp)
    margin = chosen_rewards - rejected_rewards
    metrics = {
        "loss": loss.detach(),
        "chosen_rewards": chosen_rewards.detach().mean(),
        "rejected_rewards": rejected_rewards.detach().mean(),
        "margin": margin.detach().mean(),
    }
    return loss, metrics


def compute_ipo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """Compute the IPO loss (Identity Preference Optimization).

    Unlike DPO, IPO does not rely on the Bradley-Terry model and optimizes a
    squared loss directly:

        Loss = (logits - 1 / (2β))^2

    where ``logits = β * (policy_ratio - ref_ratio)``.
    """
    policy_ratio = policy_chosen_logp - policy_rejected_logp
    ref_ratio = ref_chosen_logp - ref_rejected_logp
    logits = beta * (policy_ratio - ref_ratio)
    target = 1.0 / (2.0 * beta)
    loss = ((logits - target) ** 2).mean()

    metrics = {
        "loss": loss.detach(),
        "logits": logits.detach().mean(),
    }
    return loss, metrics


def compute_kto_loss(
    policy_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    is_desirable: torch.Tensor,
    beta: float = 0.1,
    kl_reference: Optional[float] = None,
) -> Tuple[torch.Tensor, dict]:
    """Compute a simplified KTO-style loss.

    Parameters
    ----------
    policy_logp, ref_logp : torch.Tensor [B]
        Sequence log-probs under policy and reference.
    is_desirable : torch.Tensor [B]
        Boolean tensor: ``True`` for chosen/desirable sequences, ``False`` for
        rejected/undesirable sequences.
    beta : float
        Temperature.
    kl_reference : float or None
        Running estimate of KL(ref || policy).  If None, it is computed from the
        batch.

    Returns
    -------
    loss : torch.Tensor scalar
    metrics : dict
    """
    kl = beta * (policy_logp - ref_logp)
    if kl_reference is None:
        kl_reference = kl.detach().mean().item()
    desirable = is_desirable.float()
    # Desirable: maximize KL; undesirable: minimize KL, anchored by reference.
    loss = desirable * (1.0 - kl.exp()) + (1.0 - desirable) * (kl.exp() - 1.0)
    loss = loss.mean()

    metrics = {
        "loss": loss.detach(),
        "kl": kl.detach().mean(),
        "kl_reference": kl_reference,
    }
    return loss, metrics
