"""Shared utilities for reinforcement-learning alignment.

This module provides numerically stable helpers used by both training and
evaluation scripts so that metrics are computed with the same semantics.
"""

from typing import Optional

import torch
import torch.nn.functional as F

LOGPROB_CLIP_MIN: float = -20.0
LOGPROB_CLIP_MAX: float = 0.0


def clip_logprob(logprob: torch.Tensor) -> torch.Tensor:
    """Clamp token log-probabilities to a numerically safe range.

    Parameters
    ----------
    logprob : torch.Tensor
        Tensor of log-probabilities (typically negative).

    Returns
    -------
    torch.Tensor
        Clamped log-probabilities in ``[LOGPROB_CLIP_MIN, LOGPROB_CLIP_MAX]``.
    """
    return torch.clamp(logprob, min=LOGPROB_CLIP_MIN, max=LOGPROB_CLIP_MAX)


def compute_token_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-token log-probabilities from logits and target ids.

    Parameters
    ----------
    logits : torch.Tensor [B, T, V]
    targets : torch.Tensor [B, T]
    mask : torch.Tensor [B, T] or None
        Boolean mask; positions where ``mask == False`` are ignored.

    Returns
    -------
    token_logprobs : torch.Tensor [B, T]
        Log-prob for each target position, clamped to the safe range.
    """
    logp = F.log_softmax(logits, dim=-1)
    token_logprobs = logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    token_logprobs = clip_logprob(token_logprobs)
    if mask is not None:
        token_logprobs = token_logprobs * mask.float()
    return token_logprobs


def compute_kl_divergence(
    ref_logprobs: torch.Tensor,
    policy_logprobs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute KL(ref || policy) in a numerically stable way.

    This uses the reverse KL form consistent with the GRPO training loss:

        KL(ref || policy) = ref_logp - policy_logp

    which is equivalent to ``E_ref[log ref - log policy]`` and does not require
    exponentiating policy log-probabilities.

    Parameters
    ----------
    ref_logprobs : torch.Tensor [B, T]
    policy_logprobs : torch.Tensor [B, T]
    mask : torch.Tensor [B, T] or None
        Boolean mask selecting positions to include.
    reduction : {"mean", "sum", "none"}
        ``mean`` returns the token-averaged KL; ``sum`` returns the total;
        ``none`` returns the per-token KL without reduction.

    Returns
    -------
    torch.Tensor
        Scalar for ``mean``/``sum``, or tensor of shape ``[B, T]`` for ``none``.
    """
    kl = ref_logprobs - policy_logprobs
    if mask is None:
        if reduction == "mean":
            return kl.mean()
        if reduction == "sum":
            return kl.sum()
        return kl

    masked_kl = kl * mask.float()
    if reduction == "none":
        return masked_kl

    total = masked_kl.sum()
    count = mask.sum()
    if reduction == "sum":
        return total
    # mean
    return total / count.clamp_min(1)
