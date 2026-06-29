"""Tests for GRPO advantage normalization and clipping."""
import os

import pytest
import torch


from training.train_grpo import GRPOTrainer


def _rewards():
    # Two prompts (columns), three rollouts per prompt (rows).
    # Column 0 has a large spread, column 1 has a small spread — exercises
    # both per-group and per-batch scaling behavior.
    return torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [4.0, 0.5],
        ],
        dtype=torch.float32,
    )


class TestAdvantageGroup:
    def test_group_normalizes_per_prompt(self):
        r = _rewards()
        adv = GRPOTrainer.compute_advantages(r, "group")
        # Per-column mean must be ~0.
        assert torch.allclose(adv.mean(dim=0), torch.zeros(2), atol=1e-6)
        # Per-column std must be ~1.
        assert torch.allclose(adv.std(dim=0), torch.ones(2), atol=1e-6)

    def test_group_is_zero_mean_in_batch(self):
        r = _rewards()
        adv = GRPOTrainer.compute_advantages(r, "group")
        # Mean over the whole tensor is the average of per-column zeros -> 0.
        assert abs(adv.mean().item()) < 1e-6


class TestAdvantageBatch:
    def test_batch_normalizes_globally(self):
        r = _rewards()
        adv = GRPOTrainer.compute_advantages(r, "batch")
        assert abs(adv.mean().item()) < 1e-6
        assert abs(adv.std().item() - 1.0) < 1e-5

    def test_batch_differs_from_group_when_columns_have_different_std(self):
        r = _rewards()
        group_adv = GRPOTrainer.compute_advantages(r, "group")
        batch_adv = GRPOTrainer.compute_advantages(r, "batch")
        # They differ because column 0 has larger std than column 1.
        assert not torch.allclose(group_adv, batch_adv, atol=1e-3)


class TestAdvantageNone:
    def test_none_only_centers(self):
        r = _rewards()
        adv = GRPOTrainer.compute_advantages(r, "none")
        # Per-column centered (since we use mean(dim=0))
        assert torch.allclose(adv.mean(dim=0), torch.zeros(2), atol=1e-6)
        # Std is NOT rescaled to 1.
        assert not torch.allclose(adv.std(dim=0), torch.ones(2), atol=1e-2)


class TestAdvantageClip:
    def test_clip_zero_disables_clipping(self):
        r = _rewards()
        adv_unclipped = GRPOTrainer.compute_advantages(r, "group", adv_clip=0.0)
        adv_zero = GRPOTrainer.compute_advantages(r, "group", adv_clip=0.0)
        assert torch.allclose(adv_unclipped, adv_zero)

    def test_clip_bounds_advantages(self):
        r = _rewards()
        adv = GRPOTrainer.compute_advantages(r, "group", adv_clip=1.0)
        assert adv.max().item() <= 1.0 + 1e-6
        assert adv.min().item() >= -1.0 - 1e-6

    def test_clip_tighter_value_stronger_clipping(self):
        r = _rewards()
        adv_loose = GRPOTrainer.compute_advantages(r, "group", adv_clip=2.0)
        adv_tight = GRPOTrainer.compute_advantages(r, "group", adv_clip=0.5)
        # Tighter clip must compress the range at least as much.
        assert adv_tight.abs().max().item() <= adv_loose.abs().max().item() + 1e-6


class TestAdvantageEdgeCases:
    def test_constant_rewards_group_yields_near_zero(self):
        # All rewards equal -> std clamped to 1e-8, advantages ~0.
        r = torch.full((4, 2), 3.0)
        adv = GRPOTrainer.compute_advantages(r, "group")
        assert adv.abs().max().item() < 1e-3

    def test_invalid_mode_raises(self):
        r = _rewards()
        with pytest.raises(ValueError):
            GRPOTrainer.compute_advantages(r, "invalid_mode")

    def test_returns_correct_shape(self):
        r = _rewards()
        adv = GRPOTrainer.compute_advantages(r, "group")
        assert adv.shape == r.shape
