"""Tests for GRPO trainer behavior."""

import argparse
import os
import tempfile

import pytest
import torch

from model.modern_gpt import ModernGPT, ModernGPTConfig
from training.train_grpo import GRPOTrainer
from utils.checkpoint import save_checkpoint


def _make_sft_checkpoint(path, dropout=0.0):
    config = ModernGPTConfig(block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=dropout)
    model = ModernGPT(config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_checkpoint(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        iter_num=0,
        best_val_loss=1.0,
        config=config,
        out_dir=os.path.dirname(path),
        filename=os.path.basename(path),
    )


def test_grpo_rejects_dropout_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        init_from = os.path.join(tmpdir, "sft.pt")
        ref_from = init_from
        _make_sft_checkpoint(init_from, dropout=0.1)

        args = argparse.Namespace(
            init_from=init_from,
            ref_from=ref_from,
            out_dir=os.path.join(tmpdir, "out"),
            group_size=2,
            num_steps=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_prompt_len=16,
            max_response_len=8,
            learning_rate=1e-5,
            min_lr=1e-6,
            weight_decay=0.01,
            grad_clip=1.0,
            beta=0.04,
            eps=0.2,
            lr_schedule="cosine",
            seed=0,
            device="cpu",
            backend="nccl",
            use_wandb=False,
            eval_interval=1,
            num_train=4,
            num_val=2,
            resume=None,
            allow_dropout=False,
            keep_last_n=0,
        )
        with pytest.raises(ValueError, match="GRPO requires dropout=0.0"):
            GRPOTrainer(args)


def test_grpo_allows_dropout_with_flag():
    with tempfile.TemporaryDirectory() as tmpdir:
        init_from = os.path.join(tmpdir, "sft.pt")
        ref_from = init_from
        _make_sft_checkpoint(init_from, dropout=0.1)

        args = argparse.Namespace(
            init_from=init_from,
            ref_from=ref_from,
            out_dir=os.path.join(tmpdir, "out"),
            group_size=2,
            num_steps=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_prompt_len=16,
            max_response_len=8,
            learning_rate=1e-5,
            min_lr=1e-6,
            weight_decay=0.01,
            grad_clip=1.0,
            beta=0.04,
            eps=0.2,
            lr_schedule="cosine",
            seed=0,
            device="cpu",
            backend="nccl",
            use_wandb=False,
            eval_interval=1,
            num_train=4,
            num_val=2,
            resume=None,
            allow_dropout=True,
            keep_last_n=0,
        )
        trainer = GRPOTrainer(args)
        trainer.cleanup()


# ---------------------------------------------------------------------------
# Migrated from tests/test_bugfixes.py (FR11b split)
# ---------------------------------------------------------------------------

from types import SimpleNamespace

import numpy as np
from helpers import tiny_modern_config


def _tiny_grpo_args(tmpdir, ckpt_path, **overrides):
    """Build a minimal args Namespace for GRPO testing."""
    defaults = dict(
        init_from=ckpt_path,
        ref_from=ckpt_path,
        out_dir=tmpdir,
        group_size=2,
        num_steps=4,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_prompt_len=32,
        max_response_len=16,
        learning_rate=1e-4,
        min_lr=1e-5,
        weight_decay=0.0,
        grad_clip=1.0,
        beta=0.04,
        eps=0.2,
        lr_schedule="constant",
        seed=1337,
        device="cpu",
        backend="nccl",
        use_wandb=False,
        eval_interval=10,
        num_train=4,
        num_val=4,
        resume=None,
        config=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_tiny_checkpoint(tmpdir):
    """Create a tiny ModernGPT checkpoint on disk and return its path."""
    config = tiny_modern_config(
        n_layer=1,
        block_size=128,
        vocab_size=50257,
    )
    model = ModernGPT(config)
    opt = model.configure_optimizers(0.0, 1e-3, (0.9, 0.95), "cpu")
    path = save_checkpoint(
        model,
        opt,
        iter_num=0,
        best_val_loss=float("inf"),
        config=config,
        out_dir=tmpdir,
        filename="ckpt.pt",
    )
    return path


def test_grpo_batched_logprobs_consistency():
    """Batched logprob computation must match per-sequence logprobs."""
    from training.train_grpo import GRPOTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_tiny_checkpoint(tmpdir)
        args = _tiny_grpo_args(tmpdir, ckpt, num_train=2, num_val=2)
        trainer = GRPOTrainer(args)
        try:
            model = trainer.policy
            model.eval()

            sequences = [
                [1, 2, 3, 4, 5],
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22],
            ]
            prompt_lens = [2, 3, 1]
            response_lens = [3, 3, 2]

            batched_logps, masks = trainer._batch_logprobs(
                model, sequences, prompt_lens, response_lens
            )

            # Compare response-token logprobs against per-sequence computation.
            for i, seq in enumerate(sequences):
                inp = torch.tensor([seq[:-1]], dtype=torch.long)
                tgt = torch.tensor([seq[1:]], dtype=torch.long)
                with torch.no_grad():
                    logits, _, _ = model(inp)
                    logp = torch.log_softmax(logits, dim=-1)
                    single_logps = logp.gather(2, tgt.unsqueeze(-1)).squeeze(-1)
                resp_mask = masks[i, : len(seq) - 1]
                assert torch.allclose(
                    batched_logps[i, : len(seq) - 1][resp_mask],
                    single_logps[0][resp_mask],
                    atol=1e-5,
                )
                expected_mask = [False] * (prompt_lens[i] - 1) + [True] * response_lens[i]
                assert torch.equal(resp_mask, torch.tensor(expected_mask))

            print("[OK] GRPO batched logprob computation is consistent")
        finally:
            trainer.cleanup()


def test_grpo_gradient_accumulation():
    """GRPO optimizer should step only every gradient_accumulation_steps rollouts."""
    from training.train_grpo import GRPOTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_tiny_checkpoint(tmpdir)
        args = _tiny_grpo_args(
            tmpdir,
            ckpt,
            num_steps=6,
            batch_size=2,
            group_size=2,
            gradient_accumulation_steps=3,
            num_train=4,
            num_val=4,
        )
        trainer = GRPOTrainer(args)

        # Mock sampling to avoid expensive generation.
        vocab_size = trainer.raw_model.config.vocab_size
        prompt_len = 4
        response_len = 4

        def _mock_sample_group(prompts, answers, prompt_tokens=None):
            group_size, batch_size = args.group_size, len(prompts)
            responses = [["dummy"] * batch_size for _ in range(group_size)]
            response_ids = [
                [[vocab_size // 2] * response_len for _ in range(batch_size)]
                for _ in range(group_size)
            ]
            rewards = np.ones((group_size, batch_size), dtype=np.float32)

            flat_sequences = []
            flat_prompt_lens = []
            flat_response_lens = []
            for g in range(group_size):
                for b in range(batch_size):
                    ptoks = prompt_tokens[b] if prompt_tokens else [0] * prompt_len
                    full = ptoks + response_ids[g][b]
                    flat_sequences.append(full)
                    flat_prompt_lens.append(len(ptoks))
                    flat_response_lens.append(response_len)

            old_logprobs, masks = trainer._batch_logprobs(
                trainer.policy, flat_sequences, flat_prompt_lens, flat_response_lens
            )
            ref_logprobs, _ = trainer._batch_logprobs(
                trainer.ref, flat_sequences, flat_prompt_lens, flat_response_lens
            )

            return {
                "responses": responses,
                "response_ids": response_ids,
                "rewards": rewards,
                "old_logprobs": old_logprobs,
                "ref_logprobs": ref_logprobs,
                "masks": masks,
                "prompt_lens": [[prompt_len] * batch_size for _ in range(group_size)],
                "response_lens": [[response_len] * batch_size for _ in range(group_size)],
                "prompt_tokens": prompt_tokens or [[0] * prompt_len for _ in range(batch_size)],
            }

        original_sample_group = trainer.sample_group
        trainer.sample_group = _mock_sample_group

        step_counts = []
        original_optimizer_step = trainer.optimizer.step

        def _counting_step():
            step_counts.append(len(step_counts))
            return original_optimizer_step()

        trainer.optimizer.step = _counting_step

        try:
            trainer.train()
        finally:
            trainer.sample_group = original_sample_group
            trainer.optimizer.step = original_optimizer_step
            trainer.cleanup()

        # 6 rollouts / 3 accum = 2 optimizer steps.
        assert len(step_counts) == 2, f"expected 2 optimizer steps, got {len(step_counts)}"
        print("[OK] GRPO gradient accumulation steps correctly")


def test_iterative_grpo_rejection_pool():
    """IterativeGRPOTrainer must build train/val/rejection pools."""
    from training.iterative_grpo import IterativeGRPOTrainer

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = _make_tiny_checkpoint(tmpdir)
        args = _tiny_grpo_args(tmpdir, ckpt, num_train=4, num_val=4)
        args.ref_update_interval = 2
        args.ref_update_ratio = 0.5
        args.rejection_interval = 2
        args.rejection_samples = 4
        args.rejection_top_k = 2
        args.rejection_sft_steps = 1
        args.rejection_sft_lr = 1e-4
        args.rejection_batch_size = 2

        trainer = IterativeGRPOTrainer(args)
        try:
            assert hasattr(trainer, "rejection_pool")
            assert len(trainer.rejection_pool) == 3 * args.num_train
            assert trainer.ref_updates_done == 0
            assert trainer.rejection_rounds_done == 0
            print("[OK] IterativeGRPOTrainer builds data pools")
        finally:
            trainer.cleanup()
