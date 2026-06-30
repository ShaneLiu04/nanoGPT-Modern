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
    config = ModernGPTConfig(
        block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=dropout
    )
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
