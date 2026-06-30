"""End-to-end smoke tests for training/train_dpo.py."""
import argparse
import os
import tempfile

import pytest
import torch

from model.modern_gpt import ModernGPT, ModernGPTConfig
from training.train_dpo import DPOTrainer


def _make_minimal_checkpoint(path, seed=42):
    torch.manual_seed(seed)
    config = ModernGPTConfig(
        block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0,
    )
    model = ModernGPT(config)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "model_type": "modern",
        },
        path,
    )


def test_dpo_trainer_scheduler_and_train_step():
    """DPOTrainer should initialize its scheduler and complete one train step."""
    tmpdir = tempfile.mkdtemp()
    try:
        ckpt = os.path.join(tmpdir, "sft.pt")
        _make_minimal_checkpoint(ckpt)

        args = argparse.Namespace(
            init_from=ckpt,
            ref_from=None,
            out_dir=os.path.join(tmpdir, "out"),
            batch_size=2,
            gradient_accumulation_steps=1,
            max_length=16,
            epochs=1,
            learning_rate=1e-4,
            min_lr=1e-5,
            lr_schedule="cosine",
            warmup_iters=0,
            weight_decay=0.0,
            grad_clip=1.0,
            beta=0.1,
            preference_loss="dpo",
            preference_source="synthetic",
            label_smoothing=0.0,
            eval_interval=1000,
            seed=42,
            device="cpu",
            use_wandb=False,
            backend="gloo",
            keep_last_n=0,
            resume=None,
            config=None,
            attn_backend="auto",
            num_train=8,
            num_val=4,
            vocab_size=50257,
        )

        trainer = DPOTrainer(args)
        # Scheduler must exist and produce a finite LR.
        lr = trainer.scheduler(0)
        assert lr > 0
        # Run one train step.
        train_loss = trainer._run_epoch(0, is_train=True)
        assert train_loss >= 0
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
