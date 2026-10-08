"""Tests for FR3a (AR-v0.5.0-hpc-readiness / T003): resume integrity.

Training-side resume hardening contract under test:

  * ``CheckpointManager.load`` (training/trainer_base.py): when the loaded
    extra state contains ``rng_state``, ``set_rng_state`` (from
    utils.checkpoint) is called with that bundle, so a resumed run continues
    the exact RNG stream of the save point.
  * ``PretrainTrainer`` (training/train_pretrain.py):
      - ``resume_offset`` really advances during training (one
        ``batch_actual_size`` per consumed batch) and is persisted as
        ``consumed_chunks`` in checkpoints; a resumed trainer's first batch
        equals the next batch of the uninterrupted run (data-position
        resume).
      - ``early_stop_counter`` is persisted and restored; old-format
        checkpoints without the key default to 0 without error.
      - new CLI flag ``--ckpt_interval`` (int, default 0): when > 0,
        ``latest_ckpt.pt`` is additionally saved every ckpt_interval steps
        (off eval points); when 0 (default), latest saves happen only at
        eval points / the final block.
      - old-format checkpoints (no rng_state / resume_offset /
        early_stop_counter / consumed_chunks keys) load without raising and
        default every missing field.

Tests run tiny synthetic OpenWebText-format data on CPU: small
``train.bin``/``val.bin`` (np.uint16, chunked as block_size+1 tokens per
chunk) + ``.idx`` metadata + a *sequential* ``train.bin.shuffle.idx``
(np.arange) so chunk order is deterministic for the resume-equality
assertions.  Trainers are built from a fully populated argparse.Namespace
(no CLI), device="cpu", model="baseline", and cleaned up via
``trainer.cleanup()`` in ``finally`` blocks.
"""

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

import training.trainer_base  # noqa: F401  (monkeypatch target module)
import utils.checkpoint as checkpoint_mod
from training.train_pretrain import PretrainTrainer
from utils.checkpoint import set_rng_state


# ---------------------------------------------------------------------------
# Helpers: tiny synthetic data + full args namespace
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 32
_N_TRAIN_CHUNKS = 200
_N_VAL_CHUNKS = 40


def _write_tiny_data(tmp_path):
    """Write tiny train/val .bin + .idx + sequential train shuffle idx."""
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, n_chunks in (("train", _N_TRAIN_CHUNKS), ("val", _N_VAL_CHUNKS)):
        total = n_chunks * (_BLOCK_SIZE + 1)
        # Deterministic token pattern, all ids < 50257 and uint16-safe.
        tokens = ((np.arange(total, dtype=np.int64) * 7 + 3) % 50000)
        tokens.astype(np.uint16).tofile(data_dir / f"{split}.bin")
        (data_dir / f"{split}.idx").write_text(
            f"dtype=uint16\n"
            f"vocab_size=50257\n"
            f"total_tokens={total}\n"
            f"num_docs={n_chunks}\n"
            f"shard_size={total}\n"
            f"eot_token=50256\n"
            f"block_size={_BLOCK_SIZE}\n",
            encoding="utf-8",
        )
    # Sequential shuffle index: chunk order is 0, 1, 2, ... (deterministic).
    with open(data_dir / "train.bin.shuffle.idx", "wb") as f:
        np.save(f, np.arange(_N_TRAIN_CHUNKS, dtype=np.int64))
    return str(data_dir)


def _make_args(data_dir, out_dir, **overrides):
    """Fully populated args namespace for a tiny CPU PretrainTrainer."""
    defaults = dict(
        model="baseline",
        data_dir=data_dir,
        out_dir=out_dir,
        batch_size=4,
        block_size=_BLOCK_SIZE,
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_kv_head=None,
        position_encoding="rope",
        ffn_variant="swiglu",
        qk_norm=False,
        num_experts=1,
        dropout=0.0,
        learning_rate=1e-3,
        max_iters=8,
        warmup_iters=2,
        lr_decay_iters=8,
        min_lr=1e-4,
        lr_schedule="cosine",
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=8,
        eval_iters=2,
        log_interval=1000,
        ckpt_interval=0,
        seed=1337,
        device="cpu",
        compile=False,
        backend="nccl",
        fsdp=False,
        fsdp_sharding_strategy="full",
        num_workers=0,
        gradient_accumulation_steps=1,
        use_wandb=False,
        use_ema=False,
        ema_decay=0.999,
        early_stopping_patience=0,
        keep_last_n=0,
        use_packing=False,
        gradient_checkpointing=False,
        shuffle_buffer=None,
        resume=None,
        exp_id="tiny-resume-test",
        no_metrics=True,
        no_monitoring=True,
        config=None,
        attn_backend="auto",
        use_ring_attention=False,
        ring_block_size_q=16,
        ring_block_size_kv=16,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# 1. CheckpointManager.load restores RNG state
# ---------------------------------------------------------------------------

def test_checkpoint_load_restores_rng(tmp_path, monkeypatch):
    data_dir = _write_tiny_data(tmp_path)
    out_dir = str(Path(tmp_path) / "out")

    args_a = _make_args(data_dir, out_dir, max_iters=3, eval_interval=100)
    trainer_a = PretrainTrainer(args_a)
    try:
        trainer_a.train()
        # Save a checkpoint; rng_state is captured inside save.
        ckpt_path = trainer_a._save_checkpoint("rng_ckpt.pt")
        ckpt = _load_ckpt(ckpt_path)
        assert "rng_state" in ckpt
        saved_rng = ckpt["rng_state"]

        # Advance the RNG past the save point (a resumed run that fails to
        # restore would keep drawing from this advanced stream).
        torch.rand(3)

        # --- Behavioral: constructing a --resume trainer rewinds the torch
        # RNG to the save point (nothing after _maybe_resume consumes RNG).
        args_b = _make_args(
            data_dir, out_dir, max_iters=3, eval_interval=100,
            resume=str(ckpt_path),
        )
        trainer_b = PretrainTrainer(args_b)
        try:
            after = torch.rand(3)
            set_rng_state(saved_rng)
            expected = torch.rand(3)
            assert torch.equal(after, expected)
        finally:
            trainer_b.cleanup()

        # --- Call wiring: CheckpointManager.load calls set_rng_state with
        # the checkpoint's rng_state bundle.
        rng_calls = []

        def fake_set_rng_state(rng_state):
            rng_calls.append(rng_state)

        monkeypatch.setattr(
            checkpoint_mod, "set_rng_state", fake_set_rng_state
        )
        monkeypatch.setattr(
            "training.trainer_base.set_rng_state",
            fake_set_rng_state,
            raising=False,
        )

        args_c = _make_args(
            data_dir, out_dir, max_iters=3, eval_interval=100,
            resume=str(ckpt_path),
        )
        trainer_c = PretrainTrainer(args_c)
        try:
            assert len(rng_calls) >= 1
            received = rng_calls[-1]
            assert isinstance(received, dict)
            assert torch.equal(
                received["torch"].cpu().type(torch.ByteTensor),
                saved_rng["torch"].cpu().type(torch.ByteTensor),
            )
        finally:
            trainer_c.cleanup()
    finally:
        trainer_a.cleanup()


# ---------------------------------------------------------------------------
# 2. resume_offset advances during training and resumes the data position
# ---------------------------------------------------------------------------

def test_resume_offset_advances_and_resumes_data(tmp_path):
    data_dir = _write_tiny_data(tmp_path)
    out_dir = str(Path(tmp_path) / "out")

    # Run A: 8 iters x batch_size 4 -> 32 chunks consumed.
    args_a = _make_args(data_dir, out_dir, max_iters=8, eval_interval=8)
    trainer_a = PretrainTrainer(args_a)
    try:
        trainer_a.train()
        assert trainer_a.resume_offset == 32

        ckpt_path = Path(out_dir) / "latest_ckpt.pt"
        assert ckpt_path.is_file()
        ckpt = _load_ckpt(ckpt_path)
        assert ckpt.get("consumed_chunks", 0) > 0
        assert ckpt.get("consumed_chunks") == 32

        # Reference: the batch the uninterrupted run would consume next.
        ref_x, ref_y = next(trainer_a.train_iter)
    finally:
        trainer_a.cleanup()

    # Run B resumes from the checkpoint: its first batch must equal the
    # uninterrupted run's next batch (chunk 32..35, sequential shuffle idx).
    args_b = _make_args(
        data_dir, out_dir, max_iters=16, eval_interval=8,
        resume=str(ckpt_path),
    )
    trainer_b = PretrainTrainer(args_b)
    try:
        assert trainer_b.resume_offset == 32
        b_x, b_y = next(trainer_b.train_iter)
        assert torch.equal(b_x, ref_x)
        assert torch.equal(b_y, ref_y)
    finally:
        trainer_b.cleanup()


# ---------------------------------------------------------------------------
# 3. early_stop_counter save / restore / old-format default
# ---------------------------------------------------------------------------

def test_early_stop_counter_roundtrip(tmp_path):
    data_dir = _write_tiny_data(tmp_path)
    out_dir = str(Path(tmp_path) / "out")

    # Run A: two in-loop evals with a non-improving second val loss
    # (faked estimate_loss) -> early_stop_counter == 1 at the last save.
    args_a = _make_args(data_dir, out_dir, max_iters=9, eval_interval=4)
    trainer_a = PretrainTrainer(args_a)
    try:
        vals = iter([0.5, 0.6])

        def fake_estimate_loss():
            return {"train": 1.0, "val": next(vals, 0.7)}

        trainer_a.estimate_loss = fake_estimate_loss
        trainer_a.train()
        assert trainer_a.early_stop_counter == 1

        ckpt_path = Path(out_dir) / "latest_ckpt.pt"
        ckpt = _load_ckpt(ckpt_path)
        assert "early_stop_counter" in ckpt
        assert ckpt["early_stop_counter"] == 1
    finally:
        trainer_a.cleanup()

    # Resume from the new-format checkpoint: counter restored to 1.
    args_b = _make_args(
        data_dir, out_dir, max_iters=12, eval_interval=4,
        resume=str(ckpt_path),
    )
    trainer_b = PretrainTrainer(args_b)
    try:
        assert trainer_b.early_stop_counter == 1
    finally:
        trainer_b.cleanup()

    # Old-format checkpoint (key absent) -> counter defaults to 0, no error.
    config_dict = trainer_b.config.to_dict()
    config_dict["model_type"] = "baseline"
    old_path = Path(tmp_path) / "old_ckpt.pt"
    torch.save(
        {
            "model": trainer_b.raw_model.state_dict(),
            "optimizer": trainer_b.optimizer.state_dict(),
            "iter_num": 5,
            "best_val_loss": 0.5,
            "config": config_dict,
        },
        old_path,
    )
    args_c = _make_args(
        data_dir, out_dir, max_iters=12, eval_interval=4,
        resume=str(old_path),
    )
    trainer_c = PretrainTrainer(args_c)
    try:
        assert trainer_c.early_stop_counter == 0
    finally:
        trainer_c.cleanup()


# ---------------------------------------------------------------------------
# 4. --ckpt_interval: extra latest saves between eval points
# ---------------------------------------------------------------------------

def test_ckpt_interval_saves_more_often(tmp_path):
    data_dir = _write_tiny_data(tmp_path)
    out_dir = str(Path(tmp_path) / "out")

    # --- ckpt_interval=5: latest saves at iters 5 and 10 (off eval points).
    args = _make_args(
        data_dir, out_dir, max_iters=11, eval_interval=100, ckpt_interval=5
    )
    trainer = PretrainTrainer(args)
    try:
        saved = []

        def fake_save(filename):
            saved.append((filename, trainer.iter_num))
            return str(Path(out_dir) / filename)

        trainer._save_checkpoint = fake_save
        trainer.train()

        latest_iters = [it for fn, it in saved if fn == "latest_ckpt.pt"]
        assert 5 in latest_iters
        assert 10 in latest_iters
        assert len(latest_iters) >= 2
    finally:
        trainer.cleanup()

    # --- Default ckpt_interval=0: latest saves only at eval-multiple steps
    # (in-loop eval at 4, final block at 8 — both multiples of eval_interval).
    args2 = _make_args(
        data_dir, out_dir, max_iters=8, eval_interval=4, ckpt_interval=0
    )
    trainer2 = PretrainTrainer(args2)
    try:
        saved2 = []

        def fake_save2(filename):
            saved2.append((filename, trainer2.iter_num))
            return str(Path(out_dir) / filename)

        trainer2._save_checkpoint = fake_save2
        trainer2.train()

        latest_iters2 = [it for fn, it in saved2 if fn == "latest_ckpt.pt"]
        assert set(latest_iters2) == {4, 8}
    finally:
        trainer2.cleanup()


# ---------------------------------------------------------------------------
# 5. Old-format checkpoint: missing keys default without error
# ---------------------------------------------------------------------------

def test_old_checkpoint_missing_keys_defaults(tmp_path):
    data_dir = _write_tiny_data(tmp_path)
    out_dir = str(Path(tmp_path) / "out")

    # Reference trainer to build a matching model / optimizer / config.
    args_ref = _make_args(data_dir, out_dir, max_iters=1, eval_interval=100)
    ref = PretrainTrainer(args_ref)
    try:
        config_dict = ref.config.to_dict()
        config_dict["model_type"] = "baseline"
        old_path = Path(tmp_path) / "old_min_ckpt.pt"
        torch.save(
            {
                "model": ref.raw_model.state_dict(),
                "optimizer": ref.optimizer.state_dict(),
                "iter_num": 5,
                "best_val_loss": 2.5,
                "config": config_dict,
                # No rng_state / resume_offset / early_stop_counter /
                # consumed_chunks keys: the old minimal format.
            },
            old_path,
        )
    finally:
        ref.cleanup()

    args = _make_args(
        data_dir, out_dir, max_iters=6, eval_interval=100,
        resume=str(old_path),
    )
    trainer = PretrainTrainer(args)
    try:
        # Loading does not raise; every missing field defaults.
        assert trainer.iter_num == 5
        assert trainer.best_val_loss == 2.5
        assert trainer.resume_offset == 0
        assert trainer.early_stop_counter == 0

        extra = trainer.ckpt_manager.load(str(old_path))
        assert extra["iter_num"] == 5
        assert extra.get("rng_state") is None
        assert extra.get("resume_offset") is None
        assert extra.get("consumed_chunks") is None
        assert extra.get("early_stop_counter") is None
    finally:
        trainer.cleanup()
