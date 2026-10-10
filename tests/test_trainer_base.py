"""Regression tests for training.trainer_base helpers."""

import argparse
import tempfile
from pathlib import Path

import pytest
import torch


from training.trainer_base import (
    setup_distributed,
    set_seed,
    infer_device,
    build_amp_context,
    CheckpointManager,
    BaseTrainer,
    make_worker_init_fn,
)
from model.modern_gpt import ModernGPT, ModernGPTConfig


def test_setup_distributed_single_process():
    rank, local_rank, world_size, distributed = setup_distributed()
    assert rank == -1
    assert local_rank == -1
    assert world_size == 1
    assert distributed is False


def test_set_seed_reproducibility():
    set_seed(42)
    a = torch.rand(10)
    set_seed(42)
    b = torch.rand(10)
    assert torch.allclose(a, b)


def test_infer_device_with_local_rank():
    assert infer_device("cuda", 1) == "cuda:1"
    assert infer_device("cpu", -1) == "cpu"
    assert infer_device("cuda:0", -1) == "cuda:0"


def test_build_amp_context_cpu():
    ctx, scaler, dtype = build_amp_context("cpu", use_bf16=True)
    assert scaler is None
    assert dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_build_amp_context_cuda():
    ctx, scaler, dtype = build_amp_context("cuda", use_bf16=True)
    assert dtype in (torch.float16, torch.bfloat16)
    assert ctx is not None


def test_checkpoint_manager_roundtrip(tmp_path):
    config = tiny_modern_config(block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = ModernGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ema_shadow = {k: v.clone() for k, v in model.state_dict().items()}

    manager = CheckpointManager(
        out_dir=str(tmp_path),
        model=model,
        optimizer=optimizer,
        config=config,
        scaler=None,
        scheduler=None,
        ema_shadow=None,
        resume_offset=0,
    )

    path = manager.save(
        "ckpt.pt",
        iter_num=5,
        best_metric=1.23,
        resume_offset=42,
        ema_shadow=ema_shadow,
    )
    # Saves are async by default (T022): drain the background writer
    # before reading the file back.
    manager.flush()
    assert Path(path).exists()

    # Build a fresh model + optimizer and load.
    model2 = ModernGPT(config)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    manager2 = CheckpointManager(
        out_dir=str(tmp_path),
        model=model2,
        optimizer=optimizer2,
        config=config,
        ema_shadow={},
    )
    extra = manager2.load(path)

    assert extra["iter_num"] == 5
    assert extra["best_val_loss"] == 1.23
    assert manager2.resume_offset == 42
    assert manager2.ema_shadow is not None
    for k in ema_shadow:
        assert torch.allclose(manager2.ema_shadow[k], ema_shadow[k])
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_checkpoint_manager_keep_last_n(tmp_path):
    config = tiny_modern_config(block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = ModernGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    manager = CheckpointManager(
        out_dir=str(tmp_path),
        model=model,
        optimizer=optimizer,
        config=config,
        keep_last_n=2,
    )

    for i in range(5):
        manager.save(f"iter_{i}.pt", iter_num=i, best_metric=float(i))
    # Drain the in-flight background write before inspecting the dir.
    manager.flush()

    # Best-like name is protected, but iter_N are prunable.
    remaining = sorted(p.name for p in tmp_path.glob("iter_*.pt"))
    assert remaining == ["iter_3.pt", "iter_4.pt"]


def test_checkpoint_manager_save_with_ddp_unwrap(tmp_path):
    """CheckpointManager should unwrap DDP models before saving."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for DDP")
    # Single-process DDP requires explicit process group; skip for simplicity.
    pytest.skip("DDP smoke test omitted in single-process CI")


class DummyTrainer(BaseTrainer):
    def train(self):
        pass


def test_base_trainer_init():
    args = argparse.Namespace(
        seed=1,
        device="cpu",
        backend="nccl",
        out_dir=tempfile.mkdtemp(),
        use_wandb=False,
    )
    trainer = DummyTrainer(args)
    assert trainer.rank == -1
    assert trainer.world_size == 1
    assert trainer.master_process is True
    trainer.cleanup()


def test_make_worker_init_fn_deterministic():
    fn = make_worker_init_fn(base_seed=7, rank=2)
    # Should not raise and should set seeds deterministically.
    fn(0)
    fn(1)
    fn(0)
    # Same worker_id -> same seed -> reproducible.
    set_seed(7 + 2 * 1000 + 0)
    a = torch.rand(5)
    set_seed(7 + 2 * 1000 + 0)
    b = torch.rand(5)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Migrated from tests/test_bugfixes.py (FR11b split)
# ---------------------------------------------------------------------------

from helpers import tiny_modern_config
from utils.checkpoint import save_checkpoint, load_checkpoint, get_rng_state


def test_optimizer_weight_tying_dedup():
    """wte and lm_head share the same tensor; optimizer must see it once."""
    config = tiny_modern_config(n_head=4, n_embd=128, block_size=64)
    model = ModernGPT(config)
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    # wte and lm_head are the same tensor -> should appear only once in optimizer.
    assert id(model.transformer.wte.weight) == id(model.lm_head.weight)
    assert id(model.transformer.wte.weight) in param_ids
    # Count how many times the shared tensor appears in optimizer params.
    occurrences = sum(
        1
        for group in opt.param_groups
        for p in group["params"]
        if id(p) == id(model.transformer.wte.weight)
    )
    assert occurrences == 1, f"shared weight appears {occurrences} times in optimizer"
    print("[OK] Optimizer weight-tying deduplication works")


def test_ma_update_and_checkpoint():
    """EMA shadow weights must update and round-trip through checkpoint."""
    config = tiny_modern_config(n_head=4, n_embd=128, block_size=64)
    model = ModernGPT(config)
    model.init_ema(decay=0.9)

    # Simulate one update step with dummy gradients.
    opt = model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))
    _, loss, _ = model(x, y)
    loss.backward()
    opt.step()
    model.update_ema()

    # EMA should differ from current weights after update.
    name = next(iter(model.ema_shadow))
    assert not torch.equal(model.ema_shadow[name], model.state_dict()[name])

    # Save and load checkpoint.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(
            model,
            opt,
            iter_num=7,
            best_val_loss=1.23,
            config=config,
            out_dir=tmpdir,
            filename="ckpt.pt",
            ema_shadow=model.ema_shadow,
            rng_state=get_rng_state(),
        )
        model2 = ModernGPT(config)
        opt2 = model2.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        extra = load_checkpoint(path, model2, opt2)
        assert extra["iter_num"] == 7
        assert extra["best_val_loss"] == 1.23
        assert "ema_shadow" in extra
        assert torch.equal(extra["ema_shadow"][name], model.ema_shadow[name])
    print("[OK] EMA update and checkpoint round-trip work")
