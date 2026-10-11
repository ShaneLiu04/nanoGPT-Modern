"""RED-phase tests for T010 (AR-v0.5.0-hpc-readiness, FR9).

Pins the rank-aware data sharding + DDP tuning contracts that are NOT
implemented yet (design.md L277-285 shard/offset semantics, L319 DDP
construction, L482 test spec, L527 B3 boundary; srs.md L210-220 FR9):

* ``MemmapDataset`` gains ``shard_rank`` / ``shard_world_size``: rank r
  of W consumes the STRIDED virtual stream at global positions
  r, r+W, r+2W, ... of the global chunk order (``shuffle_idx[r::W]``
  when a shuffle index exists, chunk ids r, r+W, ... otherwise).
  ``resume_offset`` counts consumed items within the rank's OWN stream
  (local index), not global position.
* ``_maybe_resume`` (train_pretrain): world_size mismatch -> WARN (the
  WARN exists today) + RESET resume_offset to the rank shard head (the
  reset does NOT exist yet - the RED failure for this class).
* DDP construction must pass ``static_graph=True`` and
  ``gradient_as_bucket_view=True`` (trainer_base.wrap_distributed).
* ``PretrainTrainer.train`` must wrap non-final micro-steps of each
  accumulation window in ``model.no_sync()`` when distributed.
* ``_build_data`` shards ONLY the train loader; val / train_eval
  loaders stay unsharded (every rank evaluates the same val data).

The tiny-data / args-namespace helper pattern is a self-contained copy
of the T009/T022 pattern in tests/test_training_perf_flags.py.
"""

import argparse
import contextlib
import inspect
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed

# ---------------------------------------------------------------------------
# Local helpers (copies of the T009/T022 pattern; kept local to this file)
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 32
_T010_N_TRAIN_CHUNKS = 40
_T010_N_VAL_CHUNKS = 8

# Shard-identifiable data: 12 chunks, every token of chunk i == i (uint16),
# so a consumed sample's chunk id is simply x[0].item().
_SHARD_N_CHUNKS = 12
_SHARD_SEED = 20260101


def _write_tiny_data(tmp_path):
    """Tiny train/val .bin + .idx + sequential train shuffle idx.

    Self-contained copy of the ``_write_tiny_data`` pattern from
    tests/test_training_perf_flags.py (block_size=32, train 40 chunks,
    val 8 chunks, sequential shuffle idx).
    """
    data_dir = Path(tmp_path) / "data"
    # Skip the rewrite when the files already exist: a second trainer
    # built in the same test keeps the first one's np.memmap handle
    # open, and rewriting train.bin under it fails on Windows
    # (file lock).  The content is deterministic, so reuse is safe.
    if (data_dir / "train.bin").exists():
        return str(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, n_chunks in (
        ("train", _T010_N_TRAIN_CHUNKS),
        ("val", _T010_N_VAL_CHUNKS),
    ):
        total = n_chunks * (_BLOCK_SIZE + 1)
        # Deterministic token pattern, all ids < 50257 and uint16-safe.
        tokens = (np.arange(total, dtype=np.int64) * 7 + 3) % 50000
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
        np.save(f, np.arange(_T010_N_TRAIN_CHUNKS, dtype=np.int64))
    return str(data_dir)


def _write_shard_data(tmp_path, with_shuffle=True):
    """Identifiable shard data: 12 chunks, all tokens of chunk i == i.

    Returns ``(data_dir, bin_path)``.  When *with_shuffle* a fixed-seed
    permutation (np.random.default_rng) is written as
    ``train.bin.shuffle.idx``; otherwise no shuffle index exists (the
    unshuffled-stride case).
    """
    sub = "shard_data" if with_shuffle else "shard_noshuffle"
    data_dir = Path(tmp_path) / sub
    data_dir.mkdir(parents=True, exist_ok=True)
    n = _SHARD_N_CHUNKS
    total = n * (_BLOCK_SIZE + 1)
    tokens = np.repeat(np.arange(n, dtype=np.int64), _BLOCK_SIZE + 1)
    tokens.astype(np.uint16).tofile(data_dir / "train.bin")
    (data_dir / "train.idx").write_text(
        f"dtype=uint16\n"
        f"vocab_size=50257\n"
        f"total_tokens={total}\n"
        f"num_docs={n}\n"
        f"shard_size={total}\n"
        f"eot_token=50256\n"
        f"block_size={_BLOCK_SIZE}\n",
        encoding="utf-8",
    )
    if with_shuffle:
        rng = np.random.default_rng(_SHARD_SEED)
        with open(data_dir / "train.bin.shuffle.idx", "wb") as f:
            np.save(f, rng.permutation(n))
    return str(data_dir), str(data_dir / "train.bin")


def _make_args(data_dir, out_dir, **overrides):
    """Fully populated args namespace for a tiny CPU PretrainTrainer.

    Field pattern copied from tests/test_training_perf_flags.py
    ``_make_args`` (deliberately without ``eval_batch_size``).
    """
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
        max_iters=2,
        warmup_iters=2,
        lr_decay_iters=2,
        min_lr=1e-4,
        lr_schedule="cosine",
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=1000,
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
        exp_id="t010-ddp-shard",
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


def _make_pretrain_trainer(tmp_path, out_name="out", **overrides):
    """Build a tiny CPU PretrainTrainer (construction only, no train())."""
    from training.train_pretrain import PretrainTrainer

    data_dir = _write_tiny_data(tmp_path)
    out_dir = str(Path(tmp_path) / out_name)
    args = _make_args(data_dir, out_dir, **overrides)
    return args, PretrainTrainer(args)


def _shard_dataset(bin_path, shuffle_idx_path, rank, world, resume_offset=0, shuffle_buffer=None):
    """MemmapDataset with the new shard params (T010 contract)."""
    from data.openwebtext import MemmapDataset

    return MemmapDataset(
        bin_path,
        block_size=_BLOCK_SIZE,
        resume_offset=resume_offset,
        shuffle_buffer=shuffle_buffer,
        shuffle_idx_path=shuffle_idx_path,
        shard_rank=rank,
        shard_world_size=world,
    )


def _consume_samples(dataset):
    """Fully consume a dataset, collecting the x tensors."""
    return [x for x, _ in dataset]


def _chunk_ids(samples):
    """Chunk id of each consumed sample (all tokens of chunk i == i)."""
    return [int(s[0].item()) for s in samples]


class _NoSyncRecorder:
    """Fake no_sync context manager factory attached to a model.

    Records context enters / exits so the train loop's no_sync usage on
    non-final micro-steps can be counted without a real DDP wrapper.
    """

    def __init__(self):
        self.enters = 0
        self.exits = 0

    @contextlib.contextmanager
    def no_sync(self):
        self.enters += 1
        try:
            yield
        finally:
            self.exits += 1


# ---------------------------------------------------------------------------
# A. MemmapDataset rank sharding
# ---------------------------------------------------------------------------


class TestMemmapSharding:
    """MemmapDataset shard_rank / shard_world_size contracts (FR9).

    Rank r of W consumes the strided virtual stream r, r+W, r+2W, ...;
    the union over ranks covers every chunk exactly once (mutual
    exclusion).  resume_offset is a LOCAL index into the rank's own
    stream.
    """

    def test_shard_mutual_exclusion_with_shuffle(self, tmp_path):
        """W=3 over 12 chunks with a fixed-seed shuffle.idx.

        The three rank streams are pairwise disjoint and their union is
        exactly the 12 chunk ids (each rank gets 4 items).
        """
        data_dir, bin_path = _write_shard_data(tmp_path, with_shuffle=True)
        shuffle_path = os.path.join(data_dir, "train.bin.shuffle.idx")

        sets = []
        for r in range(3):
            ds = _shard_dataset(bin_path, shuffle_path, r, 3)
            ids = _chunk_ids(_consume_samples(ds))
            assert len(ids) == 4  # 12 chunks / 3 ranks, divisible case
            sets.append(set(ids))

        for a, b in ((0, 1), (0, 2), (1, 2)):
            assert sets[a].isdisjoint(sets[b]), (a, b)
        union = sets[0] | sets[1] | sets[2]
        assert union == set(range(_SHARD_N_CHUNKS))

    def test_shard_union_covers_all_non_divisible(self, tmp_path):
        """W=5 over 12 chunks (non-divisible): ranks get 3,3,2,2,2 items.

        Union still == all 12 chunks, pairwise disjoint.
        """
        data_dir, bin_path = _write_shard_data(tmp_path, with_shuffle=True)
        shuffle_path = os.path.join(data_dir, "train.bin.shuffle.idx")

        sets = []
        counts = []
        for r in range(5):
            ds = _shard_dataset(bin_path, shuffle_path, r, 5)
            ids = _chunk_ids(_consume_samples(ds))
            counts.append(len(ids))
            sets.append(set(ids))

        assert counts == [3, 3, 2, 2, 2]  # strided r, r+5, r+10, ...
        for i in range(5):
            for j in range(i + 1, 5):
                assert sets[i].isdisjoint(sets[j]), (i, j)
        union = set()
        for s in sets:
            union |= s
        assert union == set(range(_SHARD_N_CHUNKS))

    def test_shard_without_shuffle_idx(self, tmp_path):
        """No shuffle.idx: chunk ids r, r+W, ... (stride in natural order).

        shuffle_buffer=1 keeps the buffer-shuffle path a no-op so the
        unshuffled order (and therefore the visible stride) is
        deterministic regardless of the global random state.
        """
        data_dir, bin_path = _write_shard_data(tmp_path, with_shuffle=False)

        sets = []
        for r in range(3):
            ds = _shard_dataset(bin_path, None, r, 3, shuffle_buffer=1)
            ids = _chunk_ids(_consume_samples(ds))
            assert ids[0] == r  # stride visible: first sample of rank r
            sets.append(set(ids))

        for a, b in ((0, 1), (0, 2), (1, 2)):
            assert sets[a].isdisjoint(sets[b]), (a, b)
        union = sets[0] | sets[1] | sets[2]
        assert union == set(range(_SHARD_N_CHUNKS))

    def test_resume_offset_is_rank_local(self, tmp_path):
        """resume_offset counts items in the RANK'S OWN stream.

        Self-consistent semantics: a rank-1 dataset with offset k yields
        first the k-th item (0-based) of the same rank's offset-0
        stream.
        """
        data_dir, bin_path = _write_shard_data(tmp_path, with_shuffle=True)
        shuffle_path = os.path.join(data_dir, "train.bin.shuffle.idx")

        full = _chunk_ids(_consume_samples(_shard_dataset(bin_path, shuffle_path, 1, 3)))
        k = 2
        offset_ids = _chunk_ids(
            _consume_samples(_shard_dataset(bin_path, shuffle_path, 1, 3, resume_offset=k))
        )
        assert offset_ids[0] == full[k]

    def test_world_size_1_degradation_b3(self, tmp_path):
        """B3: shard_world_size=1 degrades to the unsharded stream.

        Default construction (no shard params) and shard_rank=0 /
        shard_world_size=1 must yield the identical sample sequence
        (order + content), and identical resume behaviour (offset 2
        skips the same items).
        """
        from data.openwebtext import MemmapDataset

        data_dir, bin_path = _write_shard_data(tmp_path, with_shuffle=True)
        shuffle_path = os.path.join(data_dir, "train.bin.shuffle.idx")

        default_ds = MemmapDataset(
            bin_path,
            block_size=_BLOCK_SIZE,
            resume_offset=0,
            shuffle_idx_path=shuffle_path,
        )
        shard_ds = _shard_dataset(bin_path, shuffle_path, 0, 1)
        default_samples = _consume_samples(default_ds)
        shard_samples = _consume_samples(shard_ds)
        assert _chunk_ids(default_samples) == _chunk_ids(shard_samples)
        for a, b in zip(default_samples, shard_samples):
            assert torch.equal(a, b)
            assert a.shape[0] == _BLOCK_SIZE

        # Identical resume behaviour at the same local offset.
        default_off = _consume_samples(
            MemmapDataset(
                bin_path,
                block_size=_BLOCK_SIZE,
                resume_offset=2,
                shuffle_idx_path=shuffle_path,
            )
        )
        shard_off = _consume_samples(_shard_dataset(bin_path, shuffle_path, 0, 1, resume_offset=2))
        assert _chunk_ids(default_off) == _chunk_ids(shard_off)
        for a, b in zip(default_off, shard_off):
            assert torch.equal(a, b)

    def test_shard_params_thread_through(self, tmp_path):
        """get_dataloader / get_openwebtext_dataset pass shard params."""
        from data.openwebtext import get_dataloader, get_openwebtext_dataset

        data_dir, _ = _write_shard_data(tmp_path, with_shuffle=True)

        loader = get_dataloader(
            data_dir,
            "train",
            batch_size=4,
            block_size=_BLOCK_SIZE,
            num_workers=0,
            shard_rank=1,
            shard_world_size=3,
        )
        assert loader.dataset.shard_rank == 1
        assert loader.dataset.shard_world_size == 3

        ds = get_openwebtext_dataset(
            data_dir,
            "train",
            block_size=_BLOCK_SIZE,
            shard_rank=1,
            shard_world_size=3,
        )
        assert ds.shard_rank == 1
        assert ds.shard_world_size == 3


# ---------------------------------------------------------------------------
# B. _maybe_resume: world_size mismatch -> WARN + reset to shard head
# ---------------------------------------------------------------------------


class TestResumeShardConsistency:
    """Resume/checkpoint consistency of the sharded data stream.

    ``_maybe_resume`` runs inside ``BaseTrainer.__init__`` (template
    method, trainer_base.py), so the world_size of the resuming run is
    fixed at construction time.  The mismatch scenario therefore uses
    the ``WORLD_SIZE`` env var (setup_distributed reads RANK/WORLD_SIZE;
    WORLD_SIZE=2 without RANK gives world_size=2, rank=-1,
    distributed=False) rather than post-construction monkeypatching.
    """

    def test_resume_world_size_mismatch_warns_and_resets(self, tmp_path, monkeypatch):
        """Saved world_size=1, resumed with world_size=2.

        Contract: RuntimeWarning mentioning world_size AND
        resume_offset reset to the shard head (0) - trainer and rebuilt
        train_loader dataset alike.  The WARN already exists; the RESET
        is the RED failure.
        """
        from training.train_pretrain import PretrainTrainer

        for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            monkeypatch.delenv(var, raising=False)

        args_a, trainer_a = _make_pretrain_trainer(
            tmp_path,
            "out_a",
            max_iters=2,
            eval_interval=100,
            batch_size=2,
        )
        trainer_b = None
        try:
            trainer_a.train()
            saved_offset = trainer_a.resume_offset
            assert saved_offset > 0  # 2 iters x batch_size 2

            # Async save (T022 contract): flush before reading the file.
            ckpt_path = trainer_a._save_checkpoint("mismatch.pt")
            trainer_a.ckpt_manager.flush()
            assert os.path.exists(ckpt_path)

            # world_size=2 without RANK -> rank=-1, distributed=False,
            # but max(world_size, 1) == 2 != saved 1 -> mismatch branch.
            monkeypatch.setenv("WORLD_SIZE", "2")
            args_b = _make_args(
                args_a.data_dir,
                str(Path(tmp_path) / "out_b"),
                resume=ckpt_path,
                max_iters=2,
                eval_interval=100,
                batch_size=2,
            )
            with pytest.warns(RuntimeWarning, match="world_size"):
                trainer_b = PretrainTrainer(args_b)

            # The reset-to-shard-head contract (missing today: RED).
            assert trainer_b.resume_offset == 0
            assert trainer_b.train_loader.dataset.resume_offset == 0
        finally:
            trainer_a.cleanup()
            if trainer_b is not None:
                trainer_b.cleanup()

    def test_resume_world_size_match_keeps_offset(self, tmp_path, monkeypatch):
        """Matching world_size: offset preserved, no mismatch warning.

        Forward pin / regression guard: the current non-distributed
        resume path already satisfies this; it must keep passing after
        the sharding change (offset semantics are unchanged when
        world_size matches the saved value).
        """
        from training.train_pretrain import PretrainTrainer

        for var in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
            monkeypatch.delenv(var, raising=False)

        args_a, trainer_a = _make_pretrain_trainer(
            tmp_path,
            "out_a",
            max_iters=2,
            eval_interval=100,
            batch_size=2,
        )
        trainer_c = None
        try:
            trainer_a.train()
            saved_offset = trainer_a.resume_offset
            # 2 iterations x batch_size 2, accum=1 -> 4 consumed chunks.
            assert saved_offset == 4

            ckpt_path = trainer_a._save_checkpoint("match.pt")
            trainer_a.ckpt_manager.flush()
            assert os.path.exists(ckpt_path)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                args_c = _make_args(
                    args_a.data_dir,
                    str(Path(tmp_path) / "out_c"),
                    resume=ckpt_path,
                    max_iters=2,
                    eval_interval=100,
                    batch_size=2,
                )
                trainer_c = PretrainTrainer(args_c)

            mismatch = [
                w
                for w in caught
                if issubclass(w.category, RuntimeWarning) and "world_size" in str(w.message)
            ]
            assert not mismatch
            assert trainer_c.resume_offset == saved_offset
            assert trainer_c.train_loader.dataset.resume_offset == saved_offset
        finally:
            trainer_a.cleanup()
            if trainer_c is not None:
                trainer_c.cleanup()


# ---------------------------------------------------------------------------
# C. DDP construction flags (static)
# ---------------------------------------------------------------------------


class TestDDPConstruction:
    """DDP wrap flags in BaseTrainer.wrap_distributed (design.md L319)."""

    def test_ddp_construction_flags(self):
        """DDP(model, ...) must pass static_graph=True and
        gradient_as_bucket_view=True (structure-fixed models)."""
        from training.trainer_base import BaseTrainer

        src = inspect.getsource(BaseTrainer.wrap_distributed)
        assert "static_graph=True" in src
        assert "gradient_as_bucket_view=True" in src


# ---------------------------------------------------------------------------
# D. no_sync on non-final micro-steps
# ---------------------------------------------------------------------------


class TestNoSyncAccum:
    """no_sync wrapping of non-final micro-steps (FR9, pseudo-distributed).

    Same run shape as the T022 TestMicroStepDesync tests: max_iters=8
    with accum=2 means 8 micro-steps / 4 accumulation windows; the
    non-final micro-steps (0, 2, 4, 6) are the 4 that must run inside
    model.no_sync() when distributed.
    """

    RUN = dict(
        max_iters=8,
        gradient_accumulation_steps=2,
        eval_interval=100,  # > max_iters: no mid-train eval
        log_interval=100,
        grad_clip=0,
        use_ema=False,
        batch_size=2,
    )

    @staticmethod
    def _stub_all_reduce(monkeypatch):
        def fake_all_reduce(tensor, op=None):
            pass

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    def test_no_sync_wraps_nonfinal_microsteps(self, tmp_path, monkeypatch):
        """Distributed + accum=2: no_sync entered/exited exactly 4 times.

        One no_sync context per non-final micro-step of each of the 4
        windows; every context must close (no leaked contexts).
        """
        args, trainer = _make_pretrain_trainer(tmp_path, "out", **self.RUN)
        rec = _NoSyncRecorder()
        try:
            monkeypatch.setattr(trainer, "distributed", True)
            monkeypatch.setattr(trainer, "world_size", 1)
            self._stub_all_reduce(monkeypatch)
            # nn.Module stores a plain bound method in __dict__; the
            # train loop's hasattr/attribute lookup sees it.
            monkeypatch.setattr(trainer.model, "no_sync", rec.no_sync, raising=False)

            trainer.train()

            assert rec.enters == 4  # micro-steps 0, 2, 4, 6
            assert rec.exits == 4  # no context leaks
        finally:
            trainer.cleanup()

    def test_no_sync_absent_when_accum_1_or_nondistributed(self, tmp_path, monkeypatch):
        """Forward pin: no no_sync when accum=1 or when not distributed.

        accum=1 has no non-final micro-step; accum=2 without
        distributed must not touch no_sync at all (single-card runs are
        unchanged, B3).  Passes in RED by design (no no_sync call exists
        yet anywhere) and must keep passing after the change.
        """
        # accum=1, distributed=True
        args, t1 = _make_pretrain_trainer(
            tmp_path,
            "out_accum1",
            max_iters=4,
            gradient_accumulation_steps=1,
            eval_interval=100,
            log_interval=100,
            grad_clip=0,
            use_ema=False,
            batch_size=2,
        )
        rec1 = _NoSyncRecorder()
        try:
            monkeypatch.setattr(t1, "distributed", True)
            monkeypatch.setattr(t1, "world_size", 1)
            self._stub_all_reduce(monkeypatch)
            monkeypatch.setattr(t1.model, "no_sync", rec1.no_sync, raising=False)
            t1.train()
            assert rec1.enters == 0
        finally:
            t1.cleanup()

        # accum=2, NOT distributed
        args, t2 = _make_pretrain_trainer(
            tmp_path,
            "out_nondist",
            max_iters=4,
            gradient_accumulation_steps=2,
            eval_interval=100,
            log_interval=100,
            grad_clip=0,
            use_ema=False,
            batch_size=2,
        )
        rec2 = _NoSyncRecorder()
        try:
            monkeypatch.setattr(t2.model, "no_sync", rec2.no_sync, raising=False)
            t2.train()
            assert rec2.enters == 0
        finally:
            t2.cleanup()

    def test_no_sync_in_train_source(self):
        """Static: PretrainTrainer.train must reference no_sync."""
        from training.train_pretrain import PretrainTrainer

        src = inspect.getsource(PretrainTrainer.train)
        assert "no_sync" in src


# ---------------------------------------------------------------------------
# E. _build_data shard wiring (train split only)
# ---------------------------------------------------------------------------


class TestBuildDataSharding:
    """_build_data wiring: only the train loader is sharded.

    val / train_eval loaders stay unsharded so every rank evaluates the
    same data (estimate_loss SUM / world_size remains a correct mean).
    """

    def test_build_data_shards_train_split_only(self, tmp_path, monkeypatch):
        """Distributed rank 1 of 3: train loader sharded, eval loaders not.

        _build_data is re-callable, so distributed/rank/world_size are
        monkeypatched AFTER construction and the builders re-run.
        """
        args, trainer = _make_pretrain_trainer(tmp_path, "out")
        try:
            monkeypatch.setattr(trainer, "distributed", True)
            monkeypatch.setattr(trainer, "rank", 1)
            monkeypatch.setattr(trainer, "world_size", 3)

            trainer._build_data()

            assert trainer.train_loader.dataset.shard_rank == 1
            assert trainer.train_loader.dataset.shard_world_size == 3
            # Eval loaders are unsharded (every rank sees the same data).
            assert trainer.val_loader.dataset.shard_world_size == 1
            assert trainer.train_eval_loader.dataset.shard_world_size == 1
        finally:
            trainer.cleanup()

    def test_build_data_nondistributed_defaults(self, tmp_path):
        """Non-distributed construction: shard defaults (0, 1)."""
        args, trainer = _make_pretrain_trainer(tmp_path, "out")
        try:
            assert trainer.train_loader.dataset.shard_rank == 0
            assert trainer.train_loader.dataset.shard_world_size == 1
        finally:
            trainer.cleanup()


# ---------------------------------------------------------------------------
# F. Real gloo process group through the full train loop (GPU-rehearsal
#    finding, 2026-10-11: ReduceOp.AVG is NCCL-only)
# ---------------------------------------------------------------------------


def test_distributed_train_real_gloo_loss_all_reduce(tmp_path, monkeypatch):
    """Full train() over a real single-process gloo group must work.

    GPU-rehearsal finding (Quadro RTX 5000, Windows): the loss
    all-reduce used ``ReduceOp.AVG``, which gloo rejects
    (``Cannot use ReduceOp.AVG with Gloo``) -- NCCL-only semantics.
    Every existing distributed test either stubs all_reduce or stops at
    construction, so this was never exercised on a real group.

    Contract: the loss all-reduce is backend-generic (SUM + explicit
    world_size division, numerically identical to AVG); a single-rank
    gloo group through ``PretrainTrainer.train`` completes both the
    train loop and the eval path (estimate_loss uses the same
    all-reduce family).

    Skips (like the FSDP group tests) if a single-process group cannot
    be created on this machine.
    """
    import torch.distributed as dist

    from training.trainer_base import cleanup_distributed

    monkeypatch.setenv("USE_LIBUV", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29514")

    try:
        args, trainer = _make_pretrain_trainer(
            tmp_path,
            "out_gloo_train",
            backend="gloo",
            max_iters=2,
            eval_interval=2,
            eval_iters=2,
            log_interval=1,
            ckpt_interval=0,
        )
    except Exception as exc:
        if not dist.is_initialized():
            pytest.skip(f"single-process gloo group init failed here: {exc!r}")
        cleanup_distributed()
        raise
    try:
        assert trainer.distributed and dist.is_initialized()
        assert dist.get_backend() == "gloo"
        trainer.train()
    finally:
        trainer.cleanup()
        cleanup_distributed()
