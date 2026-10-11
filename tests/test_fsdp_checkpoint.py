"""RED-phase tests for T011 (AR-v0.5.0-hpc-readiness, FR9b).

Pins the FSDP construction + checkpoint-hook contracts that are NOT
implemented yet (design.md L285-288 FSDP impl, L317-320 interface
table, L483 this file's spec, L550 E7; srs.md L219 FR9b AC):

* ``BaseTrainer.wrap_distributed`` FSDP branch must construct
  ``FSDP(model, ..., auto_wrap_policy=transformer_auto_wrap_policy(Block
  classes), use_orig_params=True)``, remember the wrapper as
  ``self._fsdp_module``, and only then ``torch.compile`` (FSDP-first,
  compile outermost -- FSDP1 has no ``_orig_mod`` handling).
* ``BaseTrainer`` gains generalized ``_get_model_state_dict`` /
  ``_load_model_state_dict`` hooks (moved up from PretrainTrainer, which
  currently keeps them dead code); ``configure_checkpointing`` wires
  them into ``CheckpointManager`` via ``state_dict_fn`` /
  ``load_state_fn`` so FSDP FULL_STATE_DICT save/load actually runs.
* ``CheckpointManager.save`` uses the ``state_dict_fn`` payload hook
  (plus ``_orig_mod`` unwrap for model_type detection); ``load``
  delegates to ``load_state_fn``.
* E7: a failed save must never corrupt the previous checkpoint
  (tmp+rename atomicity, WARN, no leftover ``*.tmp``).

Real-FSDP construction tests run single-process gloo
(``USE_LIBUV=0`` before ``init_process_group``) and are skipped on
machines without an accelerator: FSDP1 requires a non-CPU device
handle even when the process group is gloo.  FSDP1 FULL_STATE_DICT
gathering is CUDA-only, so state-dict tests use stubs -- never a real
``FULL_STATE_DICT`` gather.

The tiny-data / args-namespace helper pattern is a self-contained copy
of the T009/T010/T022 pattern in tests/test_training_perf_flags.py /
tests/test_ddp_shard.py (block_size=32).
"""

import argparse
import contextlib
import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

# ---------------------------------------------------------------------------
# Local helpers (copies of the T009/T010/T022 pattern; kept local to this
# file)
# ---------------------------------------------------------------------------

_BLOCK_SIZE = 32
_T011_N_TRAIN_CHUNKS = 40
_T011_N_VAL_CHUNKS = 8


def _write_tiny_data(tmp_path):
    """Tiny train/val .bin + .idx + sequential train shuffle idx.

    Self-contained copy of the ``_write_tiny_data`` pattern from
    tests/test_ddp_shard.py (block_size=32, train 40 chunks, val 8
    chunks, sequential shuffle idx).
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
        ("train", _T011_N_TRAIN_CHUNKS),
        ("val", _T011_N_VAL_CHUNKS),
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
        np.save(f, np.arange(_T011_N_TRAIN_CHUNKS, dtype=np.int64))
    return str(data_dir)


def _make_args(data_dir, out_dir, **overrides):
    """Fully populated args namespace for a tiny CPU PretrainTrainer.

    Field pattern copied from tests/test_ddp_shard.py ``_make_args``.
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
        exp_id="t011-fsdp-ckpt",
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


def _make_fsdp_trainer(tmp_path, monkeypatch, port, out_name="out", **overrides):
    """Build a tiny single-process gloo PretrainTrainer with fsdp=True.

    FSDP1 constructs a device handle for a non-CPU accelerator even
    when the process group is gloo, so on machines without one the
    caller tests are skipped (real-FSDP contracts are only meaningful
    where FSDP can actually be constructed).

    ``USE_LIBUV=0`` must be in the environment BEFORE
    ``init_process_group`` (gloo on this torch build lacks libuv
    support otherwise).  If the process group itself cannot be created
    on this machine, the caller tests are skipped: real-FSDP contracts
    are only meaningful where a (single-process) group can exist.
    Construction failures BEYOND group init are re-raised (after
    destroying the leaked group) so they surface as real failures.
    """
    if not torch.cuda.is_available():
        pytest.skip(
            "single-process FSDP construction needs a non-CPU "
            "accelerator device (no CUDA on this machine)"
        )
    from training.train_pretrain import PretrainTrainer
    from training.trainer_base import cleanup_distributed

    monkeypatch.setenv("USE_LIBUV", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(port))

    data_dir = _write_tiny_data(tmp_path)
    out_dir = str(Path(tmp_path) / out_name)
    args = _make_args(
        data_dir,
        out_dir,
        backend="gloo",
        fsdp=True,
        fsdp_sharding_strategy="full",
        compile=False,
        **overrides,
    )
    try:
        trainer = PretrainTrainer(args)
    except Exception as exc:
        if not dist.is_initialized():
            pytest.skip(
                f"single-process gloo process group init failed on this "
                f"machine (real-FSDP tests are CPU-infeasible here): {exc!r}"
            )
        cleanup_distributed()
        raise
    return args, trainer


def _reference_baseline_gpt(args):
    """Plain unwrapped BaselineGPT with the trainer's exact config.

    Same config fields / seed as ``PretrainTrainer._build_model`` for
    ``model="baseline"``; used as the parameter-name reference for the
    ``use_orig_params`` contract.
    """
    from model.baseline_gpt import BaselineGPT, BaselineGPTConfig

    torch.manual_seed(args.seed)
    config = BaselineGPTConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        attention_backend="sdpa",
        gradient_checkpointing=args.gradient_checkpointing,
    )
    return BaselineGPT(config)


def _recording_state_dict_type(monkeypatch, recorded):
    """Monkeypatch FSDP.state_dict_type with a recording contextmanager.

    Records ``(module, args, kwargs)`` per enter and yields without
    touching anything (the real FULL_STATE_DICT gather is CUDA-only;
    tests must never trigger it on CPU).
    """

    from torch.distributed.fsdp import FullyShardedDataParallel

    @contextlib.contextmanager
    def fake_state_dict_type(module, *sd_args, **sd_kwargs):
        recorded.append((module, sd_args, sd_kwargs))
        yield

    monkeypatch.setattr(
        FullyShardedDataParallel,
        "state_dict_type",
        staticmethod(fake_state_dict_type),
    )


# ---------------------------------------------------------------------------
# A. TestFSDPConstruction (real FSDP, gloo, CPU)
# ---------------------------------------------------------------------------


class TestFSDPConstruction:
    """Real single-process FSDP construction contracts (FR9b).

    design.md L286-287: auto_wrap_policy per Block, use_orig_params=True;
    L287/L320: FSDP wrap first, torch.compile after (compile outermost).
    """

    def test_fsdp_wrap_sets_fsdp_module_and_raw_model(self, tmp_path, monkeypatch):
        """wrap_distributed must remember the FSDP wrapper.

        ``trainer._fsdp_module`` is the FullyShardedDataParallel root,
        ``trainer.raw_model is trainer._fsdp_module.module`` and is the
        plain GPT class (ModernGPT / BaselineGPT).
        """
        from torch.distributed.fsdp import FullyShardedDataParallel

        args, trainer = _make_fsdp_trainer(tmp_path, monkeypatch, port=29511)
        try:
            assert isinstance(trainer._fsdp_module, FullyShardedDataParallel)
            assert trainer.raw_model is trainer._fsdp_module.module
            assert type(trainer.raw_model).__name__ in {
                "ModernGPT",
                "BaselineGPT",
            }
        finally:
            trainer.cleanup()

    def test_use_orig_params_preserves_param_names(self, tmp_path, monkeypatch):
        """use_orig_params=True: original parameter slots survive wrapping.

        torch 2.5.1 FSDP1 keeps ``_fsdp_wrapped_module.`` prefixes in
        ``named_parameters`` keys even with use_orig_params=True (name
        stripping happens in FULL_STATE_DICT output, which is CUDA-only
        and covered by the routing/static tests).  The CPU-observable
        use_orig_params contract is: no ``FlatParameter`` objects, and
        every original parameter slot still exists with its original
        shape (params are FSDP-managed views, not flattened away).
        """
        args, trainer = _make_fsdp_trainer(tmp_path, monkeypatch, port=29512)
        try:
            reference = _reference_baseline_gpt(args)
            ref_params = dict(reference.named_parameters())
            assert ref_params  # sanity: reference is a real model

            def strip_prefix(name):
                while "_fsdp_wrapped_module." in name:
                    name = name.replace("_fsdp_wrapped_module.", "")
                return name

            fsdp_params = {
                strip_prefix(name): param for name, param in trainer._fsdp_module.named_parameters()
            }
            assert fsdp_params, (
                "FSDP root has no parameters: FSDP flattened them away "
                "(use_orig_params=True is missing)"
            )
            assert not any(
                type(p).__name__ == "FlatParameter" for p in fsdp_params.values()
            ), "FSDP exposes FlatParameter objects (use_orig_params missing)"
            assert set(fsdp_params) == set(ref_params), (
                "parameter slots under FSDP differ from the unwrapped "
                "model (use_orig_params=True must preserve them)"
            )
            for name, ref_p in ref_params.items():
                assert (
                    fsdp_params[name].shape == ref_p.shape
                ), f"parameter {name} changed shape under FSDP"
        finally:
            trainer.cleanup()

    def test_auto_wrap_policy_per_block(self, tmp_path, monkeypatch):
        """auto_wrap_policy: one FSDP wrapper per transformer Block.

        With ``transformer_auto_wrap_policy`` over the Block classes,
        the tiny 2-layer model yields at least root + 2 layer wrappers
        (>= 3 FSDP instances in the module tree).  Without the policy
        only the root wrapper exists (1 instance) -- the RED state.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel

        args, trainer = _make_fsdp_trainer(tmp_path, monkeypatch, port=29513)
        try:
            n_wrappers = sum(
                1 for m in trainer._fsdp_module.modules() if isinstance(m, FullyShardedDataParallel)
            )
            assert n_wrappers >= 3, (
                f"expected root + >= 2 per-Block FSDP wrappers, got "
                f"{n_wrappers} (auto_wrap_policy missing or not per-Block)"
            )
        finally:
            trainer.cleanup()

    def test_compile_order_static(self):
        """Static pin: FSDP( before torch.compile, with the new kwargs.

        torch.compile is unreliable on Windows so the order is pinned
        statically: within ``wrap_distributed`` the first ``FSDP(``
        construction must precede the first ``torch.compile`` call
        (FSDP-first, compile outermost -- FSDP1 has no ``_orig_mod``
        handling), and the source must pass ``use_orig_params=True``
        and an ``auto_wrap_policy`` (design.md L287, L320).
        """
        from training.trainer_base import BaseTrainer

        src = inspect.getsource(BaseTrainer.wrap_distributed)
        lines = src.splitlines()

        fsdp_lines = [i for i, line in enumerate(lines) if "FSDP(" in line]
        compile_lines = [i for i, line in enumerate(lines) if "torch.compile(" in line]
        assert fsdp_lines, "wrap_distributed never constructs FSDP(...)"
        assert compile_lines, "wrap_distributed never calls torch.compile"

        assert fsdp_lines[0] < compile_lines[0], (
            "FSDP wrap must happen BEFORE torch.compile (FSDP-first, "
            "compile outermost; compiling first then wrapping FSDP breaks "
            "because FSDP1 does not handle _orig_mod)"
        )
        assert "use_orig_params=True" in src, (
            "FSDP construction must pass use_orig_params=True " "(torch.compile compatibility)"
        )
        assert "auto_wrap_policy" in src, (
            "FSDP construction must pass an auto_wrap_policy "
            "(per-Block sharding, compute/communication overlap)"
        )


# ---------------------------------------------------------------------------
# B. TestStateDictHooks (shim-based, no real FSDP state dict)
# ---------------------------------------------------------------------------


class TestStateDictHooks:
    """BaseTrainer state-dict hooks + CheckpointManager wiring (FR9b).

    ``_get_model_state_dict`` / ``_load_model_state_dict`` move from
    PretrainTrainer (dead code today) to BaseTrainer and get wired into
    CheckpointManager via ``state_dict_fn`` / ``load_state_fn``.
    FSDP routing is exercised with stubs: the real FULL_STATE_DICT
    gather raises on CPU ("parameters on cpu") and must never run here.
    """

    def test_get_model_state_dict_nonfsdp_clean(self, tmp_path):
        """Non-FSDP path: clean keys + both hooks wired into the manager.

        ``trainer._get_model_state_dict()`` returns the raw model's
        state dict with no ``_orig_mod.`` / ``_fsdp_wrapped_module.``
        prefixes, and ``configure_checkpointing`` wired BOTH
        ``state_dict_fn`` and ``load_state_fn`` on the CheckpointManager
        (the dead-code fix: the hooks actually run on every save/load).
        """
        args, trainer = _make_pretrain_trainer(tmp_path, "out")
        try:
            sd = trainer._get_model_state_dict()
            keys = list(sd)
            assert keys  # sanity: a real state dict
            assert not any("_orig_mod." in key for key in keys)
            assert not any("_fsdp_wrapped_module." in key for key in keys)

            assert trainer.ckpt_manager.state_dict_fn is not None, (
                "configure_checkpointing must pass " "state_dict_fn=self._get_model_state_dict"
            )
            assert trainer.ckpt_manager.load_state_fn is not None, (
                "configure_checkpointing must pass " "load_state_fn=self._load_model_state_dict"
            )
        finally:
            trainer.cleanup()

    def test_checkpoint_manager_uses_state_dict_fn(self, tmp_path):
        """CheckpointManager.save takes the payload from state_dict_fn.

        A standalone manager with ``state_dict_fn=lambda: {"custom":
        ones(3)}`` must serialize exactly that payload (NOT
        ``model.state_dict()``), while model_type detection keeps
        working (the hook must not break it).
        """
        from training.trainer_base import CheckpointManager

        tiny = nn.Sequential(nn.Linear(4, 8))
        opt = torch.optim.SGD(tiny.parameters(), lr=0.01)
        out_dir = str(Path(tmp_path) / "ckpt_out")
        manager = CheckpointManager(
            out_dir=out_dir,
            model=tiny,
            optimizer=opt,
            config={"foo": 1},
            state_dict_fn=lambda: {"custom": torch.ones(3)},
        )
        path = manager.save("a.pt", 1, 1.0)
        manager.flush()  # async by default (T022): flush before reading

        ckpt = torch.load(path, weights_only=False)
        assert set(ckpt["model"]) == {
            "custom"
        }, "payload must come from state_dict_fn, not model.state_dict()"
        assert torch.equal(ckpt["model"]["custom"], torch.ones(3))
        assert ckpt["config"]["model_type"] in {"modern", "baseline"}, (
            "model_type detection must keep working alongside the " "state_dict_fn hook"
        )

    def test_checkpoint_manager_load_delegates(self, tmp_path):
        """CheckpointManager.load delegates to load_state_fn.

        With a ``load_state_fn`` recorder installed, ``manager.load``
        must hand it ``(model_src, ckpt["model"])`` exactly once and
        must NOT run the default ``model.load_state_dict`` path (the
        recorder does nothing, so the model's weights are untouched --
        e.g. an FSDP FULL_STATE_DICT load runs inside the hook instead).
        """
        from training.trainer_base import CheckpointManager

        tiny = nn.Sequential(nn.Linear(4, 8))
        opt = torch.optim.SGD(tiny.parameters(), lr=0.01)
        weights_before = [p.detach().clone() for p in tiny.parameters()]

        calls = []

        def load_state_fn(model, state_dict):
            calls.append((model, state_dict))

        out_dir = str(Path(tmp_path) / "ckpt_out")
        manager = CheckpointManager(
            out_dir=out_dir,
            model=tiny,
            optimizer=opt,
            config={"foo": 1},
            state_dict_fn=lambda: {"custom": torch.arange(3.0)},
            load_state_fn=load_state_fn,
        )
        path = manager.save("b.pt", 1, 1.0)
        manager.flush()

        manager.load(path)

        assert len(calls) == 1, "load must delegate to load_state_fn exactly once"
        model_arg, sd_arg = calls[0]
        assert model_arg is tiny
        assert set(sd_arg) == {"custom"}
        assert torch.equal(sd_arg["custom"], torch.arange(3.0))

        # The default path must not have run: weights are unchanged.
        for before, after in zip(weights_before, [p.detach().clone() for p in tiny.parameters()]):
            assert torch.equal(before, after)

    def test_get_model_state_dict_fsdp_routing(self, tmp_path, monkeypatch):
        """_get_model_state_dict routes through the FSDP FULL_STATE_DICT
        context on the recorded wrapper.

        FSDP guard satisfied via stubs (args.fsdp=True, distributed=True,
        ``_fsdp_module`` = plain stub): the call must enter
        ``FSDP.state_dict_type(self._fsdp_module, StateDictType.
        FULL_STATE_DICT, StateDictConfig(full_state_dict=True,
        cpu_offload=True))`` and return ``self._fsdp_module.state_dict()``
        -- never the raw model's (shard-partial, prefixed) state dict.
        """
        from torch.distributed.fsdp import StateDictType

        args, trainer = _make_pretrain_trainer(tmp_path, "out")
        try:
            stub = nn.Sequential(nn.Linear(2, 2))
            monkeypatch.setattr(trainer.args, "fsdp", True)
            monkeypatch.setattr(trainer, "distributed", True)
            monkeypatch.setattr(trainer, "_fsdp_module", stub, raising=False)
            monkeypatch.setattr(stub, "state_dict", lambda: {"fake": torch.zeros(2)})

            recorded = []
            _recording_state_dict_type(monkeypatch, recorded)

            out = trainer._get_model_state_dict()

            assert set(out) == {"fake"}
            assert torch.equal(out["fake"], torch.zeros(2))
            assert recorded, "FSDP.state_dict_type context must be entered"
            module_arg, sd_args, sd_kwargs = recorded[0]
            assert module_arg is stub, "state_dict_type must target self._fsdp_module"
            sd_type = sd_args[0] if sd_args else sd_kwargs.get("state_dict_type")
            assert sd_type == StateDictType.FULL_STATE_DICT
        finally:
            trainer.cleanup()

    def test_load_model_state_dict_fsdp_routing(self, tmp_path, monkeypatch):
        """_load_model_state_dict routes into the FSDP wrapper.

        Same stub setup: ``trainer._load_model_state_dict(model, sd)``
        must enter the ``FSDP.state_dict_type(..., FULL_STATE_DICT)``
        context and call ``self._fsdp_module.load_state_dict(sd)``;
        the passed-in (raw) model's ``load_state_dict`` must NOT be
        called on the FSDP path.
        """
        from torch.distributed.fsdp import StateDictType

        args, trainer = _make_pretrain_trainer(tmp_path, "out")
        try:
            stub = nn.Sequential(nn.Linear(2, 2))
            monkeypatch.setattr(trainer.args, "fsdp", True)
            monkeypatch.setattr(trainer, "distributed", True)
            monkeypatch.setattr(trainer, "_fsdp_module", stub, raising=False)

            recorded = []
            _recording_state_dict_type(monkeypatch, recorded)

            stub_calls = []
            raw_calls = []
            monkeypatch.setattr(
                stub,
                "load_state_dict",
                lambda sd: stub_calls.append(sd),
                raising=False,
            )
            monkeypatch.setattr(
                trainer.raw_model,
                "load_state_dict",
                lambda sd, strict=True: raw_calls.append(sd),
            )

            trainer._load_model_state_dict(trainer.raw_model, {"w": torch.ones(1)})

            assert recorded, "FSDP.state_dict_type context must be entered"
            module_arg, sd_args, sd_kwargs = recorded[0]
            assert module_arg is stub
            sd_type = sd_args[0] if sd_args else sd_kwargs.get("state_dict_type")
            assert sd_type == StateDictType.FULL_STATE_DICT

            assert len(stub_calls) == 1, "the FSDP wrapper's load_state_dict must be called once"
            assert torch.equal(stub_calls[0]["w"], torch.ones(1))
            assert raw_calls == [], "raw model load_state_dict must not run on the FSDP path"
        finally:
            trainer.cleanup()

    def test_fsdp_source_contract(self):
        """Static: the base hooks use FULL_STATE_DICT (and offload_to_cpu
        on the save side) -- the actual gather semantics of FR9b.

        torch 2.5.1's StateDictConfig exposes ``offload_to_cpu`` /
        ``rank0_only`` (``full_state_dict`` kwarg only exists in
        torch >= 2.6; the "full" semantics come from
        StateDictType.FULL_STATE_DICT itself).
        """
        from training.trainer_base import BaseTrainer

        src_get = inspect.getsource(BaseTrainer._get_model_state_dict)
        assert "FULL_STATE_DICT" in src_get
        assert "offload_to_cpu" in src_get, (
            "save-side FULL_STATE_DICT config must offload_to_cpu the "
            "gathered dict (rank0 checkpointing without device copies)"
        )
        assert "rank0_only" in src_get, (
            "full state dict only materializes on rank 0 (the " "checkpointing rank)"
        )

        src_load = inspect.getsource(BaseTrainer._load_model_state_dict)
        assert "FULL_STATE_DICT" in src_load


# ---------------------------------------------------------------------------
# C. TestFSDPAtomicity (E7)
# ---------------------------------------------------------------------------


class TestFSDPAtomicity:
    """E7 (design.md L550): a failed FSDP save must never corrupt the
    previous checkpoint (tmp + atomic rename; no half-written file)."""

    def test_fsdp_save_interrupt_keeps_old_file(self, tmp_path, monkeypatch):
        """torch.save always failing: WARN, old checkpoint intact, no tmp.

        First save+flush leaves a good ``latest_ckpt.pt`` (iter 1).  With
        ``torch.save`` patched to always raise, a second async save (iter
        2) must WARN, leave the OLD file fully loadable at the target
        path, and leave no ``*.tmp`` residue behind.
        """
        from training.trainer_base import CheckpointManager

        tiny = nn.Sequential(nn.Linear(4, 8))
        opt = torch.optim.SGD(tiny.parameters(), lr=0.01)
        out_dir = str(Path(tmp_path) / "ckpt_out")
        manager = CheckpointManager(
            out_dir=out_dir,
            model=tiny,
            optimizer=opt,
            config={"foo": 1},
            state_dict_fn=lambda: {"custom": torch.ones(3)},
        )
        path = manager.save("latest_ckpt.pt", 1, 0.5)
        manager.flush()
        old = torch.load(path, weights_only=False)
        assert old["iter_num"] == 1  # baseline: the good old file exists

        def boom(*save_args, **save_kwargs):
            raise OSError("simulated disk failure mid-save")

        monkeypatch.setattr(torch, "save", boom)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            manager.save("latest_ckpt.pt", 2, 0.4, async_save=True)
            manager.flush()

        assert any(
            issubclass(w.category, RuntimeWarning) for w in caught
        ), "a failed save must WARN"

        # E7: the old checkpoint is untouched and fully loadable.
        still = torch.load(path, weights_only=False)
        assert (
            still["iter_num"] == 1
        ), "the interrupted save must not corrupt the previous checkpoint"
        assert (
            list(Path(out_dir).glob("*.tmp")) == []
        ), "no half-written tmp file may survive the failed save"


# ---------------------------------------------------------------------------
# D. TestPretrainRemoval
# ---------------------------------------------------------------------------


class TestPretrainRemoval:
    """The dead-code hooks move UP to BaseTrainer (design L288)."""

    def test_pretrain_no_local_state_dict_override(self):
        """PretrainTrainer must not shadow the base hooks.

        ``_get_model_state_dict`` (and its load twin) belong to
        BaseTrainer now; a PretrainTrainer-local override is exactly the
        dead-code duplication this task removes.
        """
        from training.train_pretrain import PretrainTrainer
        from training.trainer_base import BaseTrainer

        assert "_get_model_state_dict" not in vars(PretrainTrainer), (
            "_get_model_state_dict moved to BaseTrainer; the "
            "PretrainTrainer override must be removed"
        )
        assert "_get_model_state_dict" in vars(
            BaseTrainer
        ), "BaseTrainer provides the generalized state-dict hook"
        assert "_load_model_state_dict" in vars(
            BaseTrainer
        ), "BaseTrainer provides the generalized load hook"
