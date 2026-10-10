"""Tests for T008 (FR8a): enable_gqa SDPA flag + module-level TF32.

Covers three contract points of AR-v0.5.0-hpc-readiness:

* ``sdpa_supports_enable_gqa()`` — static constructor-time detection of the
  ``enable_gqa`` kwarg in ``F.scaled_dot_product_attention`` (torch >= 2.5).
* ``CausalSelfAttention`` GQA "native" mode — ``gqa_broadcast="auto"`` now
  resolves statically in ``__init__`` (compile graph-break fix: the runtime
  probe is no longer called from ``forward``).  Native mode passes
  ``enable_gqa=True`` to SDPA and falls back permanently to the grouped
  broadcast (with a one-time ``UserWarning``) if the runtime call raises.
* ``enable_tf32()`` in ``training/trainer_base.py`` — idempotent TF32
  enablement, invoked at module level by all five trainer scripts
  (train_pretrain / train_sft / train_grpo / train_dpo / iterative_grpo).
* T009 (FR8b): env-tiered ``num_workers`` defaults + ``persistent_workers``
  in ``data/openwebtext.py``, and the slimmed ``PretrainTrainer.estimate_loss``
  (eval batch amplification, device-side loss accumulation with all_reduce,
  dedicated ``train_eval_loader``).
"""

import argparse
import contextlib
import importlib
import inspect
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from model.modern_gpt import ModernGPT, ModernGPTConfig
from helpers import tiny_modern_config


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------


def _sdpa_accepts_enable_gqa():
    """Ground-truth probe: does the installed SDPA accept ``enable_gqa``?

    Used only to guard environment-dependent assertions (torch < 2.5 does
    not have the kwarg at all).
    """
    try:
        q = torch.randn(1, 4, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return True
    except Exception:
        return False


def _make_gqa_model(gqa_broadcast="auto"):
    """Build a tiny GQA ModernGPT (n_kv_head=2, n_head=4) for testing."""
    cfg = tiny_modern_config(
        n_layer=1,
        n_head=4,
        n_embd=64,
        block_size=32,
        n_kv_head=2,
        vocab_size=100,
        dropout=0.0,
        gqa_broadcast=gqa_broadcast,
    )
    return ModernGPT(cfg)


def _make_input():
    torch.manual_seed(1234)
    return torch.randint(0, 100, (1, 8))


# ---------------------------------------------------------------------------
#  sdpa_supports_enable_gqa (attention_utils)
# ---------------------------------------------------------------------------


class TestSDPAEnableGQASupport:
    """Tests for model.attention_utils.sdpa_supports_enable_gqa."""

    def test_returns_bool(self):
        """The detector must return a plain bool."""
        # Lazy import: the function does not exist yet (Red phase), so a
        # module-level import would break collection for the whole file.
        from model.attention_utils import sdpa_supports_enable_gqa

        result = sdpa_supports_enable_gqa()
        assert isinstance(result, bool)

    def test_true_when_torch_sdpa_supports_enable_gqa(self):
        """On torch >= 2.5 (SDPA accepts enable_gqa) the detector is True.

        Environment tolerance: if the installed torch genuinely cannot
        accept the kwarg (torch < 2.5), the True assertion is skipped
        instead of failing for reasons unrelated to the contract.
        """
        if not _sdpa_accepts_enable_gqa():
            pytest.skip("installed torch SDPA does not accept enable_gqa (<2.5)")
        from model.attention_utils import sdpa_supports_enable_gqa

        assert sdpa_supports_enable_gqa() is True


# ---------------------------------------------------------------------------
#  GQA native mode (CausalSelfAttention)
# ---------------------------------------------------------------------------


class TestGQANativeMode:
    """Tests for the "native" GQA mode and static auto resolution."""

    def test_auto_resolves_to_native_at_init(self):
        """gqa_broadcast='auto' must resolve to 'native' in __init__.

        Static constructor-time resolution (no forward, no lazy probe) is
        what removes the torch.compile graph break.
        """
        model = _make_gqa_model()  # default gqa_broadcast="auto"
        assert model.transformer.h[0].attn._gqa_mode == "native"

    def test_native_forward_passes_enable_gqa_true(self, monkeypatch):
        """Native forward must call SDPA with enable_gqa=True at least once.

        The wrapper records kwargs and delegates to the real SDPA.  The
        grouped fallback also calls SDPA (without the kwarg), so we only
        require *at least one* call carrying ``enable_gqa=True``.
        """
        model = _make_gqa_model()
        model.eval()
        x = _make_input()

        calls = []
        orig_sdpa = torch.nn.functional.scaled_dot_product_attention

        def recording_sdpa(*args, **kwargs):
            calls.append(kwargs)
            return orig_sdpa(*args, **kwargs)

        monkeypatch.setattr(
            torch.nn.functional, "scaled_dot_product_attention", recording_sdpa
        )
        with torch.no_grad():
            model(x)
        assert any(kw.get("enable_gqa") is True for kw in calls)

    def test_native_output_matches_repeat_interleave(self):
        """Native mode logits must match the repeat_interleave reference.

        Guard: we assert the mode is still "native" after the first forward
        so the comparison truly exercises the native path (and so this test
        stays Red until native mode exists).
        """
        torch.manual_seed(42)
        model = _make_gqa_model()
        model.eval()
        x = _make_input()

        with torch.no_grad():
            logits_native, _, _ = model(x)
        attn = model.transformer.h[0].attn
        assert attn._gqa_mode == "native"

        attn._gqa_mode = "repeat"
        with torch.no_grad():
            logits_repeat, _, _ = model(x)
        assert logits_native.shape == logits_repeat.shape
        assert torch.allclose(logits_native, logits_repeat, atol=1e-5, rtol=1e-5)

    def test_auto_falls_back_to_grouped_when_unsupported(self, monkeypatch):
        """B8: when the detector says unsupported, auto resolves to grouped.

        modern_gpt imports the detector via
        ``from .attention_utils import sdpa_supports_enable_gqa`` so
        patching the attribute on the modern_gpt module is effective.
        """
        monkeypatch.setattr(
            "model.modern_gpt.sdpa_supports_enable_gqa", lambda: False
        )
        model = _make_gqa_model()
        assert model.transformer.h[0].attn._gqa_mode == "grouped"

        # The grouped fallback must still produce a valid forward.
        model.eval()
        x = _make_input()
        with torch.no_grad():
            logits, _, _ = model(x)
        assert logits.shape == (1, 8, 100)

    def test_native_runtime_error_falls_back_to_grouped(self, monkeypatch):
        """E6: runtime failure of the enable_gqa SDPA call degrades once.

        The wrapper raises RuntimeError only for calls carrying
        ``enable_gqa=True`` and delegates everything else, so the grouped
        fallback path remains usable.  The first forward must:
        1. emit a UserWarning mentioning "enable_gqa",
        2. still return a valid result,
        3. permanently switch _gqa_mode to "grouped".
        """
        model = _make_gqa_model()
        model.eval()
        x = _make_input()

        orig_sdpa = torch.nn.functional.scaled_dot_product_attention

        def flaky_sdpa(*args, **kwargs):
            if kwargs.get("enable_gqa") is True:
                raise RuntimeError("simulated enable_gqa runtime failure")
            return orig_sdpa(*args, **kwargs)

        monkeypatch.setattr(
            torch.nn.functional, "scaled_dot_product_attention", flaky_sdpa
        )
        with pytest.warns(UserWarning, match="enable_gqa"):
            with torch.no_grad():
                logits, _, _ = model(x)
        assert logits.shape == (1, 8, 100)
        assert model.transformer.h[0].attn._gqa_mode == "grouped"

    def test_fallback_warning_emitted_only_once(self, monkeypatch):
        """One-time WARN: the second forward after E6 must not warn again.

        The wrapper still raises on enable_gqa calls, but the model has
        permanently switched to grouped so the native path is never
        re-entered.  The first forward is asserted inside pytest.warns so
        this test stays Red until the fallback exists.
        """
        model = _make_gqa_model()
        model.eval()
        x = _make_input()

        orig_sdpa = torch.nn.functional.scaled_dot_product_attention

        def flaky_sdpa(*args, **kwargs):
            if kwargs.get("enable_gqa") is True:
                raise RuntimeError("simulated enable_gqa runtime failure")
            return orig_sdpa(*args, **kwargs)

        monkeypatch.setattr(
            torch.nn.functional, "scaled_dot_product_attention", flaky_sdpa
        )
        with pytest.warns(UserWarning, match="enable_gqa"):
            with torch.no_grad():
                model(x)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                logits, _, _ = model(x)
        assert logits.shape == (1, 8, 100)
        assert not any("enable_gqa" in str(w.message) for w in caught)

    def test_forward_does_not_call_runtime_probe(self, monkeypatch):
        """Compile fix: forward must not call probe_gqa_sdpa_support.

        The runtime probe inside forward is a graph break for
        torch.compile; it must be gone.  We patch the probe to raise
        AssertionError and require a clean forward.
        """
        def _probe_must_not_run(*args, **kwargs):
            raise AssertionError("probe must not be called")

        monkeypatch.setattr(
            "model.attention_utils.probe_gqa_sdpa_support", _probe_must_not_run
        )
        model = _make_gqa_model()
        model.eval()
        x = _make_input()
        with torch.no_grad():
            logits, _, _ = model(x)
        assert logits.shape == (1, 8, 100)

    @pytest.mark.parametrize("mode", ["grouped", "repeat"])
    def test_explicit_mode_preserved_at_init(self, mode):
        """Explicit gqa_broadcast stays honoured at __init__ and forward.

        Regression guard: explicit "grouped"/"repeat" behaviour is unchanged
        by FR8a, so this case may already pass in the Red phase (it does
        not depend on the not-yet-existing native mode); it must keep
        passing after the change.
        """
        model = _make_gqa_model(gqa_broadcast=mode)
        assert model.transformer.h[0].attn._gqa_mode == mode
        model.eval()
        x = _make_input()
        with torch.no_grad():
            logits, _, _ = model(x)
        assert logits.shape == (1, 8, 100)


# ---------------------------------------------------------------------------
#  enable_tf32 (trainer_base) + trainer script wiring
# ---------------------------------------------------------------------------

TRAINER_MODULES = (
    "training.train_pretrain",
    "training.train_sft",
    "training.train_grpo",
    "training.train_dpo",
    "training.iterative_grpo",
)

TRAINER_FILES = (
    "train_pretrain.py",
    "train_sft.py",
    "train_grpo.py",
    "train_dpo.py",
    "iterative_grpo.py",
)


class TestEnableTF32:
    """Tests for training.trainer_base.enable_tf32 and its wiring."""

    def test_enable_tf32_sets_both_flags(self):
        """enable_tf32 must flip both matmul and cudnn TF32 flags to True.

        Global-state tolerance: we force both flags to False first (another
        import earlier in the session may have left them True) and restore
        the original values afterwards.  Idempotency is checked by calling
        twice.
        """
        from training.trainer_base import enable_tf32

        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            enable_tf32()
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
            # Idempotent: a second call keeps them True.
            enable_tf32()
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

    @pytest.mark.parametrize("module_name", TRAINER_MODULES)
    def test_trainer_import_enables_tf32(self, module_name):
        """Importing each trainer module must leave both TF32 flags True.

        Global-state tolerance: flags are forced to False and the module is
        removed from sys.modules before importing, so the assertion checks
        the module-level ``enable_tf32()`` call itself rather than a stale
        flag value left over from an earlier import in the session.
        """
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

    @pytest.mark.parametrize("fname", TRAINER_FILES)
    def test_trainer_source_contains_enable_tf32_call(self, fname):
        """FR8-AC3 (grep semantics): each trainer source calls enable_tf32()."""
        path = Path(__file__).resolve().parents[1] / "training" / fname
        source = path.read_text(encoding="utf-8")
        assert "enable_tf32()" in source


# ---------------------------------------------------------------------------
#  T009 (FR8b) helpers: tiny synthetic data + args namespace
# ---------------------------------------------------------------------------

_T009_BLOCK_SIZE = 32
_T009_N_TRAIN_CHUNKS = 40
_T009_N_VAL_CHUNKS = 8


@contextlib.contextmanager
def _os_name(name):
    """Temporarily patch ``os.name``, restoring it on *any* exit path.

    A plain monkeypatch of ``os.name`` to "posix" on Windows breaks
    pytest's own failure reporting (pathlib refuses to build PosixPath
    while the patch is live), so the patch scope must end before any
    exception reaches the reporter.
    """
    old = os.name
    os.name = name
    try:
        yield
    finally:
        os.name = old


def _write_tiny_data(tmp_path):
    """Write tiny train/val .bin + .idx + sequential train shuffle idx.

    Self-contained copy of the ``_write_tiny_data`` pattern from
    tests/test_resume_integrity.py, shrunk for T009 (block_size=32,
    train 40 chunks, val 8 chunks).
    """
    data_dir = Path(tmp_path) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, n_chunks in (
        ("train", _T009_N_TRAIN_CHUNKS),
        ("val", _T009_N_VAL_CHUNKS),
    ):
        total = n_chunks * (_T009_BLOCK_SIZE + 1)
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
            f"block_size={_T009_BLOCK_SIZE}\n",
            encoding="utf-8",
        )
    # Sequential shuffle index: chunk order is 0, 1, 2, ... (deterministic).
    with open(data_dir / "train.bin.shuffle.idx", "wb") as f:
        np.save(f, np.arange(_T009_N_TRAIN_CHUNKS, dtype=np.int64))
    return str(data_dir)


def _make_args(data_dir, out_dir, **overrides):
    """Fully populated args namespace for a tiny CPU PretrainTrainer.

    Field pattern copied from tests/test_resume_integrity.py ``_make_args``.
    Deliberately does NOT include ``eval_batch_size``: the trainer must
    tolerate a missing key via getattr fallback (defaults to 2*batch_size).
    """
    defaults = dict(
        model="baseline",
        data_dir=data_dir,
        out_dir=out_dir,
        batch_size=4,
        block_size=_T009_BLOCK_SIZE,
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
        exp_id="t009-eval-slimming",
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


# ---------------------------------------------------------------------------
#  T009 (FR8b): default_num_workers env tiering
# ---------------------------------------------------------------------------


class TestDefaultNumWorkers:
    """data.openwebtext.default_num_workers: env-tiered default (FR8b).

    Windows must default to 0 (spawn overhead dominates), everything else
    to 4.
    """

    def test_windows_returns_zero(self):
        from data.openwebtext import default_num_workers

        with _os_name("nt"):
            result = default_num_workers()
        assert result == 0

    def test_posix_returns_four(self):
        from data.openwebtext import default_num_workers

        with _os_name("posix"):
            result = default_num_workers()
        assert result == 4


# ---------------------------------------------------------------------------
#  T009 (FR8b): get_dataloader data path (num_workers / persistent_workers /
#  pin_memory)
# ---------------------------------------------------------------------------


class TestGetDataloaderDataPath:
    """get_dataloader signature change (FR8b).

    * ``num_workers=None`` resolves via ``default_num_workers()`` (env
      tiered); an explicit value wins regardless of the environment.
    * ``persistent_workers=None`` auto-resolves to ``num_workers > 0``;
      an explicit value wins.
    * ``pin_memory=True`` stays unchanged.
    """

    @staticmethod
    def _make_loader(data_dir, **kwargs):
        from data.openwebtext import get_dataloader

        params = dict(
            data_dir=data_dir,
            split="train",
            batch_size=4,
            block_size=_T009_BLOCK_SIZE,
        )
        params.update(kwargs)
        return get_dataloader(**params)

    def test_num_workers_none_resolves_zero_on_windows(self, tmp_path):
        data_dir = _write_tiny_data(tmp_path)
        with _os_name("nt"):
            loader = self._make_loader(data_dir, num_workers=None)
        assert loader.num_workers == 0
        assert loader.pin_memory is True

    def test_num_workers_none_resolves_four_on_posix(self, tmp_path):
        data_dir = _write_tiny_data(tmp_path)
        with _os_name("posix"):
            loader = self._make_loader(data_dir, num_workers=None)
        assert loader.num_workers == 4

    def test_explicit_num_workers_wins_over_env(self, tmp_path):
        data_dir = _write_tiny_data(tmp_path)
        with _os_name("nt"):
            loader = self._make_loader(data_dir, num_workers=2, persistent_workers=None)
        assert loader.num_workers == 2
        # Auto persistent_workers for a positive worker count.
        assert loader.persistent_workers is True

    def test_persistent_workers_auto_and_explicit(self, tmp_path):
        data_dir = _write_tiny_data(tmp_path)
        # Auto: num_workers=0 -> persistent_workers False.
        loader0 = self._make_loader(data_dir, num_workers=0, persistent_workers=None)
        assert loader0.persistent_workers is False
        # Auto: num_workers=2 -> persistent_workers True.
        loader2 = self._make_loader(data_dir, num_workers=2, persistent_workers=None)
        assert loader2.persistent_workers is True
        # Explicit override wins even with a positive worker count.
        loader_off = self._make_loader(
            data_dir, num_workers=2, persistent_workers=False
        )
        assert loader_off.persistent_workers is False

    def test_pin_memory_stays_true(self, tmp_path):
        data_dir = _write_tiny_data(tmp_path)
        with _os_name("nt"):
            loader = self._make_loader(data_dir, num_workers=None)
        assert loader.pin_memory is True


# ---------------------------------------------------------------------------
#  T009 (FR8b): PretrainTrainer eval slimming + data-path wiring
# ---------------------------------------------------------------------------


class TestPretrainEvalSlimming:
    """PretrainTrainer changes (FR8b).

    * ``eval_batch_size`` resolution: missing key -> 2*batch_size,
      explicit value respected (train batch size untouched).
    * ``num_workers=None`` resolves via ``default_num_workers()``.
    * ``train_eval_loader``: train split, eval batch size, dedicated
      dataset instance, resume offset fixed at construction time.
    * persistent_workers layout: train_loader always False,
      val_loader/train_eval_loader follow the resolved worker count.
    * ``estimate_loss``: finite float contract, train value read from
      train_eval_loader, no per-iter ``.item()`` sync (static check),
      distributed path via a single all_reduce per split.
    """

    @staticmethod
    def _make_trainer(tmp_path, **overrides):
        from training.train_pretrain import PretrainTrainer

        data_dir = _write_tiny_data(tmp_path)
        out_dir = str(Path(tmp_path) / "out")
        args = _make_args(data_dir, out_dir, **overrides)
        return args, PretrainTrainer(args)

    def test_eval_batch_size_defaults_to_double(self, tmp_path):
        args, trainer = self._make_trainer(tmp_path)
        try:
            assert not hasattr(args, "eval_batch_size")
            expected = 2 * args.batch_size
            assert trainer.val_loader.batch_size == expected
            assert trainer.train_eval_loader.batch_size == expected
        finally:
            trainer.cleanup()

    def test_eval_batch_size_explicit_respected(self, tmp_path):
        args, trainer = self._make_trainer(tmp_path, eval_batch_size=3)
        try:
            assert trainer.val_loader.batch_size == 3
            assert trainer.train_eval_loader.batch_size == 3
            # Training batch size is untouched by the eval amplification.
            assert trainer.train_loader.batch_size == args.batch_size
        finally:
            trainer.cleanup()

    def test_num_workers_none_resolves_via_env(self, tmp_path):
        from data.openwebtext import default_num_workers

        args, trainer = self._make_trainer(tmp_path, num_workers=None)
        try:
            resolved = default_num_workers()
            assert trainer.train_loader.num_workers == resolved
            assert trainer.val_loader.num_workers == resolved
        finally:
            trainer.cleanup()

    def test_train_eval_loader_split_and_layout(self, tmp_path):
        args, trainer = self._make_trainer(tmp_path)
        try:
            eval_bs = 2 * args.batch_size
            assert trainer.train_loader.batch_size == args.batch_size
            assert trainer.val_loader.batch_size == eval_bs
            assert trainer.train_eval_loader.batch_size == eval_bs
            # train split (40 chunks), not the val split (8 chunks).
            assert len(trainer.train_eval_loader.dataset) == _T009_N_TRAIN_CHUNKS
            assert len(trainer.val_loader.dataset) == _T009_N_VAL_CHUNKS
            # Dedicated dataset instance, offset fixed at build time.
            assert (
                trainer.train_eval_loader.dataset is not trainer.train_loader.dataset
            )
            assert (
                trainer.train_eval_loader.dataset.resume_offset
                == trainer.resume_offset
            )
        finally:
            trainer.cleanup()

    def test_persistent_workers_layout(self, tmp_path):
        from data.openwebtext import default_num_workers

        args, trainer = self._make_trainer(tmp_path, num_workers=None)
        try:
            resolved_nw = default_num_workers()
            # train_loader is never persistent: epoch wrap rebuilds workers
            # so dataset.load_state_dict is re-pickled with the new offset.
            assert trainer.train_loader.persistent_workers is False
            assert trainer.val_loader.persistent_workers == (resolved_nw > 0)
            assert trainer.train_eval_loader.persistent_workers == (resolved_nw > 0)
        finally:
            trainer.cleanup()

    def test_estimate_loss_finite_and_from_train_eval_loader(self, tmp_path):
        args, trainer = self._make_trainer(tmp_path)
        try:
            out = trainer.estimate_loss()
            assert set(out) == {"train", "val"}
            for split in ("train", "val"):
                assert isinstance(out[split], float)
                assert math.isfinite(out[split])

            # The train value must equal the mean loss over the first
            # eval_iters batches of the dedicated train_eval_loader (and
            # iterating it must not disturb its state, so a second pass
            # yields the same batches).
            trainer.model.eval()
            ref_losses = []
            with torch.no_grad():
                for k, batch in enumerate(trainer.train_eval_loader):
                    if k >= args.eval_iters:
                        break
                    batch = [t.to(trainer.device, non_blocking=True) for t in batch]
                    with trainer.ctx:
                        _, loss, _ = trainer._forward(batch)
                    ref_losses.append(loss.detach())
            ref = torch.stack(ref_losses).mean().item()
            assert math.isclose(out["train"], ref, rel_tol=1e-5, abs_tol=1e-6)
        finally:
            trainer.cleanup()

    def test_estimate_loss_source_is_sync_free(self):
        """Static (grep semantics) checks on the rewritten estimate_loss.

        * no per-iter ``losses[k] = loss.item()`` device sync,
        * distributed path present via all_reduce,
        * train split read from train_eval_loader (never train_loader),
        * no bare ``[t.to(self.device) for t in batch]`` migration
          (the new path uses non_blocking=True).
        """
        from training.train_pretrain import PretrainTrainer

        src = inspect.getsource(PretrainTrainer.estimate_loss)
        assert "losses[k] = loss.item()" not in src
        assert "all_reduce" in src
        assert "train_eval_loader" in src
        assert "[t.to(self.device) for t in batch]" not in src
        assert "non_blocking" in src

    def test_estimate_loss_pseudo_distributed_allreduce(self, tmp_path, monkeypatch):
        args, trainer = self._make_trainer(tmp_path)
        try:
            base = trainer.estimate_loss()

            calls = []

            def fake_all_reduce(tensor, op=None):
                # SUM over world_size=2 ranks holding identical values
                # doubles the tensor; the trainer then divides by
                # world_size, restoring the per-rank mean.
                calls.append(op)
                tensor.mul_(2)

            monkeypatch.setattr(trainer, "distributed", True)
            monkeypatch.setattr(trainer, "world_size", 2)
            monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

            out = trainer.estimate_loss()
            assert math.isclose(
                out["train"], base["train"], rel_tol=1e-5, abs_tol=1e-6
            )
            assert math.isclose(
                out["val"], base["val"], rel_tol=1e-5, abs_tol=1e-6
            )
            # Exactly one all_reduce per split (train, val).
            assert len(calls) == 2
            for op in calls:
                assert op == torch.distributed.ReduceOp.SUM
        finally:
            trainer.cleanup()


# ---------------------------------------------------------------------------
#  T022 (AR-v0.5.0-hpc-readiness) module-level extras (appended section)
# ---------------------------------------------------------------------------

import copy
import gc
import re


# ---------------------------------------------------------------------------
#  T022 (FR8): EMA foreach update (ModernGPT.update_ema)
# ---------------------------------------------------------------------------


class TestEMAForeachUpdate:
    """Tests for the foreach-based ``ModernGPT.update_ema`` (T022).

    Contract (design.md L297): ``update_ema`` must route through
    ``torch._foreach_lerp_`` on flattened lists of requires_grad
    parameters (200+ per-parameter kernels -> 2-3 foreach kernels) while
    staying numerically identical to the previous per-parameter loop
    ``shadow.mul_(decay).add_(param.data, alpha=1-decay)``.  The
    ``ema_shadow`` dict-of-tensors disk format (populated by
    ``init_ema``) is unchanged.
    """

    @staticmethod
    def _make_ema_model():
        cfg = tiny_modern_config(
            n_layer=2,
            n_head=4,
            n_embd=32,
            block_size=32,
            vocab_size=100,
            dropout=0.0,
        )
        torch.manual_seed(1234)
        return ModernGPT(cfg)

    @staticmethod
    def _reference_update(model, decay):
        """Old per-parameter EMA loop (the numerical ground truth)."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    model.ema_shadow[name].mul_(decay).add_(
                        param.data, alpha=1.0 - decay
                    )

    def test_update_ema_uses_foreach_lerp(self, monkeypatch):
        """update_ema must invoke torch._foreach_lerp_ at least once.

        The recorder delegates to the original so a correct foreach
        implementation is unaffected; only the routing is asserted.
        """
        model = self._make_ema_model()
        model.init_ema(decay=0.9)

        calls = []
        orig_foreach = torch._foreach_lerp_

        def recording_foreach(*args, **kwargs):
            calls.append(args)
            return orig_foreach(*args, **kwargs)

        monkeypatch.setattr(torch, "_foreach_lerp_", recording_foreach)
        model.update_ema()
        assert len(calls) >= 1

    def test_update_ema_matches_reference(self):
        """Foreach update must equal the per-parameter loop (atol=1e-6).

        decay=0.9 maximises sensitivity to a decay/(1-decay) inversion in
        the lerp weight, which a naive ``_foreach_lerp_(s, p, decay)``
        implementation would introduce.
        """
        decay = 0.9
        model_a = self._make_ema_model()
        model_a.init_ema(decay=decay)

        torch.manual_seed(42)
        with torch.no_grad():
            for param in model_a.parameters():
                if param.requires_grad:
                    param.data.add_(torch.randn_like(param.data) * 0.1)

        model_b = copy.deepcopy(model_a)  # identical params + shadow
        model_a.update_ema()
        self._reference_update(model_b, decay)

        assert set(model_a.ema_shadow) == set(model_b.ema_shadow)
        for name in model_a.ema_shadow:
            assert torch.allclose(
                model_a.ema_shadow[name], model_b.ema_shadow[name], atol=1e-6
            )

    def test_update_ema_multiple_steps(self):
        """Equivalence must hold across consecutive updates, not just one."""
        decay = 0.9
        model_a = self._make_ema_model()
        model_a.init_ema(decay=decay)
        model_b = copy.deepcopy(model_a)

        for step in range(3):
            torch.manual_seed(500 + step)
            with torch.no_grad():
                for (_, param_a), (_, param_b) in zip(
                    model_a.named_parameters(), model_b.named_parameters()
                ):
                    if param_a.requires_grad:
                        delta = torch.randn_like(param_a.data) * 0.1
                        param_a.data.add_(delta)
                        param_b.data.add_(delta)
            model_a.update_ema()
            self._reference_update(model_b, decay)
            for name in model_a.ema_shadow:
                assert torch.allclose(
                    model_a.ema_shadow[name],
                    model_b.ema_shadow[name],
                    atol=1e-6,
                )

    def test_requires_grad_filter_regression(self):
        """A param frozen BEFORE init_ema stays out of ema_shadow.

        The flattened foreach lists must be built from the same
        requires_grad filter as the old loop; update_ema must not raise
        on the frozen parameter.
        """
        model = self._make_ema_model()
        frozen_name, frozen_param = next(iter(model.named_parameters()))
        frozen_param.requires_grad_(False)
        model.init_ema(decay=0.9)
        assert frozen_name not in model.ema_shadow

        pre = {name: t.clone() for name, t in model.ema_shadow.items()}
        torch.manual_seed(99)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    assert name in model.ema_shadow
                    param.data.add_(torch.randn_like(param.data) * 0.5)

        model.update_ema()  # must not raise despite the frozen param
        for name, tensor in model.ema_shadow.items():
            assert not torch.equal(tensor, pre[name])

    def test_apply_restore_swap_regression(self):
        """apply/restore EMA swap semantics are unchanged by the rewrite."""
        model = self._make_ema_model()
        model.init_ema(decay=0.9)
        tracked = [n for n, p in model.named_parameters() if p.requires_grad]

        torch.manual_seed(7)
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.add_(torch.randn_like(param.data) * 0.3)
        # Values at the moment apply_ema_weights is called ("pre-apply").
        pre_apply = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        pre_shadow = {n: t.clone() for n, t in model.ema_shadow.items()}

        model.apply_ema_weights()
        current = dict(model.named_parameters())
        for name in tracked:
            assert torch.equal(current[name].data, pre_shadow[name])

        model.restore_ema_weights()
        current = dict(model.named_parameters())
        for name in tracked:
            assert torch.equal(current[name].data, pre_apply[name])


# ---------------------------------------------------------------------------
#  T022 (FR8): micro-step desync in PretrainTrainer.train
# ---------------------------------------------------------------------------


class TestMicroStepDesync:
    """Tests for the micro-step desync rewrite of ``PretrainTrainer.train``.

    Contract (design.md L296): the train loss is accumulated as a tensor
    across micro-steps; at each accumulation boundary exactly ONE
    ``dist.all_reduce`` and ONE ``.item()`` happen (no per-micro-step
    sync); log lines read the accumulated value; tokens are counted per
    micro-step, not derived from the optimizer-step counter.

    Note on iter semantics: in this codebase the ``while`` loop consumes
    one batch per iteration, so ``iter_num`` counts MICRO-steps.  A run
    with ``max_iters=8, accum=2`` therefore performs 8 micro-steps and
    hits 4 accumulation windows - the configuration the spec describes
    as "4 train windows".
    """

    @staticmethod
    def _make_trainer(tmp_path, **overrides):
        from training.train_pretrain import PretrainTrainer

        data_dir = _write_tiny_data(tmp_path)
        out_dir = str(Path(tmp_path) / "out")
        run = dict(
            max_iters=8,
            gradient_accumulation_steps=2,
            eval_interval=100,  # > max_iters: no mid-train eval
            log_interval=100,
            grad_clip=0,
            use_ema=False,
            batch_size=2,
        )
        run.update(overrides)
        args = _make_args(data_dir, out_dir, **run)
        return args, PretrainTrainer(args)

    def test_no_per_microstep_item_in_train_source(self):
        """Static: the per-micro-step ``raw_loss = loss.item()`` is gone."""
        from training.train_pretrain import PretrainTrainer

        src = inspect.getsource(PretrainTrainer.train)
        assert "raw_loss = loss.item()" not in src

    def test_single_sync_per_accumulation_window(self, tmp_path, monkeypatch):
        """Pseudo-distributed run: one all_reduce per accumulation window.

        With 8 micro-steps / accum=2 there are 4 windows, so the train
        loop must issue exactly 4 all_reduce calls, plus exactly 2 from
        the final ``estimate_loss`` (one per split) = 6 total.  The old
        per-micro-step pattern produces 8 + 2 = 10.
        """
        args, trainer = self._make_trainer(tmp_path)
        try:
            calls = []

            def fake_all_reduce(tensor, op=None):
                calls.append(op)

            monkeypatch.setattr(trainer, "distributed", True)
            monkeypatch.setattr(trainer, "world_size", 1)
            monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

            trainer.train()
            assert len(calls) == 6
        finally:
            trainer.cleanup()

    def test_item_calls_reduced_to_optimizer_steps(self, tmp_path, monkeypatch):
        """Tensor.item counting: one call per window + one per eval split.

        Same run as the all_reduce test (4 windows, final eval): the new
        path must perform exactly 4 train-loop ``.item()`` calls plus 2
        from the final ``estimate_loss`` (one per split mean) = 6 total.
        The old per-micro-step pattern produces 8 + 2 = 10.

        The counter only counts ``.item()`` calls made directly from this
        repo's training code (train_pretrain / trainer_base): the AdamW
        optimizer's internal per-parameter ``_get_value`` item calls
        (torch site-packages) are unavoidable on every optimizer.step()
        and are deliberately excluded.
        """
        args, trainer = self._make_trainer(tmp_path)
        try:
            orig_item = torch.Tensor.item
            counter = []

            def counting_item(self):
                caller = sys._getframe(1).f_code.co_filename
                if "train" in os.path.basename(caller):
                    counter.append(caller)
                return orig_item(self)

            monkeypatch.setattr(torch.Tensor, "item", counting_item)

            calls = []

            def fake_all_reduce(tensor, op=None):
                calls.append(op)

            monkeypatch.setattr(trainer, "distributed", True)
            monkeypatch.setattr(trainer, "world_size", 1)
            monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

            trainer.train()
            assert len(counter) == 6
        finally:
            trainer.cleanup()

    def test_accum_window_train_loss_logged(self, tmp_path, capsys):
        """Logged train loss stays finite/positive from the tensor path.

        Behavioral regression guard for the rewrite: the "loss" value in
        the periodic log line comes from the accumulated-tensor path and
        the line carries the tok/s throughput.
        """
        args, trainer = self._make_trainer(tmp_path, max_iters=4, log_interval=2)
        try:
            trainer.train()
            out = capsys.readouterr().out
            logged = re.findall(r"iter (\d+): loss ([0-9]+(?:\.[0-9]+)?), .*tok/s", out)
            assert logged, "expected at least one train log line with tok/s"
            checked = 0
            for iter_str, loss_str in logged:
                if int(iter_str) == 0:
                    # First log fires before any batch is consumed.
                    continue
                loss_val = float(loss_str)
                assert math.isfinite(loss_val)
                assert loss_val > 0
                checked += 1
            assert checked >= 1
        finally:
            trainer.cleanup()

    def test_tok_per_sec_counts_microsteps(self):
        """Static: tokens must not be derived from ``running_count``.

        With accum>1 the token count must come from a per-micro-step
        token counter; the old ``batch_size * block_size * running_count``
        expression (tied to the logging/reset cadence) is banned.
        """
        from training.train_pretrain import PretrainTrainer

        src = inspect.getsource(PretrainTrainer.train)
        assert "* running_count" not in src


# ---------------------------------------------------------------------------
#  T022 (FR8): async checkpointing (CheckpointManager.save / flush)
# ---------------------------------------------------------------------------


class TestAsyncCheckpoint:
    """Tests for the async ``CheckpointManager.save`` (T022).

    Contract (design.md L291, L317): ``save(..., async_save: bool = True)``
    stages CPU deep copies on the caller thread, writes via a background
    thread (tmp file + ``os.replace``, atomicity kept), serialises
    concurrent saves, and falls back to a synchronous retry with a WARN
    on failure.  ``flush()`` blocks until in-flight writes finish;
    ``train()`` and ``cleanup()`` must drain pending writes before
    returning so callers can load files immediately afterwards.
    """

    @staticmethod
    def _make_manager(tmp_path):
        from training.trainer_base import CheckpointManager

        torch.manual_seed(11)
        cfg = tiny_modern_config(
            n_layer=1,
            n_head=4,
            n_embd=32,
            block_size=32,
            vocab_size=100,
            dropout=0.0,
        )
        model = ModernGPT(cfg)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        out_dir = str(Path(tmp_path) / "ckpt")
        os.makedirs(out_dir, exist_ok=True)
        manager = CheckpointManager(
            out_dir=out_dir,
            model=model,
            optimizer=optimizer,
            config={"model_type": "modern"},
        )
        return manager, model

    @staticmethod
    def _load(path):
        return torch.load(path, weights_only=False)

    def test_async_save_round_trip(self, tmp_path):
        """Default save is async: no TypeError, flush makes file loadable."""
        manager, model = self._make_manager(tmp_path)
        path = manager.save("latest_ckpt.pt", 5, 1.23)
        manager.flush()
        ckpt = self._load(path)
        assert ckpt["iter_num"] == 5
        for key in ("model", "optimizer", "best_val_loss"):
            assert key in ckpt
        assert list(Path(manager.out_dir).glob("*.tmp")) == []

    def test_sync_save_immediately_loadable(self, tmp_path):
        """async_save=False writes synchronously: loadable without flush."""
        manager, model = self._make_manager(tmp_path)
        path = manager.save("latest_ckpt.pt", 5, 1.23, async_save=False)
        ckpt = self._load(path)
        assert ckpt["iter_num"] == 5

    def test_staged_snapshot_isolation(self, tmp_path):
        """The staged snapshot is a deep copy, immune to later mutation.

        On CPU training ``tensor.cpu()`` is a no-op (same storage), so
        the staging step must actually clone the tensors.
        """
        manager, model = self._make_manager(tmp_path)
        pre_state = {k: v.clone() for k, v in model.state_dict().items()}
        path = manager.save("latest_ckpt.pt", 7, 1.0, async_save=True)
        with torch.no_grad():
            for param in model.parameters():
                param.data.add_(1.0)
        manager.flush()
        saved = self._load(path)["model"]
        for key, tensor in pre_state.items():
            assert torch.equal(saved[key], tensor)

    def test_serial_queue_two_saves(self, tmp_path):
        """Two back-to-back saves serialise and both land with own state."""
        manager, model = self._make_manager(tmp_path)
        path_a = manager.save("a.pt", 1, 1.0)
        path_b = manager.save("b.pt", 2, 1.0)
        manager.flush()
        assert self._load(path_a)["iter_num"] == 1
        assert self._load(path_b)["iter_num"] == 2

    def test_async_failure_falls_back_to_sync(self, tmp_path, monkeypatch):
        """Background torch.save failure -> WARN + synchronous retry.

        The flaky stub raises only on the first torch.save call and then
        delegates to the original, so the fallback retry succeeds.  The
        warnings.warn recorder delegates to the original as well; both
        patches are looked up at call time, so they also apply inside
        the background writer thread.
        """
        manager, model = self._make_manager(tmp_path)

        orig_save = torch.save
        failed = []

        def flaky_save(*args, **kwargs):
            if not failed:
                failed.append(True)
                raise RuntimeError("simulated background torch.save failure")
            return orig_save(*args, **kwargs)

        warned = []
        orig_warn = warnings.warn

        def recording_warn(*args, **kwargs):
            warned.append(args)
            return orig_warn(*args, **kwargs)

        monkeypatch.setattr(torch, "save", flaky_save)
        monkeypatch.setattr(warnings, "warn", recording_warn)

        path = manager.save("latest_ckpt.pt", 3, 1.0, async_save=True)
        manager.flush()

        assert os.path.exists(path)
        ckpt = self._load(path)
        assert ckpt["iter_num"] == 3
        assert len(warned) >= 1

    @staticmethod
    def _make_trainer(tmp_path, **overrides):
        from training.train_pretrain import PretrainTrainer

        data_dir = _write_tiny_data(tmp_path)
        out_dir = str(Path(tmp_path) / "out")
        run = dict(
            max_iters=2,
            eval_interval=1,
            batch_size=2,
            use_ema=False,
        )
        run.update(overrides)
        args = _make_args(data_dir, out_dir, **run)
        return args, PretrainTrainer(args)

    def test_train_flushes_before_return(self, tmp_path):
        """train() alone (no cleanup) leaves the checkpoint loadable.

        Pins "callers can load files right after train()": once saves
        are async by default, train() must flush pending writes before
        returning.
        """
        args, trainer = self._make_trainer(tmp_path)
        try:
            trainer.train()  # NOTE: deliberately no cleanup() call
            path = Path(args.out_dir) / "latest_ckpt.pt"
            ckpt = torch.load(path, weights_only=False)
            assert ckpt["iter_num"] == 2
        finally:
            # Release the memmap-backed dataset handles without going
            # through trainer.cleanup(); the contract under test is that
            # train() by itself leaves files readable.
            del trainer
            gc.collect()

    def test_cleanup_flushes_pending(self):
        """Static: BaseTrainer.cleanup drains pending async writes."""
        from training.trainer_base import BaseTrainer

        src = inspect.getsource(BaseTrainer.cleanup)
        assert "flush" in src.lower()
