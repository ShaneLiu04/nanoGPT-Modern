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
"""

import importlib
import sys
import warnings
from pathlib import Path

import pytest
import torch

from model.modern_gpt import ModernGPT, ModernGPTConfig


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
    cfg = ModernGPTConfig(
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
