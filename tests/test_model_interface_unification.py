"""FR12 (T019): BaselineGPT / ModernGPT interface unification.

Locks the v0.5.0 breaking contract:

- ``BaselineGPT.forward`` returns ``(logits, loss, None)`` -- same
  shape as ModernGPT (minus the KV cache slot, which is None).
- ``attention_mask`` is no longer silently dropped: a one-shot
  ``warnings.warn`` fires per model instance (B11).
- ``BaselineGPT.generate`` implements ``top_p`` (nucleus sampling);
  ``use_cache`` stays a no-op but warns explicitly instead of lying.
- Both models expose a ``supports_kv_cache`` capability attribute and
  the ``__code__.co_varnames`` reflection hacks are gone
  (run_full_evaluation + inference/generate), replaced by the
  capability flag.
- The 2-tuple compensation hack in train_pretrain disappears and no
  ``logits, loss =`` unpacking remains in trainers/evaluation/export.
- docs/CHANGELOG.md records the breaking change.
"""

import re
import warnings
from pathlib import Path

import pytest
import torch
from helpers import tiny_modern_config

from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from model.modern_gpt import ModernGPT

REPO_ROOT = Path(__file__).resolve().parents[1]


def _baseline(**overrides):
    cfg_kwargs = dict(n_layer=1, n_head=2, n_embd=32, block_size=16, vocab_size=100)
    cfg_kwargs.update(overrides)
    return BaselineGPT(BaselineGPTConfig(**cfg_kwargs))


def _read(rel):
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. forward 3-tuple
# ---------------------------------------------------------------------------


def test_baseline_forward_returns_3_tuple_with_targets():
    model = _baseline().eval()
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out = model(x, y)
    assert isinstance(out, tuple)
    assert len(out) == 3
    logits, loss, third = out
    assert logits.shape == (2, 8, 100)
    assert loss is not None and loss.requires_grad is False
    assert third is None


def test_baseline_forward_returns_3_tuple_without_targets():
    model = _baseline().eval()
    x = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, tuple)
    assert len(out) == 3
    logits, loss, third = out
    assert logits.shape == (2, 8, 100)
    assert loss is None
    assert third is None


def test_call_sites_unified_unpack_smoke():
    """Both models must unpack to length 3 for shared call sites."""
    bmodel = _baseline().eval()
    mmodel = ModernGPT(tiny_modern_config()).eval()
    x = torch.randint(0, 100, (2, 8))
    y = torch.randint(0, 100, (2, 8))
    for m in (bmodel, mmodel):
        with torch.no_grad():
            out = m(x, y)
        assert isinstance(out, tuple) and len(out) == 3


# ---------------------------------------------------------------------------
# 2. attention_mask one-shot warning (B11)
# ---------------------------------------------------------------------------


def test_baseline_attention_mask_warns_once():
    model = _baseline().eval()
    x = torch.randint(0, 100, (1, 8))
    mask = torch.ones(1, 8, dtype=torch.bool)

    with pytest.warns(UserWarning, match="attention_mask"):
        with torch.no_grad():
            model(x, attention_mask=mask)

    # second call: no further warning (one-shot per instance)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch.no_grad():
            model(x, attention_mask=mask)


def test_baseline_no_mask_no_warning():
    model = _baseline().eval()
    x = torch.randint(0, 100, (1, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with torch.no_grad():
            model(x)


# ---------------------------------------------------------------------------
# 3. generate: top_p implemented, use_cache explicit no-op
# ---------------------------------------------------------------------------


def test_baseline_generate_top_p_executes_and_keeps_greedy():
    """top_p must run through the nucleus path without breaking greedy."""
    model = _baseline().eval()
    idx = torch.randint(0, 100, (1, 4))
    with torch.no_grad():
        a = model.generate(idx, max_new_tokens=3, temperature=0.0)
        b = model.generate(idx, max_new_tokens=3, temperature=0.0, top_p=0.1)
    assert a.shape == (1, 7)
    assert b.shape == (1, 7)
    # greedy (temperature=0) argmax is always inside the nucleus
    assert torch.equal(a, b)


def test_baseline_generate_use_cache_warns():
    model = _baseline().eval()
    idx = torch.randint(0, 100, (1, 4))
    with pytest.warns(UserWarning, match="use_cache|KV"):
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=2, use_cache=True)
    assert out.shape == (1, 6)


def test_baseline_generate_top_p_source_implements_nucleus():
    """Static: the nucleus logic (sort + cumulative filter) exists."""
    src = _read("model/baseline_gpt.py")
    gen_start = src.find("def generate")
    gen_end = src.find("\n    def ", gen_start + 10)
    if gen_end < 0:
        gen_end = len(src)
    gen = src[gen_start:gen_end]
    # top_p stays in the signature and is actually referenced in the body
    assert "top_p" in gen
    body = gen[gen.find("):") :]
    assert "top_p" in body
    assert "cumsum" in body or "cumulative" in body


# ---------------------------------------------------------------------------
# 4. supports_kv_cache capability attribute + reflection removal
# ---------------------------------------------------------------------------


def test_supports_kv_cache_attribute():
    assert _baseline().supports_kv_cache is False
    assert ModernGPT(tiny_modern_config()).supports_kv_cache is True


def test_run_full_evaluation_uses_capability_flag():
    src = _read("run_full_evaluation.py")
    assert "co_varnames" not in src
    assert "supports_kv_cache" in src


def test_generate_py_uses_capability_flag():
    src = _read("inference/generate.py")
    assert '"use_cache" in model.generate' not in src
    assert "supports_kv_cache" in src


# ---------------------------------------------------------------------------
# 5. 2-tuple unpacking zero-residue (FR12-AC1)
# ---------------------------------------------------------------------------


def test_pretrain_pad_hack_removed():
    src = _read("training/train_pretrain.py")
    assert "logits, loss = self.model" not in src
    assert "return logits, loss, None" not in src


def test_no_two_tuple_unpack_in_business_code():
    """``logits, loss =`` must not appear in trainers/evaluation/export."""
    patterns = [
        "training/*.py",
        "evaluation/*.py",
        "tools/export/*.py",
        "tools/profiling/*.py",
        "tools/*.py",
        "inference/*.py",
    ]
    offenders = []
    for pattern in patterns:
        for p in REPO_ROOT.glob(pattern):
            if not p.is_file():
                continue
            src = p.read_text(encoding="utf-8")
            if re.search(r"logits,\s*loss\s*=", src):
                offenders.append(str(p.relative_to(REPO_ROOT)))
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# 6. CHANGELOG records the breaking change
# ---------------------------------------------------------------------------


def test_changelog_records_3_tuple_breaking():
    p = REPO_ROOT / "docs" / "CHANGELOG.md"
    assert p.exists()
    src = p.read_text(encoding="utf-8")
    assert "v0.5.0" in src
    assert "3-tuple" in src or "三" in src
    assert "BaselineGPT" in src
