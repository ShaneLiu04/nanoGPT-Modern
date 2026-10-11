"""GPU-rehearsal finding (AR-v0.5.0 ST leftover, rehearsal item 2).

During the KV VRAM-slope rehearsal on a Quadro RTX 5000 (sm75, fp32) the
decode-step transient memory grew linearly with the cache length at
12,335 bytes/token -- exactly 3 x [B, n_head, S, D] per layer per step.
Root cause: SDPA backends without a native GQA kernel (no flash on
sm75/fp32) materialise the ``enable_gqa`` head expansion internally,
copying K/V to the full query-head count and silently defeating the
zero-copy append-cache design (FR5-AC2 slope contract).

Fix contract: for single-token decode steps (T == 1) with GQA and no
additive mask, the attention must take a manual kernel that computes
scores at native KV-head resolution (``[B, n_kv, rep, S]`` -- 2*n_rep
times smaller than the expanded form) and never copies K/V.

Red-phase semantics
-------------------
``_decode_gqa_attention`` does not exist yet, so every test fails with
AttributeError -- never a test bug.
"""

import math

import torch
import torch.nn.functional as F

import model.modern_gpt as modern_gpt_module
from model import ModernGPT
from model.kv_cache_utils import KVCacheManager
from tests.helpers import tiny_modern_config
from tests.test_kv_append_layout import _make_cache, _prefill


def _make_gqa_model(seed, **cfg_overrides):
    """Tiny GQA ModernGPT (n_head=4, n_kv_head=2) in eval mode."""
    torch.manual_seed(seed)
    kwargs = dict(
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=64,
        dropout=0.0,
        n_kv_head=2,
    )
    kwargs.update(cfg_overrides)
    model = ModernGPT(tiny_modern_config(**kwargs))
    assert model.transformer.h[0].attn.use_gqa, "test requires a GQA model"
    model.eval()
    return model


def _sdpa_reference(q, k, v, scale):
    """Direct SDPA reference with enable_gqa (the pre-fix decode kernel)."""
    return F.scaled_dot_product_attention(q, k, v, scale=scale, enable_gqa=True)


# ---------------------------------------------------------------------------
# 1. Manual kernel exists and matches SDPA numerically (module level)
# ---------------------------------------------------------------------------


def test_decode_gqa_attention_matches_sdpa():
    """Manual kernel output must match SDPA enable_gqa within fp32 noise."""
    fn = getattr(modern_gpt_module, "_decode_gqa_attention", None)
    assert fn is not None, "_decode_gqa_attention must exist"
    torch.manual_seed(7)
    B, H, KV, D, S = 2, 4, 2, 16, 96
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, KV, S, D)
    v = torch.randn(B, KV, S, D)
    scale = 1.0 / math.sqrt(D)
    y_manual = fn(q, k, v, KV, scale=scale)
    y_sdpa = _sdpa_reference(q, k, v, scale)
    assert y_manual.shape == (B, H, 1, D)
    assert torch.allclose(
        y_manual, y_sdpa, atol=1e-5, rtol=1e-5
    ), f"max|diff|={(y_manual - y_sdpa).abs().max().item():.3e}"


def test_decode_gqa_attention_default_scale():
    """scale=None must default to 1/sqrt(head_dim) like SDPA."""
    fn = modern_gpt_module._decode_gqa_attention
    torch.manual_seed(8)
    B, H, KV, D, S = 1, 4, 2, 16, 32
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, KV, S, D)
    v = torch.randn(B, KV, S, D)
    y_default = fn(q, k, v, KV)
    y_scaled = fn(q, k, v, KV, scale=1.0 / math.sqrt(D))
    assert torch.equal(y_default, y_scaled)


def test_decode_gqa_attention_view_vs_cat_bit_identical():
    """Fast-path (view) and value-form (cat) K/V must give bit-identical
    outputs through the manual kernel -- this preserves the T013
    fast-path bit-identity contract."""
    fn = modern_gpt_module._decode_gqa_attention
    torch.manual_seed(9)
    B, H, KV, D, S = 1, 4, 2, 16, 64
    q = torch.randn(B, H, 1, D)
    k = torch.randn(B, KV, S, D)
    v = torch.randn(B, KV, S, D)
    # Same content, materialised via cat (value-form decode semantics).
    k_cat = torch.cat([k[:, :, :-1, :], k[:, :, -1:, :]], dim=2)
    v_cat = torch.cat([v[:, :, :-1, :], v[:, :, -1:, :]], dim=2)
    scale = 1.0 / math.sqrt(D)
    y_view = fn(q, k, v, KV, scale=scale)
    y_cat = fn(q, k_cat, v_cat, KV, scale=scale)
    assert torch.equal(y_view, y_cat)


# ---------------------------------------------------------------------------
# 2. The T == 1 decode forward dispatches to the manual kernel
# ---------------------------------------------------------------------------


def _install_kernel_spies(monkeypatch):
    """Spy on _decode_gqa_attention and F.scaled_dot_product_attention."""
    calls = {"manual": 0, "sdpa": 0}
    manual_fn = getattr(modern_gpt_module, "_decode_gqa_attention", None)
    assert manual_fn is not None, "_decode_gqa_attention must exist"
    monkeypatch.setattr(
        modern_gpt_module,
        "_decode_gqa_attention",
        lambda *a, **kw: (
            calls.__setitem__("manual", calls["manual"] + 1),
            manual_fn(*a, **kw),
        )[1],
        raising=True,
    )
    sdpa = F.scaled_dot_product_attention

    def counting_sdpa(*a, **kw):
        calls["sdpa"] += 1
        return sdpa(*a, **kw)

    monkeypatch.setattr(F, "scaled_dot_product_attention", counting_sdpa)
    return calls


def test_decode_forward_uses_manual_gqa_kernel(monkeypatch):
    """A T==1 GQA decode forward (both fast and value forms) must route
    attention through _decode_gqa_attention and make zero SDPA calls."""
    model = _make_gqa_model(101)
    torch.manual_seed(102)
    prompt = torch.randint(0, model.config.vocab_size, (1, 8))
    with torch.no_grad():
        _prefill(model, prompt)  # ensure rope/cache warm, no spies yet
    cache = _make_cache(model)
    cache.init_cache(1, prompt.device, torch.float32)
    logits_last, cache = _prefill(model, prompt, cache)
    tok = logits_last.argmax(dim=-1, keepdim=True)

    for mode in ("object", "values"):
        calls = _install_kernel_spies(monkeypatch)
        with torch.no_grad():
            model(
                tok,
                past_kvs=cache if mode == "object" else cache.get_cache(),
                use_cache=True,
                start_pos=cache.start_pos,
            )
        assert calls["manual"] == model.config.n_layer, (
            f"mode={mode}: expected manual kernel once per layer "
            f"({model.config.n_layer}), got {calls['manual']}"
        )
        assert calls["sdpa"] == 0, (
            f"mode={mode}: T==1 GQA decode must not call SDPA "
            f"(got {calls['sdpa']} calls -- head expansion would follow)"
        )
        monkeypatch.undo()
        cache.advance(1)


def test_prefill_forward_keeps_sdpa(monkeypatch):
    """T>1 prefill must keep the SDPA dispatch (flash/ring paths) and not
    use the manual decode kernel."""
    model = _make_gqa_model(103)
    torch.manual_seed(104)
    prompt = torch.randint(0, model.config.vocab_size, (1, 8))
    calls = _install_kernel_spies(monkeypatch)
    with torch.no_grad():
        model(prompt, use_cache=True, start_pos=0)
    assert calls["manual"] == 0, "prefill (T>1) must not use the decode kernel"
    assert calls["sdpa"] >= model.config.n_layer, "prefill must keep SDPA dispatch"


def test_masked_decode_keeps_sdpa(monkeypatch):
    """A T==1 step with an additive attention mask (e.g. batch padding)
    must keep the SDPA path: the manual kernel has no mask support."""
    model = _make_gqa_model(105)
    torch.manual_seed(106)
    tok = torch.randint(0, model.config.vocab_size, (1, 1))
    calls = _install_kernel_spies(monkeypatch)
    mask = torch.ones(1, 1, dtype=torch.bool)  # [B, T] padding mask
    with torch.no_grad():
        model(tok, use_cache=False, attention_mask=mask, start_pos=0)
    assert calls["manual"] == 0, "masked attention must keep the SDPA path"
    assert calls["sdpa"] >= model.config.n_layer


# ---------------------------------------------------------------------------
# 3. End-to-end contracts stay intact
# ---------------------------------------------------------------------------


def test_generate_cache_vs_nocache_gqa_identical():
    """Greedy generate with the manual decode kernel must stay
    token-identical to the no-cache full-recompute path (GQA model)."""
    model = _make_gqa_model(4242)
    torch.manual_seed(4243)
    idx = torch.randint(0, model.config.vocab_size, (1, 5))
    with torch.no_grad():
        out_cache = model.generate(idx, max_new_tokens=30, use_cache=True, temperature=0.0)
        out_nocache = model.generate(idx, max_new_tokens=30, use_cache=False, temperature=0.0)
    assert out_cache.shape == out_nocache.shape
    assert torch.equal(out_cache, out_nocache), (
        "generate(use_cache=True) must match use_cache=False token-for-token "
        "with the manual GQA decode kernel"
    )


def test_non_gqa_decode_unaffected(monkeypatch):
    """Non-GQA models (n_kv == n_head) keep the plain SDPA decode path."""
    torch.manual_seed(107)
    kwargs = dict(
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=64,
        dropout=0.0,
    )
    model = ModernGPT(tiny_modern_config(**kwargs))
    assert not model.transformer.h[0].attn.use_gqa
    model.eval()
    prompt = torch.randint(0, model.config.vocab_size, (1, 8))
    cache = KVCacheManager.from_config(model.config, cache_dtype="fp32")
    cache.init_cache(1, prompt.device, torch.float32)
    logits_last, cache = _prefill(model, prompt, cache)
    tok = logits_last.argmax(dim=-1, keepdim=True)
    calls = _install_kernel_spies(monkeypatch)
    with torch.no_grad():
        model(tok, past_kvs=cache, use_cache=True, start_pos=cache.start_pos)
    assert calls["manual"] == 0, "non-GQA decode must not use the GQA kernel"
    assert calls["sdpa"] == model.config.n_layer
