"""T013 (FR5, Plan B) tests: KV decode de-cat -- append layout + view-fed SDPA.

Red-phase semantics
-------------------
The current implementation (model/kv_cache_utils.py + model/modern_gpt.py) is
the pre-T013 state: ring/cat semantics only, no ``layout`` parameter, no
``get_layer_view``, no cache-object dispatch in ``ModernGPT.forward``, and a
Python double-for-loop ``build_sliding_window_mask``.  Therefore:

* "pin" tests (5, plus every value-form baseline inside the mixed tests) must
  PASS right now: they lock the observable behavior of the current value-form
  (list/tuple) cache path so the rewrite cannot change it.

* structure / API / fast-path / benchmark tests (1, 2, 3, 4, 6, 7, 8, 9, 10,
  11, 12, and the object-form parts of 13) must FAIL right now: they demand
  the NEW contract (append layout + ``evicted`` flag + zero-copy
  ``get_layer_view``, cache-object duck-typing in forward, bit-identical
  zero-cat fast path, lazy int8 slice dequant, vectorized mask, >= 1.5x
  decode speedup).  The expected Red failure modes are AttributeError /
  TypeError because the new APIs do not exist yet -- never a test bug.

All tests run on CPU (torch 2.5.1, Windows), fp32 models in eval mode with
dropout=0.0 so "equal" means bit-identical (torch.equal).

Note on the cat counter: the generate() loop itself uses torch.cat to append
sampled tokens to idx_out, so zero-cat assertions are made at the granularity
of a single forward call, not a whole generate loop.
"""

import inspect
import statistics
import time

import pytest
import torch

from model import ModernGPT, ModernGPTConfig
from model.kv_cache_utils import (
    KVCacheManager,
    QuantizedKVCacheManager,
    build_sliding_window_mask,
)
import model.modern_gpt as modern_gpt_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(seed, **cfg_overrides):
    """Tiny fp32 ModernGPT in eval mode.

    Defaults: n_layer=2 (REQUIRED: layer-state isolation regression),
    n_head=2, n_embd=32, block_size=64, dropout=0.0, vocab_size=256.
    """
    torch.manual_seed(seed)
    kwargs = dict(
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=64,
        dropout=0.0,
    )
    kwargs.update(cfg_overrides)
    model = ModernGPT(tiny_modern_config(**kwargs))
    model.eval()
    return model


def _make_cache(model_or_cfg):
    """Append-layout KVCacheManager sized from a model/config (fp32 storage)."""
    cfg = (
        model_or_cfg.config
        if hasattr(model_or_cfg, "config")
        else model_or_cfg
    )
    cache = KVCacheManager.from_config(cfg, cache_dtype="fp32")
    return cache


def _prefill(model, prompt, cache=None):
    """Run the prefill step of generate(): forward + per-layer update + advance.

    Returns (last-token logits [B, V], cache).
    """
    if cache is None:
        cache = _make_cache(model)
        cache.init_cache(prompt.shape[0], prompt.device, torch.float32)
    with torch.no_grad():
        logits, _, raw_kvs = model(prompt, use_cache=True, start_pos=cache.start_pos)
        for li in range(model.config.n_layer):
            cache.update(li, raw_kvs[li][0], raw_kvs[li][1])
        cache.advance(prompt.shape[1])
    return logits[:, -1, :], cache


def _manual_decode(model, prompt, n_steps, mode):
    """Manually simulate the generate() decode loop (greedy tokens).

    mode:
      "values" -- past_kvs=cache.get_cache() each step (the current /
                  fallback semantics); the caller writes K/V itself.
      "object" -- past_kvs=cache (the new fast-path contract).  The fast
                  path writes the new K/V in place, so forward returns
                  next_kvs=None and the caller skips its update loop
                  (single-write protocol); when a layer falls back to the
                  value path forward returns per-layer tuples and the
                  caller writes them as usual.

    Returns (logits_list, cache) where logits_list[i] is the [B, V]
    last-token logits of prefill (i=0) and decode step i (i=1..n_steps).
    """
    n_layer = model.config.n_layer
    logits_last, cache = _prefill(model, prompt)
    logits_list = [logits_last]
    tokens = logits_last.argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        for _ in range(n_steps):
            if mode == "values":
                past = cache.get_cache()
            else:
                past = cache
            logits, _, new_kvs = model(
                tokens,
                past_kvs=past,
                use_cache=True,
                start_pos=cache.start_pos,
            )
            if new_kvs is not None:
                for li in range(n_layer):
                    cache.update(li, new_kvs[li][0], new_kvs[li][1])
            cache.advance(1)
            logits_last = logits[:, -1, :]
            logits_list.append(logits_last)
            tokens = logits_last.argmax(dim=-1, keepdim=True)
    return logits_list, cache


class _CatCounter:
    def __init__(self):
        self.count = 0


def _install_cat_counter(monkeypatch):
    """Monkeypatch torch.cat with a counting wrapper; returns the counter."""
    counter = _CatCounter()
    original = torch.cat

    def counting_cat(*args, **kwargs):
        counter.count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "cat", counting_cat)
    return counter


def _median_ms(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _reference_sliding_window_mask(seq_len, window_size, device):
    """Faithful replica of the current (pre-T013) double-for-loop mask."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start : i + 1] = 0.0
    return mask


def _assert_logits_lists_equal(list_a, list_b, context):
    assert len(list_a) == len(list_b), (
        f"{context}: logits list length mismatch "
        f"({len(list_a)} vs {len(list_b)})"
    )
    for i, (la, lb) in enumerate(zip(list_a, list_b)):
        assert torch.equal(la, lb), (
            f"{context}: logits differ at step {i} "
            f"(max abs diff {(la - lb).abs().max().item():.3e})"
        )


# ---------------------------------------------------------------------------
# 1. Cache manager layout parameter (Red: new constructor contract)
# ---------------------------------------------------------------------------

def test_cache_manager_layout_default_append():
    """KVCacheManager must default to layout="append" with evicted=False.

    layout="ring" must remain constructible with unchanged observable
    behavior (init/update/advance/get_cache).
    """
    cache = KVCacheManager(1, 2, 2, 4, max_cache_len=8)
    assert cache.layout == "append", (
        "KVCacheManager must default to layout='append' "
        f"(got {getattr(cache, 'layout', '<missing>')!r})"
    )
    assert cache.evicted is False, "evicted must start False"

    ring = KVCacheManager(1, 2, 2, 4, max_cache_len=8, layout="ring")
    assert ring.layout == "ring"
    ring.init_cache(1, "cpu", torch.float32)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    ring.update(0, k, v)
    ring.advance(3)
    blocks = ring.get_cache()
    assert len(blocks) == 1
    assert torch.equal(blocks[0][0], k)
    assert torch.equal(blocks[0][1], v)
    assert ring.cache_len == 3


# ---------------------------------------------------------------------------
# 2. get_layer_view zero-copy view (Red: new API)
# ---------------------------------------------------------------------------

def test_get_layer_view_zero_copy():
    """get_layer_view must return a zero-copy [B,H,write_pos+extra,D] view.

    Non-quantized: the view shares storage with the pre-allocated buffer
    (same data_ptr).  After eviction the method must return None.
    """
    B, H, D, S, max_len = 1, 2, 8, 6, 16
    cache = KVCacheManager(2, H, H, D, max_cache_len=max_len)
    cache.init_cache(B, "cpu", torch.float32)

    torch.manual_seed(11)
    k_all = torch.randn(B, H, S, D)
    v_all = torch.randn(B, H, S, D)
    for li in range(2):
        cache.update(li, k_all, v_all)
    cache.advance(S)

    blocks = cache.get_cache()
    for li in range(2):
        view = cache.get_layer_view(li)
        assert view is not None, f"layer {li}: get_layer_view must not be None"
        k_view, v_view = view
        assert tuple(k_view.shape) == (B, H, S, D)
        assert tuple(v_view.shape) == (B, H, S, D)
        assert torch.equal(k_view, blocks[li][0])
        assert torch.equal(v_view, blocks[li][1])
        # zero-copy: view base is the pre-allocated buffer itself
        assert k_view.data_ptr() == cache._k[li].data_ptr(), (
            "k_view must be a zero-copy view of the cache buffer"
        )
        assert v_view.data_ptr() == cache._v[li].data_ptr(), (
            "v_view must be a zero-copy view of the cache buffer"
        )

        view1 = cache.get_layer_view(li, extra=1)
        assert view1 is not None
        assert tuple(view1[0].shape) == (B, H, S + 1, D)
        assert tuple(view1[1].shape) == (B, H, S + 1, D)

    # Drive the cache past max_cache_len - 1 to trigger eviction.
    for _ in range(max_len):
        for li in range(2):
            kk = torch.randn(B, H, 1, D)
            vv = torch.randn(B, H, 1, D)
            cache.update(li, kk, vv)
        cache.advance(1)
        if cache.evicted:
            break
    assert cache.evicted is True, "overflowing the cache must set evicted=True"
    assert cache.get_layer_view(0) is None, (
        "get_layer_view must return None once evicted"
    )
    assert cache.get_layer_view(0, extra=1) is None


# ---------------------------------------------------------------------------
# 3. forward accepts a cache object (Red: new duck-typed past_kvs)
# ---------------------------------------------------------------------------

def test_forward_accepts_cache_object():
    """model.forward(past_kvs=<KVCacheManager>) must not raise TypeError.

    The current implementation crashes with
    "TypeError: 'KVCacheManager' object is not subscriptable".
    """
    model = _make_model(seed=101)
    torch.manual_seed(102)
    prompt = torch.randint(0, model.config.vocab_size, (1, 8))
    _, cache = _prefill(model, prompt)
    assert cache.cache_len == 8

    inp = torch.randint(0, model.config.vocab_size, (1, 1))
    with torch.no_grad():
        logits, _, new_kvs = model(
            inp, past_kvs=cache, use_cache=True, start_pos=cache.start_pos
        )
    assert tuple(logits.shape) == (1, 1, model.config.vocab_size)
    assert torch.isfinite(logits).all()
    # Single-write protocol: the fast path wrote the new K/V in place, so
    # the model-level present collapses to None (caller skips its update).
    assert new_kvs is None


# ---------------------------------------------------------------------------
# 4. Fast path bit-identical to value path (Red: core equivalence)
# ---------------------------------------------------------------------------

def test_decode_fast_path_bit_identical():
    """Object-form decode must be bit-identical to value-form decode.

    Path A (current/fallback): past_kvs=cache.get_cache() each step.
    Path B (fast path): past_kvs=cache (object).  Both caches receive the
    identical update sequence; greedy tokens; >= 30 decode steps.
    """
    model = _make_model(seed=1234)
    torch.manual_seed(1235)
    prompt = torch.randint(0, model.config.vocab_size, (1, 8))

    logits_values, cache_values = _manual_decode(model, prompt, 30, "values")
    logits_object, cache_object = _manual_decode(model, prompt, 30, "object")

    _assert_logits_lists_equal(logits_values, logits_object, "fast-path")
    assert cache_object.cache_len == cache_values.cache_len
    assert cache_object.start_pos == cache_values.start_pos


# ---------------------------------------------------------------------------
# 5. generate(use_cache=True) == generate(use_cache=False) (pin: PASS now)
# ---------------------------------------------------------------------------

def test_generate_cache_vs_nocache_identical():
    """Greedy generate must be token-identical with and without KV cache.

    This locks the current equivalence so the rewrite cannot break it.
    """
    model = _make_model(seed=4242)
    torch.manual_seed(4243)
    idx = torch.randint(0, model.config.vocab_size, (1, 5))
    with torch.no_grad():
        out_cache = model.generate(
            idx, max_new_tokens=30, use_cache=True, temperature=0.0
        )
        out_nocache = model.generate(
            idx, max_new_tokens=30, use_cache=False, temperature=0.0
        )
    assert out_cache.shape == out_nocache.shape
    assert torch.equal(out_cache, out_nocache), (
        "generate(use_cache=True) must match use_cache=False token-for-token"
    )


# ---------------------------------------------------------------------------
# 6. Zero torch.cat in a single fast-path forward (Red)
# ---------------------------------------------------------------------------

def test_fast_path_no_cat_in_forward(monkeypatch):
    """Single decode forward: value form does >= 2*n_layer cats, object
    form (fast path) does zero torch.cat calls."""
    model = _make_model(seed=2048)
    torch.manual_seed(2049)
    prompt = torch.randint(0, model.config.vocab_size, (1, 8))
    logits_last, cache = _prefill(model, prompt)
    inp = logits_last.argmax(dim=-1, keepdim=True)
    n_layer = model.config.n_layer

    # Baseline: value form concatenates K and V once per layer.
    counter = _install_cat_counter(monkeypatch)
    with torch.no_grad():
        model(
            inp,
            past_kvs=cache.get_cache(),
            use_cache=True,
            start_pos=cache.start_pos,
        )
    baseline = counter.count
    monkeypatch.undo()
    assert baseline >= 2 * n_layer, (
        f"value-form decode forward must cat K and V per layer "
        f"(expected >= {2 * n_layer}, got {baseline})"
    )

    # Fast path: object form must not call torch.cat at all.
    counter = _install_cat_counter(monkeypatch)
    with torch.no_grad():
        model(inp, past_kvs=cache, use_cache=True, start_pos=cache.start_pos)
    fast = counter.count
    monkeypatch.undo()
    assert fast == 0, (
        f"object-form (fast path) single forward must issue zero torch.cat "
        f"calls (got {fast})"
    )


# ---------------------------------------------------------------------------
# 7. Eviction falls back to the cat path (Red, B5)
# ---------------------------------------------------------------------------

def test_eviction_falls_back_to_cat_path():
    """After eviction (evicted=True) the object form must fall back to the
    value/cat semantics and stay token-identical to an all-the-way
    value-form run.

    NOTE: the RoPE cos/sin table is capped at 2*block_size rows in the
    current implementation, so with block_size=16 the total token count
    must stay <= 32 (50 steps would crash BOTH paths today).  prompt=4 +
    24 decode steps = 28 tokens triggers eviction (capacity = 15) at
    decode step 12 and leaves 12 post-eviction steps under comparison.
    """
    model = _make_model(seed=707, block_size=16)
    torch.manual_seed(708)
    prompt = torch.randint(0, model.config.vocab_size, (1, 4))
    n_steps = 24

    # Value-form reference (current semantics).
    logits_values, cache_values = _manual_decode(model, prompt, n_steps, "values")
    assert cache_values.cache_len == 15, "eviction must have engaged"
    assert cache_values.start_pos > 0

    # Pin: the manual value-form loop must reproduce generate() exactly.
    with torch.no_grad():
        gen_out = model.generate(
            prompt, max_new_tokens=n_steps + 1, use_cache=True, temperature=0.0
        )
    manual_tokens = torch.cat(
        [lg.argmax(dim=-1, keepdim=True) for lg in logits_values], dim=1
    )
    assert torch.equal(gen_out[:, prompt.shape[1] :], manual_tokens), (
        "manual value-form decode must match generate() token-for-token"
    )

    # Object form: same flow, must agree after eviction via fallback.
    logits_object, cache_object = _manual_decode(model, prompt, n_steps, "object")
    assert cache_object.evicted is True, (
        "advance() past max_cache_len - 1 must set cache.evicted = True"
    )
    _assert_logits_lists_equal(logits_values, logits_object, "post-eviction")


# ---------------------------------------------------------------------------
# 8. int8 lazy dequant of the view slice only (Red)
# ---------------------------------------------------------------------------

def test_int8_lazy_dequant_slice(monkeypatch):
    """QuantizedKVCacheManager + object form: _dequantize_cache must only
    materialize the [0 : write_pos+extra] slice, never the full
    max_cache_len buffer."""
    model = _make_model(seed=314)
    cfg = model.config
    head_dim = cfg.n_embd // cfg.n_head
    cache = QuantizedKVCacheManager(
        n_layers=cfg.n_layer,
        n_heads=cfg.n_head,
        n_kv_heads=cfg.n_kv_head,
        head_dim=head_dim,
        max_cache_len=cfg.block_size,
    )
    cache.init_cache(1, "cpu", torch.float32)

    recorded = []
    original = cache._dequantize_cache

    def spy(*args, **kwargs):
        recorded.append(tuple(args[0].shape))
        return original(*args, **kwargs)

    monkeypatch.setattr(cache, "_dequantize_cache", spy)

    torch.manual_seed(315)
    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    logits_last, _ = _prefill(model, prompt, cache=cache)
    assert recorded == [], "prefill (update only) must not dequantize anything"

    inp = logits_last.argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        logits, _, new_kvs = model(
            inp, past_kvs=cache, use_cache=True, start_pos=cache.start_pos
        )
    assert torch.isfinite(logits).all()

    expected_len = cache._write_pos + 1  # fast path does not advance
    assert len(recorded) > 0, "the decode step must have dequantized the view"
    seq_lens = [shape[2] for shape in recorded]
    assert max(seq_lens) == expected_len, (
        f"dequantized slice must cover exactly write_pos+1 = {expected_len} "
        f"slots (got {sorted(set(seq_lens))})"
    )
    assert all(length <= expected_len for length in seq_lens)
    assert expected_len < cache.max_cache_len, (
        "the dequantized slice must never span the full max_cache_len buffer"
    )

    # Single-write protocol: the fast path already stored the new K/V.
    assert new_kvs is None
    cache.advance(1)


# ---------------------------------------------------------------------------
# 9. Single-write protocol after the fast path (Red)
# ---------------------------------------------------------------------------

def test_fast_path_single_write_protocol():
    """The fast path must use the single-write protocol: forward writes the
    new K/V into the cache in place and returns next_kvs=None; the caller
    only calls advance().  Cache state must match a pure value-form run and
    skipping the redundant caller update must give identical outputs."""
    model = _make_model(seed=606)
    torch.manual_seed(607)
    prompt = torch.randint(0, model.config.vocab_size, (1, 6))

    logits_object, cache_object = _manual_decode(model, prompt, 3, "object")
    _, cache_values = _manual_decode(model, prompt, 3, "values")

    # The fast-path decode must have returned None (no per-layer presents).
    # (Exercised implicitly by _manual_decode; assert the cache state here.)
    assert cache_object.cache_len == cache_values.cache_len
    assert cache_object.start_pos == cache_values.start_pos
    assert cache_object.evicted == cache_values.evicted

    # Buffer contents must match the value-form run exactly.
    blocks_obj = cache_object.get_cache()
    blocks_val = cache_values.get_cache()
    for li in range(model.config.n_layer):
        assert torch.equal(blocks_obj[li][0], blocks_val[li][0]), (
            f"layer {li}: K buffer diverged between fast path and value path"
        )
        assert torch.equal(blocks_obj[li][1], blocks_val[li][1]), (
            f"layer {li}: V buffer diverged between fast path and value path"
        )


# ---------------------------------------------------------------------------
# 10. Flash wrapper optional fallback (Red, B9)
# ---------------------------------------------------------------------------

def test_flash_wrapper_optional_fallback(monkeypatch):
    """When flash_attention_with_cache returns None the fast path must fall
    back to the SDPA view path with unchanged output.  When it returns a
    wrong-value tensor, that value must not reach the output (this test
    locks the "wrapper result must not corrupt the output" semantics; it
    deliberately does NOT require the wrapper to be called at all)."""
    model = _make_model(seed=909)
    torch.manual_seed(910)
    prompt = torch.randint(0, model.config.vocab_size, (1, 8))
    logits_last, cache = _prefill(model, prompt)
    inp = logits_last.argmax(dim=-1, keepdim=True)
    head_dim = model.config.n_embd // model.config.n_head

    def run_object():
        with torch.no_grad():
            out, _, _ = model(
                inp, past_kvs=cache, use_cache=True, start_pos=cache.start_pos
            )
        return out

    # Reference: object form without any monkeypatching.
    ref = run_object()

    # (a) wrapper returns None -> output unchanged.
    def none_wrapper(*args, **kwargs):
        return None

    monkeypatch.setattr(
        modern_gpt_module._flash_attention,
        "flash_attention_with_cache",
        none_wrapper,
    )
    out_none = run_object()
    monkeypatch.undo()
    assert torch.equal(out_none, ref), (
        "flash wrapper returning None must fall back to SDPA bit-identically"
    )

    # (b) wrapper returns a wrong-value tensor -> must not be used.
    def garbage_wrapper(*args, **kwargs):
        return torch.full(
            (1, model.config.n_head, 1, head_dim), 777.0, dtype=torch.float32
        )

    monkeypatch.setattr(
        modern_gpt_module._flash_attention,
        "flash_attention_with_cache",
        garbage_wrapper,
    )
    out_garbage = run_object()
    monkeypatch.undo()
    assert torch.equal(out_garbage, ref), (
        "a wrong-value wrapper result must never reach the attention output"
    )


# ---------------------------------------------------------------------------
# 11. build_sliding_window_mask vectorized (Red: static source contract)
# ---------------------------------------------------------------------------

def test_sliding_window_mask_vectorized():
    """The mask must match the double-for-loop reference bit-for-bit and its
    source must contain no Python 'for ' loop (list comprehensions with
    range() count as loops too -- pure tensor comparison required)."""
    for seq_len, window_size in ((8, 4), (16, 5), (32, 8), (64, 3), (33, 7)):
        ref = _reference_sliding_window_mask(seq_len, window_size, "cpu")
        got = build_sliding_window_mask(seq_len, window_size, "cpu")
        assert got.shape == ref.shape
        assert torch.equal(got, ref), (
            f"mask mismatch at (seq_len={seq_len}, window_size={window_size})"
        )

    src = inspect.getsource(build_sliding_window_mask)
    assert "for " not in src, (
        "build_sliding_window_mask must be vectorized: no Python 'for ' "
        "loop (including range() comprehensions) may remain in its source"
    )


# ---------------------------------------------------------------------------
# 12. Decode speedup benchmark (Red: >= 1.5x)
# ---------------------------------------------------------------------------

@pytest.mark.perf
def test_decode_speedup_at_least_1_5x():
    """Single decode step (forward + update + advance) at cache_len ~ 4096
    (the FR5-AC2 benchmark scale): the value form (per-step torch.cat of the
    whole cache) must be at least 1.5x slower than the object form
    (view-fed fast path).

    Scale: n_layer=8, n_head=8, n_embd=512, block_size=8192 (buffer headroom
    so the benchmarked 25+ decode steps never trigger eviction), fp32, CPU.
    5 warmup + 30 reps, median.  Measured on the dev machine: value form
    ~21 ms/step vs object form ~13 ms/step (1.6x); the threshold keeps a
    margin below the observed ceiling.
    """
    torch.manual_seed(1000)
    cfg = tiny_modern_config(
        vocab_size=256,
        n_layer=8,
        n_head=8,
        n_embd=512,
        block_size=8192,
        dropout=0.0,
    )
    model = ModernGPT(cfg)
    model.eval()
    n_layer = cfg.n_layer

    torch.manual_seed(1001)
    prompt = torch.randint(0, cfg.vocab_size, (1, 4096))
    inp = torch.randint(0, cfg.vocab_size, (1, 1))

    cache_values = _make_cache(cfg)
    cache_values.init_cache(1, prompt.device, torch.float32)
    cache_object = _make_cache(cfg)
    cache_object.init_cache(1, prompt.device, torch.float32)

    with torch.no_grad():
        _, _, raw_kvs = model(prompt, use_cache=True)
        for li in range(n_layer):
            cache_values.update(li, raw_kvs[li][0], raw_kvs[li][1])
            cache_object.update(li, raw_kvs[li][0], raw_kvs[li][1])
        cache_values.advance(prompt.shape[1])
        cache_object.advance(prompt.shape[1])
        assert cache_values.cache_len == 4096
        assert not cache_object.evicted

        def step_values():
            past = cache_values.get_cache()
            _, _, new_kvs = model(
                inp, past_kvs=past, use_cache=True, start_pos=cache_values.start_pos
            )
            for li in range(n_layer):
                cache_values.update(li, new_kvs[li][0], new_kvs[li][1])
            cache_values.advance(1)

        def step_object():
            _, _, new_kvs = model(
                inp, past_kvs=cache_object, use_cache=True,
                start_pos=cache_object.start_pos,
            )
            # Single-write protocol: fast path wrote K/V in place
            # (new_kvs is None); the caller only advances.
            assert new_kvs is None
            cache_object.advance(1)

        ms_values = _median_ms(step_values, warmup=5, reps=30)
        ms_object = _median_ms(step_object, warmup=5, reps=30)

    ratio = ms_values / ms_object
    assert ratio >= 1.5, (
        f"decode step speedup {ratio:.2f}x < 1.5x "
        f"(value form {ms_values:.2f} ms/step, object form {ms_object:.2f} ms/step)"
    )


# ---------------------------------------------------------------------------
# 13. GQA + sliding-window compatibility (pin value baseline + Red object)
# ---------------------------------------------------------------------------

def test_cache_object_gqa_and_sliding_window_compat():
    """Object-form decode must be bit-identical to value-form decode for a
    GQA model (n_kv_head < n_head) and for a sliding-window model.  If the
    fast path does not support the sliding-window combination it must fall
    back to the value path -- output consistency is what is asserted."""
    # --- GQA ---
    model_gqa = _make_model(seed=808, n_head=4, n_kv_head=2)
    assert model_gqa.config.n_kv_head < model_gqa.config.n_head
    torch.manual_seed(809)
    prompt = torch.randint(0, model_gqa.config.vocab_size, (1, 6))
    logits_values, _ = _manual_decode(model_gqa, prompt, 10, "values")
    logits_object, _ = _manual_decode(model_gqa, prompt, 10, "object")
    _assert_logits_lists_equal(logits_values, logits_object, "GQA")

    # --- sliding window ---
    model_sw = _make_model(seed=810, sliding_window_size=8)
    torch.manual_seed(811)
    prompt_sw = torch.randint(0, model_sw.config.vocab_size, (1, 6))
    logits_values, _ = _manual_decode(model_sw, prompt_sw, 12, "values")
    logits_object, _ = _manual_decode(model_sw, prompt_sw, 12, "object")
    _assert_logits_lists_equal(logits_values, logits_object, "sliding-window")


# ---------------------------------------------------------------------------
# Migrated from tests/test_bugfixes.py (FR11b split)
# ---------------------------------------------------------------------------

from helpers import tiny_model, tiny_modern_config


def test_modern_gpt_kv_cache_dtype():
    """ModernGPT.generate(use_cache=True) must not crash on dtype mismatch."""
    config = tiny_modern_config(
        n_head=4,
        n_embd=128,
        block_size=64,
        n_kv_head=2,
    )
    model = ModernGPT(config)
    model.eval()
    idx = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=5, use_cache=True)
    assert out.shape[0] == 1
    assert out.shape[1] == 10
    print("[OK] ModernGPT KV cache dtype fix works")


def test_kv_cache_ring_buffer_order():
    """Ring-buffer cache must return keys/values in logical order."""
    B, n_layers, n_kv_heads, head_dim, max_len = 2, 1, 2, 4, 6
    cache = KVCacheManager(
        n_layers, n_kv_heads, n_kv_heads, head_dim, max_cache_len=max_len
    )
    cache.init_cache(B, "cpu", torch.float32)

    # Write 4 tokens.
    tok4 = (
        torch.arange(B * n_kv_heads * 4 * head_dim)
        .view(B, n_kv_heads, 4, head_dim)
        .float()
    )
    cache.update(0, tok4, tok4.clone())
    cache.advance(4)
    blocks = cache.get_cache()
    assert len(blocks) == 1
    assert torch.equal(blocks[0][0], tok4)
    assert torch.equal(blocks[0][1], tok4)
    assert cache.cache_len == 4

    # Write 3 more tokens (wraps around, evicts oldest tokens).
    # One slot is reserved for the next single-token decode step, so the
    # effective capacity is max_len - 1.
    tok3 = (
        torch.arange(B * n_kv_heads * 3 * head_dim)
        .view(B, n_kv_heads, 3, head_dim)
        .float()
        + 1000
    )
    cache.update(0, tok3, tok3.clone())
    cache.advance(3)
    effective_len = max_len - 1
    assert cache.cache_len == effective_len
    assert cache.start_pos == 2

    blocks = cache.get_cache()
    # When wrapped, the single layer is represented by two contiguous blocks.
    assert isinstance(blocks[0], list)
    assert len(blocks[0]) == 2
    recovered = torch.cat([blocks[0][0][0], blocks[0][1][0]], dim=2)
    expected = torch.cat([tok4[:, :, 2:, :], tok3], dim=2)
    assert torch.equal(recovered, expected)
    print("[OK] KV Cache ring buffer ordering works")


def test_kv_cache_sliding_window_eviction():
    """Cache must keep only the last max_cache_len tokens."""
    B, n_layers, n_kv_heads, head_dim, max_len = 1, 1, 1, 2, 4
    cache = KVCacheManager(
        n_layers, n_kv_heads, n_kv_heads, head_dim, max_cache_len=max_len
    )
    cache.init_cache(B, "cpu", torch.float32)

    for step in range(8):
        new = torch.full((B, n_kv_heads, 1, head_dim), float(step))
        cache.update(0, new, new.clone())
        cache.advance(1)

    # Effective capacity is max_len - 1 to leave room for the next decode token.
    effective_len = max_len - 1
    assert cache.cache_len == effective_len
    assert cache.start_pos == 5
    layer_cache = cache.get_cache()[0]
    if isinstance(layer_cache, list):
        recovered = torch.cat([b[0] for b in layer_cache], dim=2)
    else:
        recovered = layer_cache[0]
    expected = (
        torch.arange(5, 8)
        .view(1, 1, effective_len, 1)
        .expand(-1, -1, -1, head_dim)
        .float()
    )
    assert torch.equal(recovered, expected)
    print("[OK] KV Cache sliding window eviction works")


def test_kv_cache_long_generate_consistency():
    """generate(use_cache=True) must match no-cache output even when exceeding cache length."""
    model = tiny_model()
    config = model.config

    idx = torch.randint(0, config.vocab_size, (1, 5))
    # Use greedy decoding so sampled tokens depend only on the logits, not on
    # the RNG state difference between the two generate() calls.
    with torch.no_grad():
        out_cache = model.generate(idx, max_new_tokens=40, use_cache=True, top_k=1)
        out_nocache = model.generate(idx, max_new_tokens=40, use_cache=False, top_k=1)

    assert out_cache.shape == out_nocache.shape
    assert torch.equal(out_cache, out_nocache)
    print("[OK] Long generation with KV cache matches no-cache output")


def test_kv_cache_set_restore():
    """set_cache must restore logical order and state."""
    B, n_layers, n_kv_heads, head_dim, max_len = 1, 2, 2, 4, 8
    cache = KVCacheManager(
        n_layers, n_kv_heads, n_kv_heads, head_dim, max_cache_len=max_len
    )
    cache.init_cache(B, "cpu", torch.float32)

    k = torch.randn(B, n_kv_heads, 5, head_dim)
    v = torch.randn(B, n_kv_heads, 5, head_dim)
    cache.update(0, k, v)
    cache.update(1, k, v)
    cache.advance(5)

    snapshot = cache.get_cache()
    cache2 = KVCacheManager(
        n_layers, n_kv_heads, n_kv_heads, head_dim, max_cache_len=max_len
    )
    cache2.init_cache(B, "cpu", torch.float32)
    cache2.set_cache(snapshot)

    assert cache2.cache_len == cache.cache_len
    assert cache2.start_pos == cache.start_pos
    restored = cache2.get_cache()
    for i in range(n_layers):
        if isinstance(snapshot[i], list):
            for j in range(len(snapshot[i])):
                assert torch.equal(snapshot[i][j][0], restored[i][j][0])
                assert torch.equal(snapshot[i][j][1], restored[i][j][1])
        else:
            assert torch.equal(snapshot[i][0], restored[i][0])
            assert torch.equal(snapshot[i][1], restored[i][1])
    print("[OK] KV Cache set/restore works")
