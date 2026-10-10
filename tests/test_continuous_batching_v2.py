"""T014 (FR7) Red-phase tests: iteration-level continuous batching rewrite.

Background (FR7)
----------------
The current ``inference/continuous_batching.py`` is a *fake* continuous
batching engine:
* batch-level recycling: ``ContinuousBatchScheduler.run`` waits for the
  whole batch to finish before back-filling from the queue;
* right-pad KV pollution: ``cache.advance(max_prompt_len)`` writes the pad
  positions' K/V into the cache and decode runs without any mask;
* ``PrefixCache`` (dead placeholder code) and ``_init_decode_compile``
  (never called) exist in the source;
* per-sequence Python sampling loop (``for b in range(B)``);
* off-by-one: the first decode step re-feeds the prompt's last token
  while the KV already contains the whole prompt, so the first generated
  token is the prediction of "tail token seen twice" instead of the
  prediction of the prompt itself.

Red-phase semantics
-------------------
* structure tests (1, 2) FAIL now: the new names (``GenerateResult``,
  ``run(max_steps=...)``, ``iteration_count``, ``threading.Lock``) do not
  exist yet and the dead code is still present.
* scheduler semantic tests (3-7, 9, 14 and the supplementary eos test)
  FAIL now: ``run`` returns a dict, re-feeds the prompt tail and pollutes
  the KV with right-pad positions.
* vectorized sampling tests (8) FAIL now: ``_sample_logits_batched`` does
  not exist.
* queue thread-safety (11): the structural lock assertion lives in test
  1 (Red); the behavioural part may pass under the GIL -- any result is
  recorded in the T014 report.
* empty-slot pin test (10) pins the *expected* behaviour; it may pass on
  the current implementation (result recorded in the T014 report).
* model forward extension tests (12, 13) FAIL now: ``positions`` and
  ``kv_mask`` are not accepted by ``ModernGPT.forward``.

Iteration contract (derivation used by test 6)
----------------------------------------------
One iteration = harvest FINISHED slots -> pop new requests into the
freed slots -> prefill the new-request group (left-aligned, right pads do
not enter the KV, per-sequence positions 0..pl-1, first token sampled
from the last real position's logits) -> every DECODING slot advances one
token -> vectorized sampling.  A request with max_tokens=N produces
exactly N completion tokens; the first comes from its prefill iteration,
so a request finishes in the iteration that samples its N-th token
(``finish_iteration``, 1-based), and a slot freed by harvest is refilled
in the *same* iteration's prefill phase (never by waiting for the rest of
the batch).

test 6 (batch_size=2; short/long/third all queued at t=0):
  it1      prefill short (t1) + long (t1)
  it2      decode: short t2, long t2
  it3      decode: short t3 (3/3 -> FINISHED, finish_iteration=3),
           long t3
  it4      harvest short -> pop third; prefill third t1 (1/1 ->
           FINISHED, finish_iteration=4); decode long t4
  it5-it8  decode long t5..t8 (8/8 -> FINISHED, finish_iteration=8)
  -> iteration_count == 8; long is not disturbed by short finishing.

Position semantics note (tests 12/13): in this codebase the implicit
absolute position of a new token is ``start_pos + past_len`` (see the
RoPE slicing in the block forward and the ``generate()`` cache path where
``start_pos`` counts evicted tokens).  Therefore the clean-slice
baselines below use ``start_pos=0`` with a past of length L_valid (the
new token lands at position L_valid), and the kv_mask runs pass explicit
``positions`` to place the new token at the same absolute position while
the padded tail of the KV buffer is masked out.

All tests run on CPU (torch 2.5.1), tiny randomly initialised models (no
training needed), ASCII-only source (Windows GBK console).
"""

import collections
import dataclasses
import inspect
import threading

import pytest
import torch

import inference.continuous_batching as cb
from inference.generate_utils import _sample_logits
from model.modern_gpt import ModernGPT, ModernGPTConfig

VOCAB = 64
# Out-of-vocab sentinel EOS id: greedy argmax is always < VOCAB, so a
# request carrying this eos_token_id is never truncated by EOS.
SENTINEL_EOS = VOCAB + 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_model(seed=0, position_encoding="rope"):
    """Tiny fp32 ModernGPT in eval mode (2 layers, 4 heads, 32 dim)."""
    torch.manual_seed(seed)
    cfg = ModernGPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=128,
        vocab_size=VOCAB,
        dropout=0.0,
        position_encoding=position_encoding,
    )
    model = ModernGPT(cfg)
    model.eval()
    return model


def _make_request(
    prompt,
    max_tokens,
    request_id,
    temperature=0.0,
    top_k=None,
    top_p=None,
    eos_token_id=None,
    repetition_penalty=None,
):
    return cb.Request(
        prompt_ids=list(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=eos_token_id,
        repetition_penalty=repetition_penalty,
        request_id=request_id,
    )


def _run_scheduler(model, requests, batch_size):
    queue = cb.RequestQueue()
    for req in requests:
        queue.enqueue(req)
    scheduler = cb.ContinuousBatchScheduler(
        queue, model, device="cpu", batch_size=batch_size
    )
    results = scheduler.run()
    return scheduler, results


def _generate_reference(model, prompt, max_tokens):
    """Greedy single-sequence reference: model.generate, no EOS stop."""
    idx = torch.tensor([list(prompt)], dtype=torch.long)
    out = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=0.0,
        top_k=None,
        top_p=None,
        use_cache=True,
        eos_token_id=None,
    )
    return out[0, len(prompt):].tolist()


def _completion_of(result, prompt):
    """Assert the prompt prefix is intact and return the completion."""
    assert result.token_ids[: len(prompt)] == list(prompt), (
        f"request {result.request_id}: token_ids must start with the "
        f"prompt (got {result.token_ids[: len(prompt)]}, "
        f"expected {list(prompt)})"
    )
    return result.token_ids[len(prompt):]


# ---------------------------------------------------------------------------
# 1. Dead code removed + queue lock (Red: structure)
# ---------------------------------------------------------------------------


def test_prefix_cache_and_dead_code_removed():
    """The rewritten module must not contain PrefixCache / BatchGenerator /
    _init_decode_compile, and the queue must be protected by
    threading.Lock."""
    src = inspect.getsource(cb)
    assert "class PrefixCache" not in src, (
        "PrefixCache (dead placeholder code) must be deleted from "
        "inference/continuous_batching.py"
    )
    assert "class BatchGenerator" not in src, (
        "BatchGenerator must be deleted from "
        "inference/continuous_batching.py"
    )
    assert "_init_decode_compile" not in src, (
        "_init_decode_compile (never called) must be deleted from "
        "inference/continuous_batching.py"
    )
    assert "threading.Lock" in src, (
        "RequestQueue must protect its internal state with "
        "threading.Lock"
    )


# ---------------------------------------------------------------------------
# 2. GenerateResult dataclass + run(max_steps) + iteration_count (Red)
# ---------------------------------------------------------------------------


def test_generate_result_and_run_signature():
    """GenerateResult is a dataclass with the four contract fields;
    ContinuousBatchScheduler(queue, model, device=..., batch_size=...) is
    constructible, run() has a max_steps parameter, the scheduler exposes
    iteration_count, and a single greedy request runs to 'length'."""
    assert hasattr(cb, "GenerateResult"), (
        "inference.continuous_batching must define GenerateResult"
    )
    generate_result_cls = cb.GenerateResult
    assert dataclasses.is_dataclass(generate_result_cls), (
        "GenerateResult must be a dataclass"
    )
    field_names = {f.name for f in dataclasses.fields(generate_result_cls)}
    assert {
        "request_id",
        "token_ids",
        "finish_reason",
        "finish_iteration",
    } <= field_names, (
        f"GenerateResult fields must include request_id/token_ids/"
        f"finish_reason/finish_iteration (got {sorted(field_names)})"
    )
    gr = generate_result_cls(
        request_id="x",
        token_ids=[1, 2, 3],
        finish_reason="length",
        finish_iteration=1,
    )
    assert gr.request_id == "x"
    assert gr.token_ids == [1, 2, 3]
    assert gr.finish_reason == "length"
    assert gr.finish_iteration == 1

    model = _tiny_model(seed=41)
    prompt = [3, 17, 5, 42, 9, 11]
    queue = cb.RequestQueue()
    queue.enqueue(
        _make_request(prompt, max_tokens=4, request_id="r0", eos_token_id=None)
    )
    scheduler = cb.ContinuousBatchScheduler(
        queue, model, device="cpu", batch_size=2
    )
    assert "max_steps" in inspect.signature(scheduler.run).parameters, (
        "ContinuousBatchScheduler.run must accept a max_steps parameter"
    )
    assert hasattr(scheduler, "iteration_count"), (
        "ContinuousBatchScheduler must expose an iteration_count attribute"
    )

    results = scheduler.run()
    assert isinstance(results, list), (
        "run() must return List[GenerateResult], not a dict"
    )
    assert len(results) == 1
    res = results[0]
    assert res.request_id == "r0"
    assert len(res.token_ids) == len(prompt) + 4, (
        "max_tokens=4 must produce exactly 4 completion tokens "
        f"(token_ids len {len(res.token_ids)}, prompt len {len(prompt)})"
    )
    _completion_of(res, prompt)
    assert res.finish_reason == "length"
    assert res.finish_iteration >= 1
    assert scheduler.iteration_count >= 1
    assert scheduler.iteration_count == res.finish_iteration, (
        "single request without queuing: the finishing iteration is the "
        f"total iteration count (finish_iteration={res.finish_iteration}, "
        f"iteration_count={scheduler.iteration_count})"
    )


# ---------------------------------------------------------------------------
# 3. Single request: scheduler == model.generate (Red, off-by-one capture)
# ---------------------------------------------------------------------------


def test_single_request_matches_generate():
    """B=1 greedy: the scheduler completion must equal the reference
    model.generate completion token-for-token.  The current
    implementation re-feeds the prompt tail on the first decode step, so
    its first token is wrong."""
    model = _tiny_model(seed=7)
    prompt = [5, 12, 33, 2, 40]
    max_tokens = 6
    scheduler, results = _run_scheduler(
        model,
        [_make_request(prompt, max_tokens=max_tokens, request_id="s")],
        batch_size=1,
    )
    assert isinstance(results, list) and len(results) == 1
    completion = _completion_of(results[0], prompt)
    expected = _generate_reference(model, prompt, max_tokens)
    assert completion == expected, (
        f"single-request greedy completion must match model.generate "
        f"(got {completion}, expected {expected})"
    )


# ---------------------------------------------------------------------------
# 4. Prefill first token: no off-by-one (Red)
# ---------------------------------------------------------------------------


def test_prefill_first_token_no_offbyone():
    """max_tokens=1: the scheduler's first (and only) completion token
    must be the greedy prediction of the prompt -- both for a single
    request and for each member of a two-prompt batch with unequal
    lengths.  The current implementation re-feeds the prompt tail with
    the whole prompt already in the KV, so its first token is the
    prediction of the tail token seen twice."""
    model = _tiny_model(seed=13)
    p_short = [2, 9, 44]
    p_long = [7, 30, 1, 55, 18, 62, 3]
    prompts = [p_short, p_long]
    scheduler, results = _run_scheduler(
        model,
        [
            _make_request(p, max_tokens=1, request_id=f"r{i}")
            for i, p in enumerate(prompts)
        ],
        batch_size=2,
    )
    assert isinstance(results, list) and len(results) == 2
    by_id = {r.request_id: r for r in results}
    for i, p in enumerate(prompts):
        completion = _completion_of(by_id[f"r{i}"], p)
        expected_first = _generate_reference(model, p, 1)[0]
        assert len(completion) == 1
        assert completion[0] == expected_first, (
            f"request r{i} (prompt len {len(p)}): first completion token "
            f"must be the greedy prediction of the prompt tail "
            f"(got {completion[0]}, expected {expected_first})"
        )


# ---------------------------------------------------------------------------
# 5. Left-padded batch == separate generation (Red, KV pollution capture)
# ---------------------------------------------------------------------------


def test_left_padded_batch_matches_separate():
    """Unequal-length prompts (3/7/11) batched together must produce the
    same completions as generating each prompt separately.  The current
    implementation writes right-pad positions into the KV cache and
    decodes without a mask, so batched outputs diverge from the separate
    runs."""
    model = _tiny_model(seed=99)
    prompts = [
        [1, 2, 3],
        [4, 5, 6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    ]
    max_tokens = 6
    scheduler, results = _run_scheduler(
        model,
        [
            _make_request(p, max_tokens=max_tokens, request_id=f"r{i}",
                          eos_token_id=SENTINEL_EOS)
            for i, p in enumerate(prompts)
        ],
        batch_size=3,
    )
    assert isinstance(results, list) and len(results) == 3
    by_id = {r.request_id: r for r in results}
    for i, p in enumerate(prompts):
        completion = _completion_of(by_id[f"r{i}"], p)
        expected = _generate_reference(model, p, max_tokens)
        assert completion == expected, (
            f"request r{i} (prompt len {len(p)}): batched greedy "
            f"completion must match the separate generate run "
            f"(got {completion}, expected {expected})"
        )


# ---------------------------------------------------------------------------
# 6. Slot recycling at iteration boundaries (Red, AC1 core)
# ---------------------------------------------------------------------------


def test_slot_recycling_iteration_boundaries():
    """batch_size=2 with three queued requests: the third request must
    enter at the iteration boundary right after 'short' finishes -- not
    by waiting for 'long' (which static batching would require).

    Iteration derivation (see module docstring):
      it1  prefill short t1 + long t1
      it2  short t2, long t2
      it3  short t3 (3/3 FINISHED, finish_iteration=3), long t3
      it4  harvest short -> pop third; prefill third t1 (1/1 FINISHED,
           finish_iteration=4); decode long t4
      it5-it8  long t5..t8 (8/8 FINISHED, finish_iteration=8)
    """
    model = _tiny_model(seed=2024)
    prompt = [8, 21, 4, 33]
    requests = [
        _make_request(prompt, max_tokens=3, request_id="short"),
        _make_request(prompt, max_tokens=8, request_id="long"),
        _make_request(prompt, max_tokens=1, request_id="third"),
    ]
    queue = cb.RequestQueue()
    for req in requests:
        queue.enqueue(req)
    scheduler = cb.ContinuousBatchScheduler(
        queue, model, device="cpu", batch_size=2
    )
    results = scheduler.run()
    assert isinstance(results, list)
    by_id = {r.request_id: r for r in results}
    assert set(by_id) == {"short", "long", "third"}, (
        "all three requests must complete"
    )
    assert by_id["short"].finish_iteration == 3, (
        "short (max_tokens=3) must finish in iteration 3 "
        f"(got {by_id['short'].finish_iteration})"
    )
    assert by_id["third"].finish_iteration == 4, (
        "third must enter in the iteration after short finishes (it4) "
        "and finish immediately (max_tokens=1); got "
        f"{by_id['third'].finish_iteration}"
    )
    assert by_id["long"].finish_iteration == 8, (
        "long (max_tokens=8) must finish in iteration 8 "
        f"(got {by_id['long'].finish_iteration})"
    )
    assert scheduler.iteration_count == 8, (
        "the whole run must take exactly 8 iterations "
        f"(got {scheduler.iteration_count})"
    )
    assert len(_completion_of(by_id["long"], prompt)) == 8, (
        "short finishing early must not disturb long's completion"
    )
    assert len(_completion_of(by_id["short"], prompt)) == 3
    assert len(_completion_of(by_id["third"], prompt)) == 1
    assert all(r.finish_reason == "length" for r in results)


# ---------------------------------------------------------------------------
# 7. run(max_steps=k) partial semantics (Red)
# ---------------------------------------------------------------------------


def test_max_steps_partial():
    """run(max_steps=5) on two long requests returns no finished results
    and counts exactly 5 iterations; a fresh full run completes both with
    exactly max_tokens completion tokens."""
    model = _tiny_model(seed=55)
    prompt = [10, 22, 31, 4]

    queue = cb.RequestQueue()
    for i in range(2):
        queue.enqueue(
            _make_request(prompt, max_tokens=20, request_id=f"r{i}")
        )
    scheduler = cb.ContinuousBatchScheduler(
        queue, model, device="cpu", batch_size=4
    )
    partial = scheduler.run(max_steps=5)
    assert isinstance(partial, list) and len(partial) == 0, (
        "after 5 iterations of max_tokens=20 requests nothing is "
        "FINISHED, so run(max_steps=5) must return an empty list"
    )
    assert scheduler.iteration_count == 5

    # Scheduler reuse-after-partial is not part of the contract; use a
    # fresh queue and a fresh scheduler for the full run.
    queue2 = cb.RequestQueue()
    for i in range(2):
        queue2.enqueue(
            _make_request(prompt, max_tokens=20, request_id=f"r{i}")
        )
    scheduler2 = cb.ContinuousBatchScheduler(
        queue2, model, device="cpu", batch_size=4
    )
    full = scheduler2.run(max_steps=None)
    assert isinstance(full, list) and len(full) == 2
    for res in full:
        assert len(_completion_of(res, prompt)) == 20
        assert res.finish_reason == "length"


# ---------------------------------------------------------------------------
# 8. Vectorized sampling: _sample_logits_batched (Red)
# ---------------------------------------------------------------------------


def test_vectorized_sampling_mixed_temperatures():
    """Unit-level contract of _sample_logits_batched:
    temperature==0 rows are strict argmax and unaffected by top_k/top_p;
    top_k=1 and top_p=0.0 force argmax at temperature 1; <=0 top_k and
    >=1 top_p mean "no filtering"; temp==0 rows agree with the per-row
    _sample_logits reference.  RNG-dependent rows are only checked for
    legal value range (batched and per-row multinomial consume RNG
    differently, so bit equality is not assertable there)."""
    from inference.generate_utils import _sample_logits_batched

    torch.manual_seed(77)
    V = 32
    logits = torch.randn(4, V)
    argmax = logits.argmax(dim=-1)

    # Mixed temperatures: rows 0/1 (temp 0) must be argmax even with
    # top_k=3 passed; rows 2/3 (temp > 0) only need legal token ids.
    temps = torch.tensor([0.0, 0.0, 1.0, 0.5])
    out = _sample_logits_batched(logits, temps, top_ks=torch.full((4,), 3))
    assert out.shape == (4,)
    assert out.dtype == torch.long
    assert out[0].item() == argmax[0].item(), (
        "temperature=0 rows must be strict argmax regardless of top_k"
    )
    assert out[1].item() == argmax[1].item()
    assert all(0 <= int(t) < V for t in out[2:].tolist())

    # top_k=1 at temperature 1 keeps only the top-1 token -> argmax.
    out_k1 = _sample_logits_batched(
        logits, torch.ones(4), top_ks=torch.ones(4, dtype=torch.long)
    )
    assert out_k1.tolist() == argmax.tolist()

    # top_p=0.0 at temperature 1: cumulative prob > 0 masks every sorted
    # position except the first, so only the argmax survives.
    out_p0 = _sample_logits_batched(logits, torch.ones(4), top_ps=torch.zeros(4))
    assert out_p0.tolist() == argmax.tolist()

    # top_k <= 0 and top_p >= 1.0 mean "no filtering": no exception, and
    # every sampled id is inside the vocab.
    out_nofilter = _sample_logits_batched(
        logits,
        torch.ones(4),
        top_ks=torch.zeros(4, dtype=torch.long),
        top_ps=torch.ones(4),
    )
    assert all(0 <= int(t) < V for t in out_nofilter.tolist())

    # Per-row reference consistency with _sample_logits: temperature=0
    # rows must agree bit-for-bit; temperature>0 rows only need to be
    # legal (RNG consumption order differs between the batched and the
    # per-row paths).
    for b in range(4):
        ref = _sample_logits(
            logits[b : b + 1], temperature=float(temps[b])
        )
        assert 0 <= int(ref[0]) < V
        if float(temps[b]) == 0.0:
            assert out[b].item() == int(ref[0]), (
                f"row {b}: temperature=0 rows must match the per-row "
                "_sample_logits argmax"
            )

    # --- integration: mixed-temperature batch through the scheduler ---
    model = _tiny_model(seed=88)
    prompts = [[3, 14, 15], [9, 2, 6, 5], [35, 8, 27, 1, 30]]
    temps_list = [0.0, 1.0, 0.7]
    max_tokens = 5

    def make_requests():
        return [
            _make_request(
                p, max_tokens=max_tokens, request_id=f"r{i}",
                temperature=t,
            )
            for i, (p, t) in enumerate(zip(prompts, temps_list))
        ]

    torch.manual_seed(123)
    _, results_a = _run_scheduler(model, make_requests(), batch_size=3)
    torch.manual_seed(123)
    _, results_b = _run_scheduler(model, make_requests(), batch_size=3)
    assert isinstance(results_a, list) and len(results_a) == 3
    by_a = {r.request_id: r for r in results_a}
    by_b = {r.request_id: r for r in results_b}
    for i, (p, t) in enumerate(zip(prompts, temps_list)):
        comp_a = _completion_of(by_a[f"r{i}"], p)
        comp_b = _completion_of(by_b[f"r{i}"], p)
        assert comp_a == comp_b, (
            f"row {i}: identical seed must reproduce identical "
            f"completions (got {comp_a} vs {comp_b})"
        )
        if t == 0.0:
            expected = _generate_reference(model, p, max_tokens)
            assert comp_a == expected, (
                f"row {i}: temperature=0 row must equal the separate "
                f"greedy generate run (got {comp_a}, expected {expected})"
            )
        else:
            assert all(0 <= tok < VOCAB for tok in comp_a)


# ---------------------------------------------------------------------------
# 9. Greedy determinism (Red)
# ---------------------------------------------------------------------------


def test_greedy_determinism():
    """Two independent greedy runs (fresh schedulers) must produce
    bit-identical results (inherits the T006 determinism semantics)."""
    model = _tiny_model(seed=64)
    prompts = [[2, 7, 11], [20, 5, 33, 9, 1]]

    def run_once():
        scheduler, results = _run_scheduler(
            model,
            [
                _make_request(p, max_tokens=4, request_id=f"r{i}")
                for i, p in enumerate(prompts)
            ],
            batch_size=2,
        )
        assert isinstance(results, list)
        return {r.request_id: r.token_ids for r in results}

    out1 = run_once()
    out2 = run_once()
    assert set(out1) == set(out2) == {"r0", "r1"}
    assert out1 == out2, (
        "greedy (temperature=0) scheduling must be deterministic across "
        "independent runs"
    )


# ---------------------------------------------------------------------------
# 10. Empty slots must not contaminate (B12 pin)
# ---------------------------------------------------------------------------


def test_empty_slot_no_contamination():
    """Pin: batch_size=4 with 3 requests must give the same completions
    as batch_size=3 (an empty slot must not perturb the others); and
    after an early-finishing request is recycled, the remaining requests
    must continue exactly as if they had run alone from the start.

    This is a *pin* test, so it tolerates both the legacy dict return
    (request_id -> full token id list) and the new List[GenerateResult]:
    the contamination semantics, not the return type, are what is locked
    here (the return type is locked by test 2)."""
    model = _tiny_model(seed=31)
    prompts = [[6, 13, 27], [40, 2, 19, 8], [55, 1]]
    max_tokens = 5

    def completions_by_id(results, prompts_by_id):
        if isinstance(results, dict):
            raw = {rid: list(tids) for rid, tids in results.items()}
        else:
            raw = {r.request_id: list(r.token_ids) for r in results}
        out = {}
        for rid, prompt in prompts_by_id.items():
            assert raw[rid][: len(prompt)] == list(prompt), (
                f"request {rid}: token_ids must start with the prompt"
            )
            out[rid] = raw[rid][len(prompt):]
        return out

    def completions(batch_size, requests):
        scheduler, results = _run_scheduler(model, requests, batch_size)
        prompts_by_id = {f"p{i}": p for i, p in enumerate(prompts)}
        return completions_by_id(results, prompts_by_id)

    reqs = [
        _make_request(p, max_tokens=max_tokens, request_id=f"p{i}")
        for i, p in enumerate(prompts)
    ]
    c4 = completions(4, reqs)
    c3 = completions(3, [
        _make_request(p, max_tokens=max_tokens, request_id=f"p{i}")
        for i, p in enumerate(prompts)
    ])
    assert set(c4) == set(c3) == {"p0", "p1", "p2"}
    assert c4 == c3, (
        "one empty slot (batch_size=4 with 3 requests) must not change "
        "any completion"
    )
    assert all(
        0 <= t < VOCAB for comp in c4.values() for t in comp
    ), "token ids must be legal (no NaN-driven garbage)"

    # Variant: p0 finishes after 1 token; p1/p2 must then continue as if
    # they had run alone with batch_size=2 from the very start.
    mix_requests = [
        _make_request(prompts[0], max_tokens=1, request_id="p0"),
        _make_request(prompts[1], max_tokens=max_tokens, request_id="p1"),
        _make_request(prompts[2], max_tokens=max_tokens, request_id="p2"),
    ]
    scheduler, results = _run_scheduler(model, mix_requests, batch_size=3)
    assert isinstance(results, list)
    mix = {r.request_id: r for r in results}
    assert set(mix) == {"p0", "p1", "p2"}

    pure_requests = [
        _make_request(prompts[1], max_tokens=max_tokens, request_id="p1"),
        _make_request(prompts[2], max_tokens=max_tokens, request_id="p2"),
    ]
    _, pure_results = _run_scheduler(model, pure_requests, batch_size=2)
    assert isinstance(pure_results, list)
    pure = {r.request_id: r for r in pure_results}
    assert set(pure) == {"p1", "p2"}
    assert _completion_of(mix["p1"], prompts[1]) == _completion_of(
        pure["p1"], prompts[1]
    ), "slot recycling must leave no residual contamination (p1)"
    assert _completion_of(mix["p2"], prompts[2]) == _completion_of(
        pure["p2"], prompts[2]
    ), "slot recycling must leave no residual contamination (p2)"


# ---------------------------------------------------------------------------
# 11. RequestQueue thread safety (Red)
# ---------------------------------------------------------------------------


def test_request_queue_thread_safety():
    """Behavioural thread safety: 8 threads x 25 enqueues -> 200 queued;
    4 concurrent drainers dequeue exactly the 200 requests with no
    duplicates or losses; a full queue (maxsize=2) rejects the third
    enqueue with RuntimeError.  (The structural threading.Lock
    assertion lives in test 1.)"""
    queue = cb.RequestQueue(maxsize=1000)
    errors = []

    def enqueue_worker(tid):
        try:
            for i in range(25):
                queue.enqueue(
                    _make_request([tid, i], max_tokens=1,
                                  request_id=f"t{tid}-{i}")
                )
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [
        threading.Thread(target=enqueue_worker, args=(t,)) for t in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent enqueue raised: {errors}"
    assert len(queue) == 200

    collected = []
    collect_lock = threading.Lock()

    def drainer():
        try:
            while True:
                req = queue.dequeue()
                if req is None:
                    return
                with collect_lock:
                    collected.append(req.request_id)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=drainer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, (
        f"concurrent dequeue raised (check-then-pop race without a "
        f"lock): {errors}"
    )
    assert len(collected) == 200, (
        f"expected all 200 requests to be dequeued (got {len(collected)})"
    )
    counts = collections.Counter(collected)
    assert all(v == 1 for v in counts.values()), (
        "no request_id may be duplicated or lost across concurrent "
        "drainers"
    )
    assert queue.is_empty()

    small = cb.RequestQueue(maxsize=2)
    small.enqueue(_make_request([1], max_tokens=1, request_id="a"))
    small.enqueue(_make_request([2], max_tokens=1, request_id="b"))
    with pytest.raises(RuntimeError):
        small.enqueue(_make_request([3], max_tokens=1, request_id="c"))


# ---------------------------------------------------------------------------
# 12. ModernGPT.forward positions semantics (Red)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position_encoding", ["rope", "learned"])
def test_forward_positions_semantics(position_encoding):
    """positions: [B, T] per-token absolute positions (0-based).  When
    given, wpe/RoPE index by it instead of the implicit arange.

    (a) identity positions == no positions;
    (b1) explicit offset positions == the implicit offset obtained from
         start_pos=2 (this codebase's implicit position of token j is
         start_pos + past_len + j; no past here);
    (b2) offset equivalence with two leading dummy tokens: the logits of
         x placed after 2 masked-out dummy slots equal the logits of x
         fed alone with positions arange(2, 2+T).  NOTE: the contract's
         literal form (unmasked dummies) cannot hold -- causal attention
         would let x see the dummies -- so the dummy slots are masked
         with the standard attention_mask (True=attend), which keeps the
         intent intact: explicit positions must reproduce the absolute
         position arithmetic of a real offset sequence.
    """
    model = _tiny_model(seed=17, position_encoding=position_encoding)
    torch.manual_seed(18)
    B, T = 2, 6
    x = torch.randint(0, VOCAB, (B, T))

    with torch.no_grad():
        # (a) identity positions == implicit arange
        pos_identity = torch.arange(T).unsqueeze(0).repeat(B, 1)
        out_explicit = model(x, positions=pos_identity)[0]
        out_implicit = model(x)[0]
        assert out_explicit.shape == out_implicit.shape
        assert torch.allclose(out_explicit, out_implicit, atol=1e-6), (
            "positions=arange(0,T) must reproduce the default "
            "(implicit) position indexing"
        )

        # (b1) explicit offset == implicit start_pos offset
        pos_offset = torch.arange(2, 2 + T).unsqueeze(0).repeat(B, 1)
        out_pos2 = model(x, positions=pos_offset)[0]
        out_sp2 = model(x, start_pos=2)[0]
        assert torch.allclose(out_pos2, out_sp2, atol=1e-6), (
            "positions=arange(2, 2+T) must equal the implicit positions "
            "of a no-past forward with start_pos=2"
        )

        # (b2) two leading dummy slots, masked out
        torch.manual_seed(19)
        dummy = torch.randint(0, VOCAB, (B, 2))
        full = torch.cat([dummy, x], dim=1)
        amask = torch.ones(B, T + 2, dtype=torch.bool)
        amask[:, :2] = False
        out_full = model(full, attention_mask=amask)[0]
        assert torch.allclose(out_full[:, 2:], out_pos2, atol=1e-6), (
            "x at absolute positions 2..2+T-1 must produce the same "
            "logits whether the two leading slots hold masked-out "
            "dummies or are simply absent (explicit positions)"
        )


# ---------------------------------------------------------------------------
# 13. ModernGPT.forward kv_mask semantics (Red)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position_encoding", ["rope", "learned"])
def test_forward_kv_mask_semantics(position_encoding):
    """kv_mask: [B, past_len] bool (True=valid KV) with a value-form
    past_kvs.  False positions are additively masked with -inf, i.e. they
    behave as if those positions' KV did not exist.

    (a) full past buffer (garbage tail) + kv_mask + explicit positions
        == (b) clean past slice without kv_mask, where the new token's
        implicit absolute position (start_pos + past_len = L_valid)
        matches the explicit positions of run (a).

    Value-form past layout (verified in the block forward): per-layer
    (k, v) tuples with k of shape [B, n_kv_head, L, head_dim].

    B=2 variant: two rows with different L_valid and different positions
    run as one batch must match each row run alone (the scheduler decode
    scenario at unit level).
    """
    model = _tiny_model(seed=23, position_encoding=position_encoding)
    cfg = model.config
    H = cfg.n_kv_head
    D = cfg.n_embd // cfg.n_head
    n_layer = cfg.n_layer
    L_total, L_valid = 10, 6

    torch.manual_seed(24)
    k_full = torch.randn(1, H, L_total, D)
    v_full = torch.randn(1, H, L_total, D)
    past_full = [(k_full, v_full) for _ in range(n_layer)]
    past_clean = [
        (k_full[:, :, :L_valid], v_full[:, :, :L_valid])
        for _ in range(n_layer)
    ]
    kv_mask = torch.zeros(1, L_total, dtype=torch.bool)
    kv_mask[:, :L_valid] = True
    x = torch.randint(0, VOCAB, (1, 1))

    with torch.no_grad():
        # (a) padded buffer + kv_mask (garbage tail masked out)
        out_masked = model(
            x,
            past_kvs=past_full,
            use_cache=False,
            positions=torch.tensor([[L_valid]]),
            kv_mask=kv_mask,
        )[0]
        # (b) clean slice, no kv_mask; implicit position = 0 + L_valid
        out_clean = model(x, past_kvs=past_clean, use_cache=False)[0]
        assert out_masked.shape == out_clean.shape
        assert torch.allclose(out_masked, out_clean, atol=1e-5), (
            "kv_mask=False positions must behave exactly like absent KV "
            "positions (masked garbage tail vs clean slice)"
        )

        # Non-vacuity guard: without kv_mask the garbage tail really does
        # change the logits, so the (a)/(b) equality above is meaningful.
        out_polluted = model(
            x,
            past_kvs=past_full,
            use_cache=False,
            positions=torch.tensor([[L_valid]]),
        )[0]
        assert not torch.allclose(out_polluted, out_clean, atol=1e-4), (
            "sanity: the unmasked garbage tail must perturb the logits"
        )

    # --- B=2: per-row valid lengths, batched vs per-row ---
    L_total2 = 12
    valid_lens = [4, 9]
    torch.manual_seed(25)
    k2 = torch.randn(2, H, L_total2, D)
    v2 = torch.randn(2, H, L_total2, D)
    kv_mask2 = torch.zeros(2, L_total2, dtype=torch.bool)
    kv_mask2[0, : valid_lens[0]] = True
    kv_mask2[1, : valid_lens[1]] = True
    x2 = torch.randint(0, VOCAB, (2, 1))
    positions2 = torch.tensor([[valid_lens[0]], [valid_lens[1]]])
    past2 = [(k2, v2) for _ in range(n_layer)]

    with torch.no_grad():
        out_batch = model(
            x2,
            past_kvs=past2,
            use_cache=False,
            positions=positions2,
            kv_mask=kv_mask2,
        )[0]
        for b in range(2):
            past_b = [
                (
                    k2[b : b + 1, :, : valid_lens[b]],
                    v2[b : b + 1, :, : valid_lens[b]],
                )
                for _ in range(n_layer)
            ]
            out_row = model(
                x2[b : b + 1], past_kvs=past_b, use_cache=False
            )
            assert torch.allclose(
                out_batch[b], out_row[0], atol=1e-5
            ), (
                f"row {b}: batched kv_mask decode must match the row run "
                "alone with a clean past slice"
            )


# ---------------------------------------------------------------------------
# 14. Repetition penalty semantics (Red, weak assertions)
# ---------------------------------------------------------------------------


def test_repetition_penalty_semantics():
    """Batching must not change repetition-penalty semantics: a penalty
    row batched with a no-penalty row must equal the same request run
    alone (B=1) with the same penalty; the no-penalty row must equal the
    plain greedy reference generate.  Whether the penalty changes the
    output at all is not asserted (weak assertion, no internal names
    locked)."""
    model = _tiny_model(seed=101)
    prompt = [12, 45, 3, 27]
    max_tokens = 6

    batch_requests = [
        _make_request(prompt, max_tokens=max_tokens, request_id="nopen"),
        _make_request(
            prompt, max_tokens=max_tokens, request_id="pen",
            repetition_penalty=1.5,
        ),
    ]
    scheduler, results = _run_scheduler(model, batch_requests, batch_size=2)
    assert isinstance(results, list) and len(results) == 2
    by_id = {r.request_id: r for r in results}

    solo_requests = [
        _make_request(
            prompt, max_tokens=max_tokens, request_id="pen",
            repetition_penalty=1.5,
        ),
    ]
    _, solo_results = _run_scheduler(model, solo_requests, batch_size=1)
    assert isinstance(solo_results, list) and len(solo_results) == 1

    assert by_id["pen"].token_ids == solo_results[0].token_ids, (
        "repetition penalty must be applied per sequence: the batched "
        "penalty row must match the same request run alone "
        f"(batch {by_id['pen'].token_ids}, solo "
        f"{solo_results[0].token_ids})"
    )

    expected = _generate_reference(model, prompt, max_tokens)
    assert _completion_of(by_id["nopen"], prompt) == expected, (
        "the no-penalty row must equal the plain greedy generate "
        "reference"
    )


# ---------------------------------------------------------------------------
# 15. Supplementary: eos finish semantics (Red)
# ---------------------------------------------------------------------------


def test_eos_finish_reason_and_token_inclusion():
    """Supplementary to the FR7 contract's eos semantics: sampling the
    eos_token_id finishes the request with finish_reason='eos', the eos
    token is included in token_ids, and the completion stops there (it
    does not run on to max_tokens).  The eos id is chosen as the greedy
    first token of the prompt, so the hit is deterministic."""
    model = _tiny_model(seed=202)
    prompt = [7, 19, 44, 2]
    greedy_first = _generate_reference(model, prompt, 1)[0]

    scheduler, results = _run_scheduler(
        model,
        [
            _make_request(
                prompt, max_tokens=5, request_id="eosr",
                eos_token_id=greedy_first,
            )
        ],
        batch_size=1,
    )
    assert isinstance(results, list) and len(results) == 1
    res = results[0]
    assert res.finish_reason == "eos", (
        f"hitting eos_token_id must finish with 'eos' "
        f"(got {res.finish_reason!r})"
    )
    assert res.token_ids[-1] == greedy_first, (
        "the eos token must be included at the end of token_ids"
    )
    completion = _completion_of(res, prompt)
    assert len(completion) == 1 and completion[0] == greedy_first, (
        "generation must stop at the sampled eos token (completion must "
        f"be exactly [eos], got {completion})"
    )
    assert res.finish_iteration == 1, (
        "the eos token is sampled in the prefill iteration, so the "
        "request finishes in iteration 1"
    )
