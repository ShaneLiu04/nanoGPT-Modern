"""T015 (FR7b) Red-phase tests: api_server continuous-batching wiring.

Background (FR7b)
-----------------
The current ``api_server.py`` has three wiring gaps:

* the module docstring (L9-10) claims a ``--continuous_batching`` flag
  ("requests are queued and dynamically packed"), but ``get_args()``
  defines no such flag -- a false documentation claim;
* generation is called *synchronously* inside the async endpoints
  (``model_state.generate(...)`` at L311-318 / L354-361), blocking the
  event loop; the design requires the call to be moved into
  ``asyncio.to_thread``;
* no continuous-batching scheduler is wired in at all (T014 delivered
  ``inference/continuous_batching.py`` but the server never uses it).

Target contract (Green)
-----------------------
1. ``get_args()`` gains ``--continuous_batching`` (store_true) and
   ``--batch_size`` (int, default 4).  Tested as a *structure
   assertion* on ``inspect.getsource(api_server.get_args)`` (the real
   ``get_args`` calls ``parse_args`` and cannot be driven from tests).
   The module source must also contain ``asyncio.to_thread``.
2. New module global ``cb_state``.  When ``cb_state`` is not None the
   two generation endpoints dispatch to
   ``cb_state.generate(prompt_text, max_tokens, temperature, top_k,
   top_p, repetition_penalty) -> (text, prompt_tokens,
   completion_tokens, finish_reason)`` (a *blocking* synchronous call,
   run by the endpoint inside ``asyncio.to_thread``).  When it is None
   the original ``model_state.generate`` path is used (also via
   ``asyncio.to_thread``).  The dispatch is pinned by a call-counting
   stub: in CB mode ``cb_state.calls`` must be >= number of requests.
3. Both modes return 200 with non-empty text, correct usage
   (``completion_tokens == max_tokens`` without EOS truncation) and
   ``finish_reason`` in ``{"stop", "length"}``.
4. 4 concurrent requests complete in both modes (``httpx.AsyncClient``
   + ``ASGITransport``); ``asyncio.to_thread`` must not block the
   event loop.
5. ``model_state`` None -> POST returns 503 (current-behaviour pin).
6. ``/health`` returns 200 + status ok (pin); ``/v1/models`` returns
   200 + non-empty data (pin).

Red-phase semantics
-------------------
* structure test (1) FAILs now: the flag and ``asyncio.to_thread`` do
  not exist in the source.
* CB-mode tests (3 cb-variant, 4 cb-variant, 5) FAIL now on the
  dispatch assertion (``stub.calls == 0``: the current endpoints call
  ``model_state.generate`` directly and ignore ``cb_state``).  Their
  *behavioural* assertions (200 / non-empty / usage) already pass via
  the legacy path because the test injects both ``model_state`` and
  ``cb_state`` -- only the dispatch distinguishes Red from Green.
* off-mode tests (2, 4 off-variant, 6) are pins of current behaviour
  and PASS now.
* uvicorn startup / ``main()`` are deliberately NOT tested (process
  level behaviour is not locked).

Determinism notes
-----------------
* tiktoken (if installed and functional) encodes prompts to ids far
  above the tiny model's vocab (300), so the tests *force* the
  documented ord/chr fallback by making ``tiktoken.get_encoding``
  raise -- this exercises the real fallback branch of
  ``ModelState.tokenizer`` and keeps the tests hermetic.
* vocab_size=300 >= 256 keeps every ord() id in range; seed 1234 makes
  greedy (temperature=0.0) decoding deterministic and non-empty for
  all prompts used here (verified: e.g. "hello world" -> 'dddddddd').
* All tests run on CPU, tiny models, ASCII-only source (Windows GBK).
"""

import asyncio
import inspect
import itertools
import threading
import time

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

import torch
from fastapi.testclient import TestClient

import api_server
import inference.continuous_batching as cb
from model.modern_gpt import ModernGPT, ModernGPTConfig

MAX_TOKENS = 8
SEED = 1234
PROMPTS = ["alpha prompt", "bravo prompt", "charlie prompt", "delta prompt"]
VALID_FINISH_REASONS = ("stop", "length")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _force_ord_tokenizer(monkeypatch):
    """Make ``tiktoken.get_encoding`` raise so ``ModelState`` uses its
    documented ord/chr fallback (hermetic across machines with/without
    a working tiktoken cache).  No-op when tiktoken is absent."""
    try:
        import tiktoken
    except Exception:
        return

    def _raise(name):
        raise RuntimeError("tiktoken disabled in api_server wiring tests")

    monkeypatch.setattr(tiktoken, "get_encoding", _raise)


def _completion_body(prompt, max_tokens=MAX_TOKENS):
    return {
        "model": "modern",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": None,
    }


def _chat_body(max_tokens=MAX_TOKENS):
    return {
        "model": "modern",
        "messages": [{"role": "user", "content": "hello there"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": None,
        "top_p": None,
        "repetition_penalty": None,
    }


def _assert_completion_ok(resp, max_tokens=MAX_TOKENS):
    """Common behavioural assertions for /v1/completions."""
    assert resp.status_code == 200, resp.text
    body = resp.json()
    text = body["choices"][0]["text"]
    assert isinstance(text, str) and text, (
        f"choices[0].text must be a non-empty string, got {text!r}"
    )
    usage = body["usage"]
    assert usage["completion_tokens"] == max_tokens, usage
    assert usage["total_tokens"] == usage["prompt_tokens"] + max_tokens, usage
    assert body["choices"][0]["finish_reason"] in VALID_FINISH_REASONS, body
    return body


class _CBStub:
    """Test-side ``cb_state`` implementing the endpoint contract.

    ``generate(prompt_text, max_tokens, temperature, top_k, top_p,
    repetition_penalty) -> (text, prompt_tokens, completion_tokens,
    finish_reason)`` -- blocking until *this* request's
    ``GenerateResult`` exists.  Internally it wraps T014's
    ``ContinuousBatchingServer``: submit a ``Request`` (unique
    request_id), then any thread holding ``_run_lock`` pumps
    ``scheduler.run()`` and stores the finished results by request_id;
    a request submitted while another run is in flight is picked up at
    that run's next iteration boundary (or by a later run).  The
    ``calls`` counter pins the endpoint dispatch: the Green endpoint
    must route through ``cb_state.generate`` exactly once per request.

    The class name/wrapper shape is NOT part of the contract -- only
    the ``generate`` signature is locked.
    """

    def __init__(self, model_state, batch_size=4):
        self._ms = model_state
        self._server = cb.ContinuousBatchingServer(
            model_state.model, device="cpu", batch_size=batch_size
        )
        self._results = {}
        self._run_lock = threading.Lock()
        self._id_iter = itertools.count()
        self.calls = 0
        self.call_prompts = []

    def generate(
        self,
        prompt_text,
        max_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    ):
        self.calls += 1
        self.call_prompts.append(prompt_text)
        rid = f"test-req-{next(self._id_iter)}"
        prompt_ids = self._ms.encode(prompt_text)
        req = cb.Request(
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=None,
            request_id=rid,
        )
        self._server.submit(req)

        deadline = time.monotonic() + 30.0
        result = None
        while time.monotonic() < deadline:
            if rid in self._results:
                result = self._results.pop(rid)
                break
            with self._run_lock:
                if len(self._server.queue) > 0:
                    for r in self._server.generate_pending():
                        self._results[r.request_id] = r
            if rid in self._results:
                result = self._results.pop(rid)
                break
            time.sleep(0.005)
        if result is None:
            raise TimeoutError(f"CB request {rid!r} did not finish within 30s")

        prompt_len = prompt_ids.shape[1]
        completion = result.token_ids[prompt_len:]
        text = self._ms.decode(torch.tensor([completion], dtype=torch.long))
        return text, prompt_len, len(completion), result.finish_reason


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def server(tmp_path, monkeypatch):
    """Tiny ModernGPT (vocab 300 >= 256 for the ord fallback) saved to a
    checkpoint and loaded through the real ``ModelState`` path; injected
    into ``api_server.model_state`` with ``cb_state`` left empty.
    ``monkeypatch`` restores both globals after each test (cb_state is
    deleted again, so the module stays clean for later importers)."""
    _force_ord_tokenizer(monkeypatch)
    torch.manual_seed(SEED)
    cfg = ModernGPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=32,
        block_size=128,
        vocab_size=300,
        dropout=0.0,
        position_encoding="rope",
    )
    model = ModernGPT(cfg)
    ckpt = tmp_path / "tiny_ckpt.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict()}, str(ckpt))
    ms = api_server.ModelState(str(ckpt), "modern", "cpu")
    monkeypatch.setattr(api_server, "model_state", ms)
    monkeypatch.setattr(api_server, "cb_state", None, raising=False)
    return ms


@pytest.fixture()
def cb_stub(server, monkeypatch):
    """``server`` + a ``_CBStub`` injected as ``api_server.cb_state``."""
    stub = _CBStub(server, batch_size=4)
    monkeypatch.setattr(api_server, "cb_state", stub, raising=False)
    return stub


def _gather_posts(path, bodies):
    """POST all bodies concurrently against the ASGI app and return the
    responses in order."""
    n = len(bodies)

    async def main():
        transport = httpx.ASGITransport(app=api_server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=30.0
        ) as client:

            async def one(body):
                return await client.post(path, json=body)

            return await asyncio.gather(*[one(b) for b in bodies])

    assert n > 0
    return asyncio.run(main())


# ---------------------------------------------------------------------------
# 1. CLI flag + to_thread structure (Red)
# ---------------------------------------------------------------------------
def test_flag_and_tothread_structure():
    """get_args() must define --continuous_batching and --batch_size, and
    the module must route generation through asyncio.to_thread."""
    getargs_src = inspect.getsource(api_server.get_args)
    assert "--continuous_batching" in getargs_src, (
        "api_server.get_args() must define the --continuous_batching flag "
        "(the module docstring already claims it exists)"
    )
    assert "--batch_size" in getargs_src, (
        "api_server.get_args() must define --batch_size (int, default 4)"
    )
    module_src = inspect.getsource(api_server)
    assert "asyncio.to_thread" in module_src, (
        "api_server must run blocking generation inside asyncio.to_thread "
        "so the event loop is not blocked during inference"
    )


# ---------------------------------------------------------------------------
# 2. /v1/completions, CB off (pin)
# ---------------------------------------------------------------------------
def test_completions_off_mode(server):
    """CB off: legacy path returns 200, non-empty text, exact usage."""
    client = TestClient(api_server.app)
    resp = client.post("/v1/completions", json=_completion_body("hello world"))
    body = _assert_completion_ok(resp)
    assert body["usage"]["prompt_tokens"] > 0


# ---------------------------------------------------------------------------
# 3. /v1/completions, CB on (Red: dispatch)
# ---------------------------------------------------------------------------
def test_completions_cb_mode(cb_stub):
    """CB on: same behaviour as off mode AND dispatched through cb_state."""
    client = TestClient(api_server.app)
    resp = client.post("/v1/completions", json=_completion_body("hello world"))
    _assert_completion_ok(resp)
    assert cb_stub.calls >= 1, (
        "with api_server.cb_state set, the completions endpoint must "
        "dispatch generation through cb_state.generate (legacy "
        "model_state.generate path was used instead)"
    )


# ---------------------------------------------------------------------------
# 4. /v1/chat/completions, both modes (off = pin, cb = Red: dispatch)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_cb", [False, True])
def test_chat_completions_both_modes(use_cb, server, monkeypatch):
    """/v1/chat/completions returns 200 + non-empty content in both
    modes; in CB mode the request must go through cb_state."""
    stub = None
    if use_cb:
        stub = _CBStub(server, batch_size=4)
        monkeypatch.setattr(api_server, "cb_state", stub, raising=False)
    client = TestClient(api_server.app)
    resp = client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content, (
        f"choices[0].message.content must be non-empty, got {content!r}"
    )
    usage = body["usage"]
    assert usage["completion_tokens"] == MAX_TOKENS, usage
    assert body["choices"][0]["finish_reason"] in VALID_FINISH_REASONS, body
    if use_cb:
        assert stub.calls >= 1, (
            "with api_server.cb_state set, the chat completions endpoint "
            "must dispatch generation through cb_state.generate"
        )


# ---------------------------------------------------------------------------
# 5. Concurrency, CB on (Red: dispatch)
# ---------------------------------------------------------------------------
def test_concurrent_cb_mode(cb_stub):
    """CB on: 4 concurrent requests all complete with correct usage and
    each is dispatched through cb_state."""
    responses = _gather_posts(
        "/v1/completions", [_completion_body(p) for p in PROMPTS]
    )
    assert len(responses) == len(PROMPTS)
    for resp in responses:
        _assert_completion_ok(resp)
    assert cb_stub.calls >= len(PROMPTS), (
        f"all {len(PROMPTS)} concurrent requests must be dispatched "
        f"through cb_state.generate (got {cb_stub.calls} calls)"
    )


# ---------------------------------------------------------------------------
# 6. Concurrency, CB off (pin; also guards the to_thread requirement)
# ---------------------------------------------------------------------------
def test_concurrent_off_mode(server):
    """CB off: 4 concurrent requests all complete (generation must not
    block the event loop once moved into asyncio.to_thread)."""
    responses = _gather_posts(
        "/v1/completions", [_completion_body(p) for p in PROMPTS]
    )
    assert len(responses) == len(PROMPTS)
    for resp in responses:
        _assert_completion_ok(resp)


# ---------------------------------------------------------------------------
# 7. /health + 503 semantics (pin)
# ---------------------------------------------------------------------------
def test_health_and_503(server, monkeypatch):
    """/health returns 200 + ok when the model is loaded; with
    model_state None (and no cb_state) the POST endpoints return 503."""
    client = TestClient(api_server.app)
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "ok"

    monkeypatch.setattr(api_server, "model_state", None)
    monkeypatch.setattr(api_server, "cb_state", None, raising=False)
    resp = client.post("/v1/completions", json=_completion_body("hello world"))
    assert resp.status_code == 503, (
        f"model not loaded -> /v1/completions must return 503, "
        f"got {resp.status_code}"
    )
    resp = client.post("/v1/chat/completions", json=_chat_body())
    assert resp.status_code == 503, (
        f"model not loaded -> /v1/chat/completions must return 503, "
        f"got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# 8. /v1/models (pin)
# ---------------------------------------------------------------------------
def test_models_endpoint(server):
    """/v1/models returns 200 with a non-empty data list."""
    client = TestClient(api_server.app)
    resp = client.get("/v1/models")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list) and body["data"], body
    assert body["data"][0]["id"] == "modern"
