"""FR4c regression tests: real per-token timing + benchmark statistics.

Contracts locked by this file (Green implementation must follow):

- ``ModernGPT.generate`` accepts an ``on_token`` callback: ``on_token(idx_next)``
  called exactly once per generated token, where ``idx_next`` is a
  ``LongTensor`` of shape ``[B, 1]`` holding the freshly sampled token.
- ``InferenceProfiler`` hooked via ``on_token=inf_prof.step`` records one real
  decode step per token (summary ``tokens`` == generated count, ``median_ms``
  in a realistic millisecond range, not ~0 as with post-hoc fake stepping).
- ``InferenceProfiler.summary`` reports per-token stats consistent with the
  recorded delay sequence: keys ``tokens``, ``mean_ms``, ``median_ms``,
  ``p95_ms`` (percentile convention: ascending sort, index ``n // 2`` for
  median, ``int(0.95 * n)`` for p95).
- ``inference.generate.benchmark`` on CPU: positive finite timings measured
  with ``perf_counter`` (no GPU-event dependency), at least 3 warmup runs that
  are excluded from statistics, and top-level percentile keys ``p50_ms``,
  ``p95_ms``, ``p99_ms`` with ``p50_ms <= p95_ms <= p99_ms``.
"""

import math
import time
from unittest.mock import Mock

import pytest
import torch

from inference.generate import benchmark
from model.modern_gpt import ModernGPT, ModernGPTConfig
from utils.profiler import InferenceProfiler

N_TOKENS = 5
PROMPT_LEN = 8


def _tiny_model():
    config = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        block_size=64,
        vocab_size=100,
        dropout=0.0,
    )
    model = ModernGPT(config)
    model.eval()
    return model


def _prompt():
    torch.manual_seed(1337)
    return torch.randint(0, 100, (1, PROMPT_LEN), dtype=torch.long)


@pytest.mark.parametrize("use_cache", [False, True])
def test_generate_supports_on_token_callback(use_cache):
    """generate(on_token=...) fires exactly once per token with the new token."""
    model = _tiny_model()
    prompt = _prompt()
    received = []
    with torch.no_grad():
        out = model.generate(
            prompt,
            max_new_tokens=N_TOKENS,
            temperature=1.0,
            use_cache=use_cache,
            on_token=received.append,
        )
    assert len(received) == N_TOKENS
    for tok in received:
        assert isinstance(tok, torch.Tensor)
        assert tok.dtype == torch.long
        assert tok.shape == (1, 1)
        assert tok.device == out.device
    replayed = torch.cat(received, dim=1)
    assert replayed.shape == (1, N_TOKENS)
    assert torch.equal(replayed, out[:, PROMPT_LEN:])


@pytest.mark.parametrize("use_cache", [False, True])
def test_on_token_hooks_real_decode_timing(use_cache, tmp_path):
    """InferenceProfiler.step via on_token yields one real decode step per token."""
    model = _tiny_model()
    prompt = _prompt()
    inf_prof = InferenceProfiler(output_dir=str(tmp_path), device="cpu")
    with inf_prof:
        with torch.no_grad():
            model.generate(
                prompt,
                max_new_tokens=8,
                temperature=1.0,
                use_cache=use_cache,
                on_token=inf_prof.step,
            )
    summary = inf_prof.summary()
    assert summary["tokens"] == 8
    assert summary["median_ms"] > 0.01
    assert summary["median_ms"] < 60000.0
    assert math.isfinite(summary["p95_ms"]) and summary["p95_ms"] > 0.0


def test_inference_profiler_reports_known_delay_sequence(tmp_path):
    """Mock step producing known delays: summary mean/percentiles match sequence."""
    n = 10
    delays = [0.003 + 0.002 * i for i in range(n)]
    model = _tiny_model()
    prompt = _prompt()
    inf_prof = InferenceProfiler(output_dir=str(tmp_path), device="cpu")

    def _fake_step(token=None):
        inf_prof.times.append(delays[len(inf_prof.times)])

    step_mock = Mock(side_effect=_fake_step)
    with inf_prof:
        with torch.no_grad():
            model.generate(
                prompt,
                max_new_tokens=n,
                temperature=1.0,
                on_token=step_mock,
            )
    assert step_mock.call_count == n
    summary = inf_prof.summary()
    assert summary["tokens"] == n
    sorted_d = sorted(delays)
    assert summary["mean_ms"] == pytest.approx((sum(delays) / n) * 1000.0, rel=1e-6)
    assert summary["median_ms"] == pytest.approx(sorted_d[n // 2] * 1000.0, rel=1e-6)
    assert summary["p95_ms"] == pytest.approx(
        sorted_d[int(0.95 * n)] * 1000.0, rel=1e-6
    )


def test_benchmark_cpu_positive_time_and_percentiles():
    """CPU benchmark returns positive perf_counter timings with p50/p95/p99."""
    model = _tiny_model()
    prompt = _prompt()
    res = benchmark(
        model,
        prompt,
        max_new_tokens=4,
        num_samples=3,
        device="cpu",
        use_cache=False,
    )
    for key in ("p50_ms", "p95_ms", "p99_ms"):
        assert key in res, f"benchmark result missing {key}"
        assert math.isfinite(res[key]) and res[key] > 0.0
    assert res["p50_ms"] <= res["p95_ms"] <= res["p99_ms"]
    assert res["total_time_ms"] > 0.0
    assert math.isfinite(res["total_time_ms"])


class _CountingGenModel:
    """Fake model: generate() sleeps a known per-call duration and counts calls."""

    def __init__(self):
        self.calls = 0
        self.config = type("C", (), {"block_size": 64})()

    def generate(
        self,
        idx,
        max_new_tokens=0,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        use_cache=True,
        compile=False,
    ):
        i = self.calls
        self.calls += 1
        time.sleep(0.08 if i == 0 else 0.002)
        return idx


def test_benchmark_warmup_at_least_three_and_excluded_from_stats():
    """warmup >= 3 extra runs, and the (slow) first warmup run is excluded."""
    model = _CountingGenModel()
    prompt = _prompt()
    num_samples = 5
    res = benchmark(
        model,
        prompt,
        max_new_tokens=4,
        num_samples=num_samples,
        device="cpu",
        use_cache=False,
    )
    warmup = model.calls - num_samples
    assert model.calls >= num_samples
    assert warmup >= 3, f"expected >= 3 warmup runs, got {warmup}"
    for key in ("p50_ms", "p95_ms", "p99_ms"):
        assert key in res, f"benchmark result missing {key}"
    assert res["p99_ms"] < 40.0, (
        f"slow warmup run leaked into stats: p99_ms={res['p99_ms']}"
    )
    assert 0.0 < res["p50_ms"] < 40.0
