"""Regression tests for utils.lr_scheduler."""
import pytest

from utils.lr_scheduler import LRScheduler


def test_cosine_scheduler_monotonic_after_warmup():
    sched = LRScheduler("cosine", learning_rate=1e-3, min_lr=1e-5,
                        warmup_iters=10, lr_decay_iters=100)
    # Warmup increases LR.
    assert sched(0) < sched(5) < sched(9)
    # After decay horizon LR is at min.
    assert pytest.approx(sched(100), rel=1e-6) == 1e-5
    # Decay phase is monotonic.
    assert sched(50) > sched(90) > sched(100)


def test_wsd_scheduler_requires_max_iters():
    sched = LRScheduler("wsd", learning_rate=1e-3, min_lr=1e-5,
                        warmup_iters=0, lr_decay_iters=20, max_iters=100)
    # Stable phase.
    assert pytest.approx(sched(50), rel=1e-6) == 1e-3
    # Decay phase ends at min_lr.
    assert pytest.approx(sched(100), rel=1e-6) == 1e-5


def test_state_dict_roundtrip():
    sched = LRScheduler("cosine", learning_rate=1e-3, min_lr=1e-5,
                        warmup_iters=10, lr_decay_iters=100, max_iters=100)
    state = sched.state_dict()
    sched2 = LRScheduler("linear", learning_rate=0.0, min_lr=0.0,
                         warmup_iters=0, lr_decay_iters=1)
    sched2.load_state_dict(state)
    for step in [0, 5, 10, 50, 100]:
        assert pytest.approx(sched(step), rel=1e-9) == sched2(step)


def test_constant_scheduler():
    sched = LRScheduler("constant", learning_rate=1e-3, min_lr=1e-5,
                        warmup_iters=0, lr_decay_iters=100)
    assert pytest.approx(sched(0), rel=1e-6) == 1e-3
    assert pytest.approx(sched(1000), rel=1e-6) == 1e-3
