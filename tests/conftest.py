"""Shared pytest fixtures and marker registration for the test suite.

The tiny-model factories live in ``tests/helpers.py`` so test modules
can also import them as plain functions; this conftest re-exports them
as fixtures per the FR11 design.
"""

import helpers as test_helpers
import pytest


@pytest.fixture
def tiny_modern_config():
    """Factory fixture: tiny ModernGPTConfig(**overrides)."""
    return test_helpers.tiny_modern_config


@pytest.fixture
def tiny_model():
    """Factory fixture: tiny ModernGPT(**overrides), eval mode."""
    return test_helpers.tiny_model


def pytest_configure(config):
    config.addinivalue_line("markers", "perf: performance benchmark (machine-dependent).")
    config.addinivalue_line("markers", "slow: long-running test, excluded from quick runs.")
