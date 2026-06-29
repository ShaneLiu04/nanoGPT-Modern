"""Tests for model.attention_utils."""
import os

import pytest


from model.attention_utils import set_attention_backend, get_attention_backend_info, VALID_BACKENDS


def test_valid_backends():
    assert "auto" in VALID_BACKENDS
    assert "flash" in VALID_BACKENDS
    assert "math" in VALID_BACKENDS


def test_set_attention_backend_invalid():
    with pytest.raises(ValueError):
        set_attention_backend("not_a_backend")


def test_set_attention_backend_math():
    if not __import__("torch").cuda.is_available():
        pytest.skip("CUDA not available")
    set_attention_backend("math")
    info = get_attention_backend_info()
    assert info["math"] is True
    assert info["flash"] is False
    assert info["mem_efficient"] is False
    # Restore default.
    set_attention_backend("auto")


def test_set_attention_backend_default_enables_all():
    if not __import__("torch").cuda.is_available():
        pytest.skip("CUDA not available")
    set_attention_backend("default")
    info = get_attention_backend_info()
    assert info["flash"] is True
    assert info["mem_efficient"] is True
    assert info["math"] is True
