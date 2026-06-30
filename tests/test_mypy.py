"""Static type-check regression test for core modules."""

import subprocess
import sys

CORE_MODULES = [
    "model/modern_gpt.py",
    "model/paged_kv_cache.py",
    "model/quantization.py",
    "model/gguf_utils.py",
    "utils/rl_utils.py",
    "utils/dpo_utils.py",
    "inference/generate_utils.py",
]


def test_mypy_on_core_modules() -> None:
    """Run mypy on annotated core modules; fail on type errors."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", *CORE_MODULES],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"mypy failed:\n{result.stdout}\n{result.stderr}"
