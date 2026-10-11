"""FR13 (T020): lint rule uniqueness and test-infrastructure contracts.

Static contracts (design 6.5):

- black and ruff both use line-length 100 and the rule set is unique;
- ruff selects E/F/I/B/UP;
- mypy python_version is pinned per design;
- the KV-cache benchmark uses machine-independent (relative)
  assertions and carries the perf marker so ``-m "not perf"`` isolates
  it;
- README test-count badge no longer drifts from the collected count.
"""

import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject():
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. lint rule uniqueness
# ---------------------------------------------------------------------------


def test_black_line_length_is_100():
    assert _pyproject()["tool"]["black"]["line-length"] == 100


def test_ruff_line_length_is_100_with_select():
    ruff = _pyproject()["tool"]["ruff"]
    assert ruff["line-length"] == 100
    # ruff moved lint settings into the [tool.ruff.lint] section; accept
    # either location but require the rule selection to be pinned.
    lint = ruff.get("lint", ruff)
    select = lint.get("select", [])
    for code in ["E", "F", "I", "B", "UP"]:
        assert code in select, code


def test_line_length_rules_unique():
    data = _pyproject()
    lengths = {
        "black": data["tool"]["black"]["line-length"],
        "ruff": data["tool"]["ruff"]["line-length"],
    }
    assert len(set(lengths.values())) == 1, lengths


def test_mypy_python_version_pinned():
    data = _pyproject()
    version = data["tool"]["mypy"]["python_version"]
    assert version in ("3.9", "3.11"), version
    # pin must be a string of the documented target
    assert re.fullmatch(r"3\.\d+", version)


# ---------------------------------------------------------------------------
# 2. KV benchmark: relative assertions + perf marker
# ---------------------------------------------------------------------------


def test_kv_benchmark_has_perf_marker():
    src = (REPO_ROOT / "tests" / "test_kv_cache_benchmark.py").read_text(encoding="utf-8")
    assert "@pytest.mark.perf" in src


def test_kv_benchmark_no_absolute_time_thresholds():
    """Timing assertions must be relative (machine-independent)."""
    src = (REPO_ROOT / "tests" / "test_kv_cache_benchmark.py").read_text(encoding="utf-8")
    # absolute wall-clock thresholds like `t_cache < 60.0` are banned
    assert not re.search(r"t_\w+\s*<\s*\d+\.\d+", src)
    assert not re.search(r"time\.time\(\)\s*<", src)


def test_perf_marker_registered():
    data = _pyproject()
    markers = "\n".join(data["tool"]["pytest"]["ini_options"].get("markers", []))
    assert "perf" in markers
    assert "slow" in markers


def test_not_perf_collects_clean():
    """``pytest -m 'not perf' --collect-only`` must exit 0."""
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "not perf", "--collect-only", "-q", "tests"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]


def test_perf_marker_selects_benchmarks():
    """``-m perf`` must select at least one test (isolatable)."""
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "perf", "--collect-only", "-q", "tests"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    m = re.search(r"(\d+) tests? collected", res.stdout)
    assert m, res.stdout[-500:]
    assert int(m.group(1)) >= 1


# ---------------------------------------------------------------------------
# 3. documentation numbers do not drift
# ---------------------------------------------------------------------------


def _collected_count():
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, res.stderr[-2000:]
    m = re.search(r"(\d+) tests? collected", res.stdout)
    assert m, res.stdout[-500:]
    return int(m.group(1))


def test_readme_test_badge_not_stale():
    """The README test badge lower bound must not understate the suite."""
    src = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"tests-(\d+)\+?%2?0?passing", src) or re.search(r"badge/tests-(\d+)", src)
    assert m, "README must carry a tests badge with a numeric bound"
    badge = int(m.group(1))
    assert _collected_count() >= badge


def test_ci_has_lint_jobs():
    src = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "black --check" in src
    assert "ruff check" in src
    assert "mypy" in src
