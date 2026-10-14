"""Contract tests for ``tools/infra/precision_probe.py`` (AR-v0.7.5 FR2, task T004).

TDD note: written against the design contract BEFORE the tool exists -- they
are expected to fail (red) until the implementation lands.

Contract under test (design.md sections 4.3.1 / 4.3.2 / 6.1 / 6.4 / 6.5):

* CLI: ``python tools/infra/precision_probe.py [--quick] [--output-json PATH]``
* Exit codes: 0 success; 1 measurement interrupted (no partial JSON written);
  2 usage error (unknown flag, ``--output-json`` parent dir missing/unwritable).
* Product JSON: ``{meta, experiments}`` two-part schema with ``meta.device ==
  "cpu"`` and ``meta.tolerance_refs`` pinned to the triton bf16 atol/rtol
  baseline (docs/results/triton_kernels.json, 0.01/0.01).
* Experiments: dtype_limits / rne_rounding / relative_error_grid /
  accumulation_precision / loss_scaling_simulation, with direction-level
  numerical assertions: bf16 (7-bit mantissa) element-wise error must dominate
  fp16 (10-bit); fp32 accumulation must beat bf16 accumulation; loss scaling
  must rescue fp16 tiny-gradient underflow and exercise unscale/inf-skip.
* Pure CPU: zero GPU dependency (torch CPU + standard library only).

The default output path (docs/results/precision_probe.json) is intentionally
never exercised here; every run passes ``--output-json`` into pytest tmp
storage so the repo's results directory is not polluted by test runs.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "infra" / "precision_probe.py"
TRITON_REFS = REPO_ROOT / "docs" / "results" / "triton_kernels.json"

REQUIRED_EXPERIMENTS = (
    "dtype_limits",
    "rne_rounding",
    "relative_error_grid",
    "accumulation_precision",
    "loss_scaling_simulation",
)

QUICK_BUDGET_SECONDS = 60.0
SUBPROCESS_TIMEOUT_SECONDS = 180


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the probe CLI as a subprocess from the repo root."""
    assert TOOL.is_file(), f"tool not implemented yet: {TOOL}"
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def get_experiment(payload: dict, name: str) -> dict:
    """Fetch one experiment record by name; fail loudly if it is missing."""
    for exp in payload["experiments"]:
        if exp["name"] == name:
            return exp
    pytest.fail(f"experiment {name!r} missing from precision_probe.json")


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One shared ``--quick`` subprocess run; every schema test reuses its JSON."""
    out = tmp_path_factory.mktemp("precision_probe") / "precision_probe.json"
    start = time.monotonic()
    proc = run_cli("--quick", "--output-json", str(out))
    elapsed = time.monotonic() - start
    return {"proc": proc, "out": out, "elapsed": elapsed}


@pytest.fixture(scope="module")
def probe_json(quick_run: dict) -> dict:
    """Parsed product JSON of the shared quick run (atomicity: it must load)."""
    proc = quick_run["proc"]
    assert proc.returncode == 0, proc.stderr
    return json.loads(quick_run["out"].read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Common CLI contract group (design 6.1, shared across the three tools)
# --------------------------------------------------------------------------


def test_cli_help_succeeds():
    """--help exits 0 and documents both flags (design 4.3.1)."""
    proc = run_cli("--help")
    assert proc.returncode == 0, proc.stderr
    assert "--quick" in proc.stdout
    assert "--output-json" in proc.stdout


def test_cli_quick_smoke_exits_zero_and_writes_json(quick_run: dict):
    """--quick smoke: exit 0, JSON on disk, under the 60 s budget, stdout summary.

    The mid-experiment interrupt branch itself (exit 1 + no partial JSON) is
    covered by the in-process injection test at the bottom of this module;
    this smoke plus the missing-parent-dir case below pin the exit-0/exit-2
    paths that the subprocess harness can reach.
    """
    proc = quick_run["proc"]
    assert proc.returncode == 0, proc.stderr
    assert quick_run["out"].is_file()
    assert quick_run["elapsed"] < QUICK_BUDGET_SECONDS
    # stdout must carry the Markdown summary (one line per experiment).
    assert proc.stdout.strip()
    assert "dtype_limits" in proc.stdout.lower()


def test_cli_unknown_flag_exits_two(tmp_path: Path):
    """An unknown flag is a usage error: argparse exits 2 with a stderr message."""
    out = tmp_path / "unused.json"
    proc = run_cli("--definitely-not-a-flag", "--output-json", str(out))
    assert proc.returncode == 2
    assert proc.stderr.strip()
    assert not out.exists()


def test_cli_output_json_missing_parent_dir_exits_two(tmp_path: Path):
    """Missing --output-json parent dir: exit 2, no implicit mkdir, no artifact."""
    target = tmp_path / "no_such_dir" / "precision_probe.json"
    proc = run_cli("--quick", "--output-json", str(target))
    assert proc.returncode == 2
    assert proc.stderr.strip()
    assert not target.parent.exists(), "tool must not silently create directories"
    assert not target.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX chmod semantics; Windows ACLs differ")
def test_cli_output_json_readonly_parent_dir_exits_two(tmp_path: Path):
    """A read-only parent dir shares the same exit-2 usage branch (design 4.3.1)."""
    parent = tmp_path / "readonly"
    parent.mkdir()
    target = parent / "precision_probe.json"
    os.chmod(parent, 0o500)
    try:
        proc = run_cli("--quick", "--output-json", str(target))
        assert proc.returncode == 2
        assert proc.stderr.strip()
        assert not target.exists()
    finally:
        os.chmod(parent, 0o700)


# --------------------------------------------------------------------------
# Product JSON schema (design 4.3.2)
# --------------------------------------------------------------------------


def test_json_meta_fields(probe_json: dict):
    """Two-part schema {meta, experiments} with the full mandatory meta field set."""
    assert set(probe_json) >= {"meta", "experiments"}
    meta = probe_json["meta"]
    assert meta["tool"] == "precision_probe"
    assert isinstance(meta["torch"], str) and meta["torch"]
    # Same interpreter as the subprocess: recorded version must be honest.
    assert torch.__version__.startswith(meta["torch"])
    assert isinstance(meta["timestamp"], str) and meta["timestamp"]
    assert meta["quick"] is True  # the shared run used --quick
    assert isinstance(meta["tolerance_refs"], dict)
    assert isinstance(probe_json["experiments"], list) and probe_json["experiments"]
    assert all("name" in exp for exp in probe_json["experiments"])


def test_meta_device_is_cpu(probe_json: dict):
    """Pure-CPU tool: meta.device must be exactly "cpu" (FR2, zero GPU deps)."""
    assert probe_json["meta"]["device"] == "cpu"


def test_all_required_experiments_present(probe_json: dict):
    """--quick shrinks tensor sizes but never drops an experiment (design 6.4)."""
    names = {exp["name"] for exp in probe_json["experiments"]}
    for required in REQUIRED_EXPERIMENTS:
        assert required in names


def test_tolerance_refs_match_triton_kernels_bf16(probe_json: dict):
    """meta.tolerance_refs must equal the triton bf16 atol/rtol baseline (FR2-2)."""
    if not TRITON_REFS.is_file():
        pytest.skip("docs/results/triton_kernels.json not available")
    refs = json.loads(TRITON_REFS.read_text(encoding="utf-8"))
    bf16_rows = [c for c in refs["consistency"] if c.get("dtype") == "bfloat16"]
    assert bf16_rows, "triton_kernels.json must carry bf16 consistency rows"
    tref = probe_json["meta"]["tolerance_refs"]
    assert tref["bf16_atol"] == pytest.approx(bf16_rows[0]["atol"])
    assert tref["bf16_rtol"] == pytest.approx(bf16_rows[0]["rtol"])
    assert tref["bf16_atol"] == pytest.approx(0.01)
    assert tref["bf16_rtol"] == pytest.approx(0.01)
    assert tref["source"] == "docs/results/triton_kernels.json"


# --------------------------------------------------------------------------
# Numerical experiment semantics (design D5 / 6.1 / 6.4)
# --------------------------------------------------------------------------


def test_dtype_limits_match_torch_finfo(probe_json: dict):
    """dtype_limits must reproduce torch.finfo field by field (FR1-1 cross-check)."""
    exp = get_experiment(probe_json, "dtype_limits")
    dtypes = exp["dtypes"]
    for name, torch_dtype in (
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
    ):
        assert name in dtypes, f"dtype_limits missing {name}"
        info = torch.finfo(torch_dtype)
        entry = dtypes[name]
        assert entry["max"] == pytest.approx(info.max)
        assert entry["min"] == pytest.approx(info.min)
        assert entry["eps"] == pytest.approx(info.eps)
        assert entry["smallest_normal"] == pytest.approx(info.smallest_normal)
        # ulp at 1.0 recomputed independently: spacing to the next representable.
        one = torch.tensor(1.0, dtype=torch_dtype)
        two = torch.tensor(2.0, dtype=torch_dtype)
        ulp_at_1 = (torch.nextafter(one, two) - one).item()
        assert entry["ulp_at_1"] == pytest.approx(ulp_at_1, rel=1e-6)
        assert entry["ulp_at_1"] == pytest.approx(info.eps, rel=1e-6)


def test_rne_rounding_tie_to_even(probe_json: dict):
    """Constructed RNE tie cases must round to even with a hit rate of exactly 1.0.

    Independent semantic cross-check first: in fp16 the binade [2048, 4096) has
    spacing 2, so 2049 and 2051 land exactly on ties and must round to even
    (2048 and 2052 respectively). The tool's constructed cases must encode the
    same IEEE 754 round-to-nearest-even behavior.
    """
    tie_probe = torch.tensor([2049.0, 2051.0], dtype=torch.float16)
    assert tie_probe.tolist() == [2048.0, 2052.0]

    exp = get_experiment(probe_json, "rne_rounding")
    assert isinstance(exp["cases"], list) and exp["cases"]
    assert exp["tie_to_even_hits"] == pytest.approx(1.0)


def test_relative_error_bf16_dominates_fp16(probe_json: dict):
    """bf16 (7-bit mantissa) element-wise error max must exceed fp16 (10-bit)."""
    exp = get_experiment(probe_json, "relative_error_grid")

    def stats_for(*needles: str) -> dict:
        for key, value in exp.items():
            if key != "name" and any(n in key.lower() for n in needles):
                return value
        pytest.fail(f"relative_error_grid has no entry matching {needles!r}")

    bf16_stats = stats_for("bf16")
    fp16_stats = stats_for("fp16", "float16")
    for stats in (bf16_stats, fp16_stats):
        for field in ("max", "mean", "p99"):
            assert field in stats
        assert 0.0 <= stats["mean"] <= stats["max"]
        assert 0.0 <= stats["p99"] <= stats["max"]
    assert bf16_stats["max"] > fp16_stats["max"]


def test_accumulation_precision_fp32_beats_bf16(probe_json: dict):
    """fp32 accumulation of a K-dim matmul must beat bf16 accumulation (D5-4)."""
    exp = get_experiment(probe_json, "accumulation_precision")
    k_dim = exp["k_dim"]
    assert isinstance(k_dim, int) and k_dim >= 512  # quick keeps a meaningful K
    fp32_err = exp["fp32_accum_rel_err"]
    bf16_err = exp["bf16_accum_rel_err"]
    assert fp32_err >= 0.0
    assert bf16_err >= 0.0
    assert fp32_err < bf16_err  # direction: fp32 accumulation strictly better
    assert bf16_err > 1e-4  # magnitude: bf16 accumulation must incur real error
    assert fp32_err < 0.01  # magnitude: fp32 accumulation stays in the low permille


def test_loss_scaling_rescues_fp16_underflow(probe_json: dict):
    """fp16 tiny-gradient underflow vs scale=2^16 recovery (GradScaler semantics)."""
    exp = get_experiment(probe_json, "loss_scaling_simulation")
    assert exp["grad_scale"] == pytest.approx(65536.0)  # 2**16 per schema
    naive_zero = exp["naive_fp16_zero_fraction"]
    recover = exp["scaled_recover_fraction"]
    assert 0.0 <= naive_zero <= 1.0
    assert 0.0 <= recover <= 1.0
    # Most tiny fp16 gradients flush to zero without scaling...
    assert naive_zero > 0.5
    # ...while the scaled path leaves far fewer zeros behind (direction).
    assert naive_zero > 1.0 - recover
    # Unscale-then-check-inf semantics (GradScaler infskip loop) must be exercised.
    assert exp["unscaled_overflow_detected"] is True


def test_abort_mid_experiment_no_partial_json(tmp_path, monkeypatch):
    """ut-abort：实验中途异常（in-process 注入）→ 退出 1 且不写部分 JSON。

    design §6.1 共有用例组第 1 项：仿 tests/test_roofline_script.py 先例以
    进程内 monkeypatch 注入（子进程无法打桩）；原子写（tmp+rename）保证
    半截结果绝不落盘——部分 JSON 会污染可追溯的 docs/results 结果链。
    exp_dtype_limits 注入 RuntimeError 触发单一 exit-1 漏斗。
    """
    import tools.infra.precision_probe as probe

    def boom():
        raise RuntimeError("simulated mid-experiment failure")

    monkeypatch.setattr(probe, "exp_dtype_limits", boom)
    out = tmp_path / "precision_probe.json"
    rc = probe.main(["--quick", "--output-json", str(out)])
    assert rc == 1
    assert not out.exists(), "中断后不得留下部分 JSON"
    assert not out.with_suffix(out.suffix + ".tmp").exists(), "不得留下 .tmp 残件"
