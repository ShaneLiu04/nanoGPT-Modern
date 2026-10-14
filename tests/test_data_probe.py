"""Tests for ``tools/infra/data_probe.py`` (T006, AR-v0.7.5-precision-scaling-data).

Contract sources: design.md 4.3.1 (CLI + exit-code semantics), 4.3.2
(data_probe.json schema), 6.1/6.4/6.5 (test matrix), srs.md 3.6 (FR6).
Every real-profile case runs against a small synthetic uint16 bin under a
tmp path; the 17GB data/openwebtext/train.bin is never touched (FR6-2).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "infra" / "data_probe.py"

EOT = 50256
STD_N_TOKENS = 20000
STD_EOT_INTERVAL = 100
STD_TOP_TOKEN = 42

REAL_EXPERIMENTS = ("bin_profile", "doc_length_dist")
SYNTHETIC_EXPERIMENTS = (
    "contamination_demo",
    "minhash_dedup_curve",
    "greedy_packing",
    "doc_boundary_stats",
)
EXPECTED_EXPERIMENTS = set(REAL_EXPERIMENTS) | set(SYNTHETIC_EXPERIMENTS)


def run_probe(*args, timeout=240):
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
        timeout=timeout,
    )


def write_bin(path, tokens):
    np.asarray(tokens, dtype=np.uint16).tofile(str(path))


def make_std_bin(path):
    """Synthetic uint16 bin: Zipf-style dominant token, EOT every 100 tokens,
    forced 0 endpoints so value_range is exactly [0, 50256]."""
    rng = np.random.default_rng(20261014)
    tokens = rng.integers(0, EOT, size=STD_N_TOKENS)
    tokens[rng.random(STD_N_TOKENS) < 0.1] = STD_TOP_TOKEN
    tokens[:10] = 0
    tokens[np.arange(STD_EOT_INTERVAL - 1, STD_N_TOKENS, STD_EOT_INTERVAL)] = EOT
    write_bin(path, tokens)


def get_experiment(result, name):
    for exp in result["experiments"]:
        if exp["name"] == name:
            return exp
    raise AssertionError(f"experiment {name!r} missing from data_probe output")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def std_bin(tmp_path_factory):
    path = tmp_path_factory.mktemp("data_probe_bins") / "std_train.bin"
    make_std_bin(path)
    return path


@pytest.fixture(scope="module")
def probe_result(std_bin, tmp_path_factory):
    out = tmp_path_factory.mktemp("data_probe_out") / "quick_result.json"
    proc = run_probe("--bin", str(std_bin), "--quick", "--output-json", str(out))
    assert proc.returncode == 0, f"exit={proc.returncode}\nstderr:\n{proc.stderr}"
    assert out.exists()
    return load_json(out)


# ---------------------------------------------------------------------------
# Common CLI contract group (shared across the three AR tools)
# ---------------------------------------------------------------------------


def test_help_smoke():
    proc = run_probe("--help")
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()


def test_quick_smoke_produces_all_experiments(probe_result):
    """--quick shrinks sample sizes but must not drop any experiment."""
    assert probe_result["meta"]["quick"] is True
    names = {exp["name"] for exp in probe_result["experiments"]}
    assert names == EXPECTED_EXPERIMENTS


def test_unknown_flag_exits_2():
    proc = run_probe("--definitely-not-a-flag")
    assert proc.returncode == 2


def test_output_json_missing_parent_dir_exits_2(std_bin, tmp_path):
    out = tmp_path / "no_such_dir" / "result.json"
    proc = run_probe("--bin", str(std_bin), "--quick", "--output-json", str(out))
    assert proc.returncode == 2
    assert proc.stderr.strip()
    assert not out.exists()


def test_meta_fields_and_experiment_sources(probe_result):
    meta = probe_result["meta"]
    assert meta["tool"] == "data_probe"
    assert meta["device"] == "cpu"
    assert isinstance(meta["quick"], bool)
    assert "timestamp" in meta
    assert "bin_path" in meta
    assert meta["bin_available"] is True
    by_name = {exp["name"]: exp for exp in probe_result["experiments"]}
    assert set(by_name) == EXPECTED_EXPERIMENTS
    for exp in probe_result["experiments"]:
        assert exp["source"] in ("real", "synthetic")
    for name in REAL_EXPERIMENTS:
        assert by_name[name]["source"] == "real"
    for name in SYNTHETIC_EXPERIMENTS:
        assert by_name[name]["source"] == "synthetic"


# ---------------------------------------------------------------------------
# Real-profile experiments on synthetic small bins (no 17GB dependency)
# ---------------------------------------------------------------------------


def test_bin_profile_matches_synthetic_construction(probe_result):
    """EOT at known interval -> measured eot_freq ~ 1/interval; Zipf-style
    dominant token -> top_tokens[0] hits; size/dtype reflect the file."""
    prof = get_experiment(probe_result, "bin_profile")
    assert prof["source"] == "real"
    assert prof["dtype"] == "uint16"
    assert prof["size_bytes"] == STD_N_TOKENS * 2
    assert 0 < prof["sampled_tokens"] <= STD_N_TOKENS
    expected_eot_freq = 1.0 / STD_EOT_INTERVAL
    assert abs(prof["eot_freq"] - expected_eot_freq) < 0.005
    top = prof["top_tokens"]
    assert top
    assert top[0][0] == STD_TOP_TOKEN
    assert top[0][1] > 0.05


def test_value_range_reports_uint16_endpoints(probe_result):
    """Bin contains 0 and EOT=50256 -> value_range must report both endpoints."""
    prof = get_experiment(probe_result, "bin_profile")
    assert prof["value_range"] == [0, EOT]


def test_doc_length_dist_matches_construction(probe_result):
    """Docs are EOT-separated every ~100 tokens -> mean doc length ~ 100."""
    exp = get_experiment(probe_result, "doc_length_dist")
    assert exp["source"] == "real"
    for key in ("mean", "p50", "p99"):
        assert key in exp
    assert 50 <= exp["mean"] <= 200
    assert 0 <= exp["p50"] <= exp["p99"]


def test_value_range_reports_out_of_vocab_values(tmp_path):
    """Out-of-vocab values (60000 > 50256) must be reported honestly, no crash."""
    bin_path = tmp_path / "out_of_vocab.bin"
    rng = np.random.default_rng(7)
    tokens = rng.integers(5, 1000, size=64)
    tokens[0] = 5
    tokens[3] = 60000
    write_bin(bin_path, tokens)
    out = tmp_path / "out_of_vocab.json"
    proc = run_probe("--bin", str(bin_path), "--quick", "--output-json", str(out))
    assert proc.returncode == 0, f"exit={proc.returncode}\nstderr:\n{proc.stderr}"
    prof = get_experiment(load_json(out), "bin_profile")
    assert prof["value_range"] == [5, 60000]


def test_tiny_bin_stride_degradation_no_crash(tmp_path):
    """Fewer tokens than the default sample_n: stride degrades to the full
    bin without out-of-bounds reads or division by zero (6.4 boundary)."""
    bin_path = tmp_path / "tiny.bin"
    tokens = list(range(1, 31)) + [EOT, EOT] + [5] * 5
    write_bin(bin_path, tokens)
    out = tmp_path / "tiny.json"
    proc = run_probe("--bin", str(bin_path), "--output-json", str(out))
    assert proc.returncode == 0, f"exit={proc.returncode}\nstderr:\n{proc.stderr}"
    prof = get_experiment(load_json(out), "bin_profile")
    assert "skipped" not in prof
    assert 0 < prof["sampled_tokens"] <= len(tokens)


def test_missing_bin_guard_degrades(tmp_path):
    """--bin -> nonexistent path: exit 0, bin_profile replaced by the skipped
    marker, all synthetic experiments still run (D4 degradation)."""
    missing = tmp_path / "does_not_exist.bin"
    out = tmp_path / "missing.json"
    proc = run_probe("--bin", str(missing), "--quick", "--output-json", str(out))
    assert proc.returncode == 0, f"exit={proc.returncode}\nstderr:\n{proc.stderr}"
    assert proc.stderr.strip()
    result = load_json(out)
    assert result["meta"]["bin_available"] is False
    prof = get_experiment(result, "bin_profile")
    assert "skipped" in prof
    assert "bin not found" in str(prof["skipped"]).lower()
    for name in SYNTHETIC_EXPERIMENTS:
        assert get_experiment(result, name)["source"] == "synthetic"


# ---------------------------------------------------------------------------
# Synthetic-corpus experiments (tool-internal corpora, output contract)
# ---------------------------------------------------------------------------


def test_contamination_demo_perfect_construction(probe_result):
    """Constructed needle demo: every needle found, zero false positives."""
    exp = get_experiment(probe_result, "contamination_demo")
    assert exp["source"] == "synthetic"
    assert exp["needle_hit_rate"] == pytest.approx(1.0)
    assert exp["false_positive_rate"] == pytest.approx(0.0, abs=1e-9)


def test_minhash_dedup_curve_monotonic(probe_result):
    """detected_rates must be non-decreasing as dup_rates increase."""
    exp = get_experiment(probe_result, "minhash_dedup_curve")
    assert exp["source"] == "synthetic"
    dup_rates = list(exp["dup_rates"])
    detected = list(exp["detected_rates"])
    assert len(dup_rates) == len(detected)
    assert len(dup_rates) >= 2
    assert isinstance(exp["band_widths"], list) and exp["band_widths"]
    pairs = sorted(zip(dup_rates, detected))
    assert pairs[-1][0] > pairs[0][0]
    for (_, d_lo), (_, d_hi) in zip(pairs, pairs[1:]):
        assert d_hi >= d_lo - 1e-9


def test_dedup_semantics_full_dup_and_independent(probe_result):
    """Fully duplicated docs are caught (~1.0); independent docs are barely
    killed (detected stays close to the actual dup rate at the low end)."""
    exp = get_experiment(probe_result, "minhash_dedup_curve")
    pairs = sorted(zip(exp["dup_rates"], exp["detected_rates"]))
    min_dup, det_at_min = pairs[0]
    max_dup, det_at_max = pairs[-1]
    assert det_at_max >= 0.9
    assert det_at_min <= min_dup + 0.1


def test_greedy_packing_fill_ratio(probe_result):
    """Mostly-short docs pack tightly: fill_ratio > 0.9 and padding_waste is
    self-consistent with fill_ratio (they partition the block budget)."""
    exp = get_experiment(probe_result, "greedy_packing")
    assert exp["source"] == "synthetic"
    assert isinstance(exp["block_size"], int) and exp["block_size"] > 0
    fill_ratio = exp["fill_ratio"]
    assert 0.9 < fill_ratio <= 1.0
    assert abs(fill_ratio + exp["padding_waste"] - 1.0) <= 0.05


def test_doc_boundary_stats_bounds(probe_result):
    """Greedy packing places at least one doc per block; cross-doc fraction
    is a probability."""
    exp = get_experiment(probe_result, "doc_boundary_stats")
    assert exp["source"] == "synthetic"
    assert exp["docs_per_block_mean"] >= 1.0
    assert 0.0 <= exp["cross_doc_fraction"] <= 1.0


def test_abort_mid_experiment_no_partial_json(tmp_path, monkeypatch):
    """ut-abort：实验中途异常（in-process 注入）→ 退出 1 且不写部分 JSON。

    design §6.1 共有用例组第 1 项：仿 tests/test_roofline_script.py 先例以
    进程内 monkeypatch 注入（子进程无法打桩）；原子写（tmp+rename）保证
    半截结果绝不落盘。--bin 指向不存在路径走真实剖析守卫降级，
    exp_contamination_demo 注入 RuntimeError 触发单一 exit-1 漏斗。
    """
    import tools.infra.data_probe as data_probe

    def boom():
        raise RuntimeError("simulated mid-experiment failure")

    monkeypatch.setattr(data_probe, "exp_contamination_demo", boom)
    out = tmp_path / "data_probe.json"
    rc = data_probe.main(
        [
            "--bin",
            str(tmp_path / "nonexistent.bin"),
            "--quick",
            "--output-json",
            str(out),
        ]
    )
    assert rc == 1
    assert not out.exists(), "中断后不得留下部分 JSON"
    assert not out.with_suffix(out.suffix + ".tmp").exists(), "不得留下 .tmp 残件"
