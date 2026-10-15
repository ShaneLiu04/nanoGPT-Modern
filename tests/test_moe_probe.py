"""Contract tests for ``tools/infra/moe_probe.py`` (AR-v0.8.5-moe-long-context, T007, E-C-6).

TDD note: RED -- ``tools/infra/moe_probe.py`` does not exist yet; every test in
this module is expected to fail until the implementation lands (task T007).

Contract under test (design.md I8 -- pure-CPU MoE capacity/load probe):

* CLI: ``python tools/infra/moe_probe.py [--quick] [--output-json PATH]
  [--checkpoint PATH] [--seed INT]``; default output
  ``docs/results/moe_probe.json`` (never exercised here -- every run passes
  ``--output-json`` into pytest tmp storage).
* Exit codes: 0 success **including guard degradation** (missing/incompatible
  ``--checkpoint`` -> skip that source, record a warning, still exit 0);
  1 run error / mid-experiment interrupt (atomic tmp+rename write means no
  partial JSON and no ``.tmp`` leftover); 2 invalid CLI args (e.g. ``--seed
  abc``).
* Synthetic setup: ModernGPTConfig(n_layer=1, n_head=2, n_embd=32,
  block_size=64, num_experts=8, dropout=0.0) instantiating model SwiGLU, with
  synthetic input ``x ~ randn(2, 128, 32)`` (B=2, T=128 -> T_total=256;
  ``--quick`` shrinks to T=64 -> T_total=128).
* JSON schema::

      {"meta":     {"tool": "moe_probe", "timestamp": str,
                    "seed": int, "torch_version": str},
       "config":   {"E": 8, "T_total": 256, "capacity_factors": [0.5, 1.0, 1.25],
                    "hidden": 32, "source": "synthetic"},   # hidden == n_embd
       "experiments": [
         {"name": "capacity_sweep", "source": "synthetic", "metrics": {
            "sweep": [{"cf": float, "capacity": int, "dropped": int,
                       "drop_rate": float, "dropped_rows_zero": bool}, ...]}},
         {"name": "load_distribution", "source": "synthetic", "metrics": {
            "fractions": [float x E], "entropy": float, "max_min_ratio": float}}},
         {"name": "skew_injection", "source": "synthetic", "metrics": {
            "curve": [{"k": int, "overflow_rate": float}, ...]}}],
       "warnings": [str, ...]}    # clean synthetic run -> []

  ``--quick``: capacity factors {0.5, 1.0}, T=64, skew factors {1, 2}.
  ``capacity_sweep`` mirrors the model forward (model/modern_gpt.py L1064 /
  L1070-1074): ``counts = bincount(selected, minlength=E)``,
  ``capacity = int(cf * T_total / E)``, ``dropped = sum(max(counts -
  capacity, 0))`` and cross-checks via a real forward that every dropped
  token's output row is exactly zero (``dropped_rows_zero`` per row).
* Importable pure function (module-level, tested in-process):

      dispatch_stats(x_flat, router_logits, capacity_factor, num_experts)
      -> {"selected":   LongTensor[N],   # top-1 expert per token (softmax argmax)
          "counts":     LongTensor[E],   # bincount(selected, minlength=E)
          "capacity":   int,             # int(cf * N / E), truncating like forward
          "dropped":    int,             # sum(max(counts - capacity, 0))
          "drop_rate":  float,           # dropped / N
          "keep_mask":  BoolTensor[N]}   # True = kept, ORIGINAL token order

  The module must also expose module-level experiment functions
  ``exp_capacity_sweep`` / ``exp_load_distribution`` / ``exp_skew_injection``
  (monkeypatch targets for the interrupt test) and ``main(argv) -> int``.

Seeding note (BS1/BS2): at T_total=256 the cf=1.0 drop rate of an unbiased
router is a multinomial fluctuation (~9.5% mean over random seeds, <5% chance
of landing under 0.05), so the *numeric* laws below pin explicit seeds that
were verified against the exact construction order used here
(``torch.manual_seed(seed) -> SwiGLU(cfg) -> x = torch.randn(2, 128, 32)``):

* seed 17: drop_rate(cf) = {0.5: 0.50, 1.0: 0.039, 1.25: 0.0}  (BS1/BS3)
* seed 0:  skew overflow_rate(k) = [0.102, 0.121, 0.160]       (BS2, strictly up)

Subprocess JSON assertions only use seed-independent laws (drop_rate is
non-increasing in cf because capacity grows monotonically; cf=0.5 always
drops >= 128/256 because sum(counts)=256 > 8*16).

Case map: IT1 = test_full_run_* / test_json_* / test_meta_fields /
test_config_fields / test_experiments_* / test_capacity_sweep_metrics_schema /
test_capacity_direction_from_json / test_load_distribution_metrics_schema /
test_skew_curve_metrics_schema; IT2 = test_missing_checkpoint_guard_degrades;
IT3 = test_abort_mid_experiment_no_partial_json; IT4 = test_invalid_seed_;
IT5 = test_quick_run_under_budget; BS1 = test_dispatch_stats_contract +
test_bs1_capacity_direction_law (+ the JSON direction law in IT1);
BS2 = test_bs2_skew_amplification_law; BS3 = test_bs3_mirror_matches_forward.
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from model.modern_gpt import ModernGPTConfig, SwiGLU

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "infra" / "moe_probe.py"

NUM_EXPERTS = 8
N_EMBD = 32
N_TOKENS = 2 * 128  # B=2, T=128 -> T_total for the full (non-quick) mode
T_TOTAL_QUICK = 2 * 64  # --quick: T=64
FULL_CAPACITY_FACTORS = [0.5, 1.0, 1.25]
QUICK_CAPACITY_FACTORS = [0.5, 1.0]
FULL_SKEW_KS = [1, 2, 4]
QUICK_SKEW_KS = [1, 2]
REQUIRED_EXPERIMENTS = {"capacity_sweep", "load_distribution", "skew_injection"}
QUICK_BUDGET_SECONDS = 30.0
SUBPROCESS_TIMEOUT_SECONDS = 120


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the probe CLI as a subprocess from the repo root.

    PYTHONPATH is injected because ``python tools/infra/moe_probe.py`` puts
    ``tools/infra`` (not the repo root) on sys.path[0], and the tool imports
    ``model.modern_gpt``.
    """
    assert TOOL.is_file(), f"tool not implemented yet: {TOOL}"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_experiment(payload: dict, name: str) -> dict:
    """Fetch one experiment record by name; fail loudly if it is missing."""
    for exp in payload["experiments"]:
        if exp["name"] == name:
            return exp
    pytest.fail(f"experiment {name!r} missing from moe_probe.json")


def _probe():
    """Import the probe module lazily; RED until tools/infra/moe_probe.py lands."""
    import tools.infra.moe_probe as probe

    return probe


def build_moe(seed: int) -> tuple[SwiGLU, torch.Tensor]:
    """Synthetic E=8 SwiGLU (dropout=0) + x ~ randn(2, 128, 32) under a pinned seed.

    The RNG consumption order (manual_seed -> SwiGLU init -> randn input) is
    part of the pinned-seed contract documented in the module docstring.
    """
    torch.manual_seed(seed)
    cfg = ModernGPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=N_EMBD,
        block_size=64,
        num_experts=NUM_EXPERTS,
        dropout=0.0,
    )
    moe = SwiGLU(cfg).eval()
    x = torch.randn(2, 128, N_EMBD)
    return moe, x


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One shared full-mode (no --quick, no --seed, no --checkpoint) subprocess run."""
    out = tmp_path_factory.mktemp("moe_probe") / "moe_probe.json"
    proc = run_cli("--output-json", str(out))
    return {"proc": proc, "out": out}


@pytest.fixture(scope="module")
def probe_json(full_run: dict) -> dict:
    """Parsed product JSON of the shared full run (atomicity: it must load)."""
    proc = full_run["proc"]
    assert proc.returncode == 0, proc.stderr
    return load_json(full_run["out"])


# ---------------------------------------------------------------------------
# IT1: full-run CLI contract + JSON schema
# ---------------------------------------------------------------------------


def test_full_run_exits_zero_and_writes_json(full_run: dict):
    """IT1: no-arg full run (only --output_json) exits 0 and writes the JSON."""
    proc = full_run["proc"]
    assert proc.returncode == 0, proc.stderr
    assert full_run["out"].is_file()


def test_json_top_level_schema(probe_json: dict):
    """IT1: four-part schema {meta, config, experiments, warnings}."""
    assert set(probe_json) >= {"meta", "config", "experiments", "warnings"}
    assert isinstance(probe_json["experiments"], list) and probe_json["experiments"]
    # A clean synthetic run (no --checkpoint) must record no warnings.
    assert probe_json["warnings"] == []


def test_meta_fields(probe_json: dict):
    """IT1: meta carries tool/timestamp/seed/torch_version."""
    meta = probe_json["meta"]
    assert meta["tool"] == "moe_probe"
    assert isinstance(meta["timestamp"], str) and meta["timestamp"]
    assert isinstance(meta["seed"], int)
    # Same interpreter as the subprocess: the recorded version must be honest.
    assert meta["torch_version"] == torch.__version__


def test_config_fields(probe_json: dict):
    """IT1: config mirrors the synthetic setup (E=8, T_total=256, hidden=32)."""
    config = probe_json["config"]
    assert config["E"] == NUM_EXPERTS
    assert config["T_total"] == N_TOKENS
    assert config["capacity_factors"] == FULL_CAPACITY_FACTORS
    assert config["hidden"] == N_EMBD  # hidden == n_embd of the synthetic config
    assert config["source"] == "synthetic"


def test_experiments_present_with_synthetic_source(probe_json: dict):
    """IT1: all three experiments present, each sourced "synthetic" with metrics."""
    experiments = probe_json["experiments"]
    names = {exp["name"] for exp in experiments}
    assert names == REQUIRED_EXPERIMENTS
    for exp in experiments:
        assert exp["source"] == "synthetic"
        assert isinstance(exp["metrics"], dict)


def test_capacity_sweep_metrics_schema(probe_json: dict):
    """IT1: capacity_sweep rows mirror the forward formula and pass the zero check."""
    sweep = get_experiment(probe_json, "capacity_sweep")["metrics"]["sweep"]
    assert isinstance(sweep, list) and len(sweep) == len(FULL_CAPACITY_FACTORS)
    rows = {row["cf"]: row for row in sweep}
    assert set(rows) == set(FULL_CAPACITY_FACTORS)
    for cf, row in rows.items():
        assert row["capacity"] == int(cf * N_TOKENS / NUM_EXPERTS)
        assert isinstance(row["dropped"], int) and row["dropped"] >= 0
        assert row["drop_rate"] == pytest.approx(row["dropped"] / N_TOKENS)
        # Forward cross-check: dropped tokens' output rows are exactly zero.
        assert row["dropped_rows_zero"] is True


def test_capacity_direction_from_json(probe_json: dict):
    """IT1/BS1 (subprocess half): seed-independent capacity direction law.

    drop_rate is non-increasing in cf (capacity grows monotonically), and
    cf=0.5 always drops at least half the tokens: sum(counts)=256 while
    8 experts x capacity 16 can hold at most 128.
    """
    rows = {
        row["cf"]: row for row in get_experiment(probe_json, "capacity_sweep")["metrics"]["sweep"]
    }
    assert rows[0.5]["drop_rate"] >= 0.5
    assert rows[0.5]["drop_rate"] > rows[1.0]["drop_rate"] >= rows[1.25]["drop_rate"]


def test_load_distribution_metrics_schema(probe_json: dict):
    """IT1: unbiased-router load stats -- fractions partition unity, entropy <= ln E."""
    metrics = get_experiment(probe_json, "load_distribution")["metrics"]
    fractions = metrics["fractions"]
    assert isinstance(fractions, list) and len(fractions) == NUM_EXPERTS
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert sum(fractions) == pytest.approx(1.0)
    assert 0.0 < metrics["entropy"] <= math.log(NUM_EXPERTS) + 1e-6
    assert metrics["max_min_ratio"] >= 1.0


def test_skew_curve_metrics_schema(probe_json: dict):
    """IT1: skew_injection records the (k, overflow_rate) curve for k in {1, 2, 4}."""
    curve = get_experiment(probe_json, "skew_injection")["metrics"]["curve"]
    assert [point["k"] for point in curve] == FULL_SKEW_KS
    for point in curve:
        assert 0.0 <= point["overflow_rate"] <= 1.0


# ---------------------------------------------------------------------------
# IT2: --checkpoint guard degradation
# ---------------------------------------------------------------------------


def test_missing_checkpoint_guard_degrades(tmp_path: Path):
    """IT2: --checkpoint -> nonexistent file: exit 0, warning recorded, synthetic full.

    Guard degradation must never lose the synthetic experiments and never
    fail the run (design I8: missing/incompatible checkpoint -> skip that
    source, record in warnings, exit 0).
    """
    missing = tmp_path / "no_such_checkpoint.pt"
    out = tmp_path / "guard.json"
    proc = run_cli("--checkpoint", str(missing), "--output-json", str(out))
    assert proc.returncode == 0, f"exit={proc.returncode}\nstderr:\n{proc.stderr}"
    assert out.is_file()
    payload = load_json(out)
    assert payload["config"]["source"] == "synthetic"
    names = {exp["name"] for exp in payload["experiments"]}
    assert names == REQUIRED_EXPERIMENTS
    assert all(exp["source"] == "synthetic" for exp in payload["experiments"])
    warnings = payload["warnings"]
    assert isinstance(warnings, list) and warnings
    assert any("skipped" in str(w).lower() for w in warnings)


# ---------------------------------------------------------------------------
# IT3: mid-experiment interrupt (in-process injection)
# ---------------------------------------------------------------------------


def test_abort_mid_experiment_no_partial_json(tmp_path: Path, monkeypatch):
    """IT3: 实验中途异常（in-process monkeypatch 模块级实验函数）→ 退出 1 且不写部分 JSON。

    子进程无法打桩，仿 tests/test_precision_probe.py 先例以进程内注入触发
    单一 exit-1 漏斗；原子写（tmp+rename）保证半截结果绝不落盘。
    """
    probe = _probe()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated mid-experiment failure")

    monkeypatch.setattr(probe, "exp_load_distribution", boom)
    out = tmp_path / "moe_probe.json"
    rc = probe.main(["--output-json", str(out)])
    assert rc == 1
    assert not out.exists(), "中断后不得留下部分 JSON"
    assert not out.with_suffix(out.suffix + ".tmp").exists(), "不得留下 .tmp 残件"


# ---------------------------------------------------------------------------
# IT4: invalid CLI argument
# ---------------------------------------------------------------------------


def test_invalid_seed_exits_two(tmp_path: Path):
    """IT4: --seed abc is an invalid argument: exit 2, stderr message, no artifact."""
    out = tmp_path / "unused.json"
    proc = run_cli("--seed", "abc", "--output-json", str(out))
    assert proc.returncode == 2
    assert proc.stderr.strip()
    assert not out.exists()


# ---------------------------------------------------------------------------
# IT5: --quick runtime budget
# ---------------------------------------------------------------------------


def test_quick_run_under_budget(tmp_path: Path):
    """IT5: --quick full subprocess stays under 30 s, exits 0, and shrinks the grid."""
    out = tmp_path / "quick.json"
    start = time.monotonic()
    proc = run_cli("--quick", "--output-json", str(out))
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, f"exit={proc.returncode}\nstderr:\n{proc.stderr}"
    assert elapsed < QUICK_BUDGET_SECONDS
    assert out.is_file()
    payload = load_json(out)
    assert payload["config"]["capacity_factors"] == QUICK_CAPACITY_FACTORS
    assert payload["config"]["T_total"] == T_TOTAL_QUICK  # B=2 x T=64
    curve = get_experiment(payload, "skew_injection")["metrics"]["curve"]
    assert [point["k"] for point in curve] == QUICK_SKEW_KS


# ---------------------------------------------------------------------------
# BS1/BS3 shared: dispatch_stats pure-function contract
# ---------------------------------------------------------------------------


def test_dispatch_stats_contract():
    """BS1/BS3 前置：dispatch_stats 纯函数契约（镜像公式，与 forward L1064/L1070-1074 同语义）。"""
    probe = _probe()
    moe, x = build_moe(seed=17)
    x_flat = x.view(N_TOKENS, N_EMBD)
    router_logits = moe.router(x_flat)
    stats = probe.dispatch_stats(
        x_flat, router_logits, capacity_factor=1.25, num_experts=NUM_EXPERTS
    )
    assert set(stats) >= {
        "selected",
        "counts",
        "capacity",
        "dropped",
        "drop_rate",
        "keep_mask",
    }
    # selected: top-1 of softmax(router_logits), identical to the forward path.
    probs = torch.softmax(router_logits, dim=-1)
    top1 = torch.topk(probs, 1, dim=-1).indices.squeeze(-1)
    assert torch.equal(stats["selected"], top1)
    counts = stats["counts"]
    assert counts.shape == (NUM_EXPERTS,)
    assert int(counts.sum()) == N_TOKENS
    # capacity: truncating int(), same as model/modern_gpt.py L1064.
    assert stats["capacity"] == int(1.25 * N_TOKENS / NUM_EXPERTS)
    dropped = stats["dropped"]
    assert dropped == int(torch.clamp(counts - stats["capacity"], min=0).sum())
    assert stats["drop_rate"] == pytest.approx(dropped / N_TOKENS)
    keep = stats["keep_mask"]
    assert keep.shape == (N_TOKENS,)
    assert keep.dtype == torch.bool
    assert int(keep.sum()) == N_TOKENS - dropped


# ---------------------------------------------------------------------------
# BS1: capacity direction law (in-process, pinned seed)
# ---------------------------------------------------------------------------


def test_bs1_capacity_direction_law():
    """BS1 容量方向律：cf=0.5 丢弃多于 cf=1.0；无偏 router 在 cf=1.0 丢弃率 < 0.05。

    seed=17 (verified): drop_rate = {0.5: 0.50, 1.0: 0.039, 1.25: 0.0}。
    cf=0.5 档 rate >= 0.5 是种子无关下界：counts 总和 256，而 8 专家 x 容量
    16 最多容纳 128 个 token。
    """
    probe = _probe()
    moe, x = build_moe(seed=17)
    x_flat = x.view(N_TOKENS, N_EMBD)
    router_logits = moe.router(x_flat)
    rates = {}
    for cf in FULL_CAPACITY_FACTORS:
        stats = probe.dispatch_stats(
            x_flat, router_logits, capacity_factor=cf, num_experts=NUM_EXPERTS
        )
        rates[cf] = stats["drop_rate"]
    assert rates[0.5] >= 0.5
    assert rates[0.5] > rates[1.0]
    assert rates[1.0] < 0.05
    assert rates[1.25] <= rates[1.0]


# ---------------------------------------------------------------------------
# BS2: skew amplification law (in-process, pinned seed)
# ---------------------------------------------------------------------------


def test_bs2_skew_amplification_law():
    """BS2 偏斜放大律：cf=1.0 下 router.weight[:, 0] *= k 的溢出率随 k 单调不减。

    seed=0 (verified): overflow_rate = [0.102, 0.121, 0.160]，k=4 严格高于基线。
    """
    probe = _probe()
    moe, x = build_moe(seed=0)
    x_flat = x.view(N_TOKENS, N_EMBD)
    base_weight = moe.router.weight.detach().clone()
    rates = []
    for k in FULL_SKEW_KS:
        with torch.no_grad():
            moe.router.weight.copy_(base_weight)
            moe.router.weight[:, 0] *= k
        router_logits = moe.router(x_flat)
        stats = probe.dispatch_stats(
            x_flat, router_logits, capacity_factor=1.0, num_experts=NUM_EXPERTS
        )
        rates.append(stats["drop_rate"])
    assert rates[0] > 0.0  # unbiased baseline already overflows a bit at cf=1.0
    assert rates[0] <= rates[1] <= rates[2]
    assert rates[2] > rates[0]


# ---------------------------------------------------------------------------
# BS3: mirror == real forward (in-process, pinned seed)
# ---------------------------------------------------------------------------


def test_bs3_mirror_matches_forward():
    """BS3 镜像=真实：dispatch_stats 的 dropped 与真实 forward 的零行数完全一致。

    seed=17, cf=0.5 (verified): 镜像 dropped=128；forward 中被丢 token 行恒为
    零（out_flat 仅对 keep 赋值），保留行全部非零，故零行数恰为 128。
    dropout=0 且 eval 模式，前向确定。
    """
    probe = _probe()
    moe, x = build_moe(seed=17)
    x_flat = x.view(N_TOKENS, N_EMBD)
    router_logits = moe.router(x_flat)
    stats = probe.dispatch_stats(
        x_flat, router_logits, capacity_factor=0.5, num_experts=NUM_EXPERTS
    )
    dropped = stats["dropped"]
    assert dropped > 0
    keep_mask = stats["keep_mask"]
    dropped_mask = ~keep_mask
    assert dropped_mask.any()

    moe.moe_capacity_factor = 0.5  # same capacity semantics as the mirror call
    out, _ = moe(x)
    out_flat = out.reshape(N_TOKENS, N_EMBD)
    # Every dropped token's output row is exactly zero.
    assert not out_flat[dropped_mask].any()
    # No kept row collapses to zero, so zero rows isolate exactly the dropped set.
    assert out_flat[keep_mask].any(dim=1).all()
    zero_rows = ~out_flat.any(dim=1)
    assert int(zero_rows.sum()) == dropped
