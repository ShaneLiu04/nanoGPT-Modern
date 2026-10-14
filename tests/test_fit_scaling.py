"""tools/infra/fit_scaling.py 缩放律拟合微实验的 TDD 回归锁（AR-v0.7.5 / T005，FR4）。

Red 先行：``tools/infra/fit_scaling.py`` 尚未实现。本模块**只走子进程**调用
``python tools/infra/fit_scaling.py``（不 import 实现模块），因此收集阶段不红、
用例运行阶段红——这是 TDD 的预期状态；实现合入后必须全绿。

契约锚点（design.md §4.3.1 / §4.3.2 / §6.1 / §6.4 / §6.5，srs.md §3.4 FR4）：
- CLI：``[--quick] [--output-json PATH] [--data PATH] [--budget PATH]``，默认数据源
  docs/results/ablation-summary.json + docs/results/budget.json（cwd=仓库根解析）；
- 退出码三态：0 成功（含不收敛诚实降级）；1 数据源缺失（--data/--budget 任一不存在，
  stderr 含实际探测路径、不写 JSON）；2 用法错误（argparse / --output-json 父目录缺失）；
- fit_quality 必含 n_points/method/identifiability_note/converged/r_squared/
  param_uncertainty{alpha_std,A_std,E_std,method}/limitations（非空数组）；
- 真实 7 组数据 D 恒定（同预算训练）→ 主拟合 Kaplan 式 L(N)=E+A/N^α，
  identifiability_note 存在且 kaplan_fit 不输出联合式 B/β 参数；
  Chinchilla 联合式仅在合成数据上演示（chinchilla_synthetic_demo）；
- 预算计算器手算口径：FLOPs=6ND（utils/infra_math.py 对照），文献默认
  α_N≈0.34 / α_D≈0.28 / D/N≈20（Hoffmann 口径）；
- 本仓记账三件套（budget.json 溯源）：corpus_tokens=1,244,464,437、
  consumed_tokens=18000×32768=589,824,000、tokens_per_param=1244464437/50e6≈24.9。

合成 Kaplan 参数恢复用例的数据定标（文献量级 E=3.0, A=1e3, α=0.1，N∈[1e6,1e9]
取 15 点）：噪声取 0.2% 相对噪声（种子 1337 固定）——多初值 Levenberg-Marquardt
恢复仿真显示 0.2% 噪声下 10 个种子的 α 相对误差最差 3.5%，对 10% 验收界留足裕度
（1% 噪声下最差 15.6% 会击穿界，故不取）。

断言约定（同 tests/test_roofline_script.py）：
- 整数锚点（token 数）用 ``==`` 精确断言；浮点用 ``pytest.approx``；
- 真实 7 点拟合不硬断言 R² 阈值（FR4「残差合理**或诚实记录不达」），
  只断言 r_squared 数值已落盘与字段齐全；
- 子进程 stderr/stdout 按 UTF-8 解码（Windows 默认代码页防乱码）。
"""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "infra" / "fit_scaling.py"
REAL_ABLATION = REPO_ROOT / "docs" / "results" / "ablation-summary.json"
REAL_BUDGET = REPO_ROOT / "docs" / "results" / "budget.json"

# 本仓记账锚点（budget.json / design §4.3.2 repo_budget_review，精确断言）
CORPUS_TOKENS = 1244464437
MAX_ITERS = 18000
TOKENS_PER_ITER = 32768
CONSUMED_TOKENS = MAX_ITERS * TOKENS_PER_ITER  # = 589,824,000
REPO_PARAMS_M = 50
TOKENS_PER_PARAM = CORPUS_TOKENS / (REPO_PARAMS_M * 1e6)  # ≈ 24.89

# 预算计算器算例锚点：本仓 50M 全量一纪元的 FLOPs（6ND 口径）
REPO_FLOPS = 6 * REPO_PARAMS_M * 1e6 * CORPUS_TOKENS

# 合成 Kaplan 数据定标（文献量级，见模块 docstring 的噪声论证）
SYN_E, SYN_A, SYN_ALPHA = 3.0, 1e3, 0.1
SYN_N_POINTS = 15
SYN_NOISE = 0.002
SYN_SEED = 1337

# scaling_fit.json 顶层键（design §4.3.2 schema，存在性断言）
TOP_LEVEL_KEYS = (
    "meta",
    "fit_quality",
    "kaplan_fit",
    "chinchilla_synthetic_demo",
    "predictions",
    "budget_calculator",
    "repo_budget_review",
)

# budget_calculator.cases 每例必含字段（design §4.3.2 / FR4 训练时长字段）
BUDGET_CASE_KEYS = {
    "name",
    "flops_budget",
    "N_star",
    "D_star",
    "D_over_N",
    "train_iters_at_repo_throughput",
    "train_hours_at_repo_throughput",
}


# ---------------------------------------------------------------------------
# 子进程调用与数据源构造
# ---------------------------------------------------------------------------


def run_tool(*args):
    """以仓库根为 cwd 子进程调用 fit_scaling（默认相对数据源才能解析）。"""
    return subprocess.run(
        [sys.executable, str(TOOL_PATH), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        timeout=300,
    )


def write_ablation_file(path, groups):
    """构造 ablation-summary 格式数据源（工具只依赖 groups[].status/params_m/
    best_val_loss 语义，其余字段省略）。"""
    payload = {"groups": list(groups)}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def copy_budget_file(path):
    """合法 budget.json：直接复制仓库真实文件（train_tokens/max_iters/
    tokens_per_iter 等字段齐全）。"""
    shutil.copy(REAL_BUDGET, path)
    return path


def _synthetic_kaplan_groups():
    """L = E + A/N^α（+0.2% 相对噪声，种子固定）的 15 点合成消融数据。

    N∈[1e6,1e9] 对应 params_m∈[1,1000]；ablation groups 以 params_m(百万)
    记参数量、best_val_loss 记验证损失。
    """
    n = np.logspace(6, 9, SYN_N_POINTS)
    rng = np.random.default_rng(SYN_SEED)
    loss = (SYN_E + SYN_A / n**SYN_ALPHA) * (1 + rng.normal(0, SYN_NOISE, SYN_N_POINTS))
    return [
        {
            "name": f"syn-{i:02d}",
            "status": "completed",
            "params_m": float(n_i / 1e6),
            "best_val_loss": float(l_i),
            "final_val_loss": float(l_i),
        }
        for i, (n_i, l_i) in enumerate(zip(n, loss))
    ]


def _constant_loss_groups(n_points=15, loss=3.5):
    """病态数据：损失全相同（零方差目标）→ 不收敛路径。"""
    n = np.logspace(6, 9, n_points)
    return [
        {
            "name": f"flat-{i:02d}",
            "status": "completed",
            "params_m": float(n_i / 1e6),
            "best_val_loss": loss,
            "final_val_loss": loss,
        }
        for i, n_i in enumerate(n)
    ]


@pytest.fixture(scope="session")
def real_run(tmp_path_factory):
    """默认参数端到端跑一次真实 7 点数据（不传 --data/--budget，输出进临时目录，
    不弄脏 docs/results）。整文件共享这一次子进程调用。"""
    out = tmp_path_factory.mktemp("real_run") / "scaling_fit.json"
    result = run_tool("--output-json", str(out))
    assert result.returncode == 0, f"fit_scaling 默认参数运行失败:\n{result.stderr}"
    return json.loads(out.read_text(encoding="utf-8"))


def _norm_key(key):
    return key.lower().replace("_", "").replace("-", "")


def _get_by_norm(mapping, norm):
    """按键名宽松取值（alpha_n/alpha_N/alphaN 等写法归一后匹配）。"""
    for key, value in mapping.items():
        if _norm_key(key) == norm:
            return value
    raise KeyError(f"no key normalizing to {norm!r} in {sorted(mapping)}")


# ---------------------------------------------------------------------------
# 共有用例组（--help / --quick 冒烟 / 非法 flag / --output-json 父目录缺失）
# ---------------------------------------------------------------------------


def test_help_available():
    """ut-help：--help 退出 0 且打印用法（三工具共有组）。"""
    result = run_tool("--help")
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
    print("[OK] fit_scaling --help exits 0 with usage text")


def test_quick_smoke_writes_json(tmp_path):
    """ut-quick：--quick 冒烟（真实数据源默认路径）→ 退出 0 + JSON 落盘，
    meta.quick=True 且全部顶层 schema 键齐全（规模缩、不缺实验）。"""
    out = tmp_path / "scaling_fit_quick.json"
    result = run_tool("--quick", "--output-json", str(out))
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    for key in TOP_LEVEL_KEYS:
        assert key in data, f"--quick 档缺顶层键 {key}"
    assert data["meta"]["quick"] is True
    print("[OK] fit_scaling --quick smoke: exit 0, JSON written, meta.quick=True")


def test_unknown_flag_exits_2():
    """ut-bad-flag：未知 flag → argparse 退出 2（§4.3.1 用法错误语义）。"""
    result = run_tool("--definitely-not-a-flag")
    assert result.returncode == 2
    assert result.stderr.strip()
    print("[OK] fit_scaling unknown flag exits 2")


def test_output_json_missing_parent_exits_2(tmp_path):
    """ut-out-parent：--output-json 父目录不存在 → 退出 2、不隐式建目录
    （数据源在位，确保触发的确是输出路径校验分支而非数据源守卫）。"""
    data_path = write_ablation_file(tmp_path / "ablation.json", _synthetic_kaplan_groups())
    budget_path = copy_budget_file(tmp_path / "budget.json")
    out = tmp_path / "no_such_dir" / "out.json"
    result = run_tool(
        "--data", str(data_path), "--budget", str(budget_path), "--output-json", str(out)
    )
    assert result.returncode == 2
    assert result.stderr.strip()
    assert not (tmp_path / "no_such_dir").exists(), "不得隐式创建输出目录"
    print("[OK] fit_scaling --output-json missing parent exits 2, no implicit mkdir")


# ---------------------------------------------------------------------------
# 合成 Kaplan 参数恢复（FR4-1：拟合入口的数值正确性）
# ---------------------------------------------------------------------------


def test_synthetic_kaplan_param_recovery(tmp_path):
    """ut-syn-kaplan：E=3.0, A=1e3, α=0.1 + 0.2% 噪声 → α 相对误差 <10%。

    拟合入口走完整 CLI 链（临时 --data/--budget + --output-json），与
    实现内部选用的多初值策略解耦；converged=True 且 R²>0.95（合成数据
    噪声极小，可硬断言——真实 7 点另见 smoke，不硬断言阈值）。
    """
    data_path = write_ablation_file(tmp_path / "syn-ablation.json", _synthetic_kaplan_groups())
    budget_path = copy_budget_file(tmp_path / "budget.json")
    out = tmp_path / "scaling_fit.json"
    result = run_tool(
        "--data", str(data_path), "--budget", str(budget_path), "--output-json", str(out)
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))

    fit_quality = data["fit_quality"]
    assert fit_quality["n_points"] == SYN_N_POINTS
    assert fit_quality["converged"] is True
    assert fit_quality["r_squared"] > 0.95

    kaplan = data["kaplan_fit"]
    assert kaplan["alpha"] == pytest.approx(
        SYN_ALPHA, rel=0.1
    ), f"合成 α 恢复失败: true={SYN_ALPHA}, fitted={kaplan['alpha']}"
    print(f"[OK] synthetic Kaplan alpha recovered: {kaplan['alpha']:.4f} (true {SYN_ALPHA})")


# ---------------------------------------------------------------------------
# 真实 7 点数据（共享 real_run fixture）：schema / 不可辨识守卫 / 外推 / 预算
# ---------------------------------------------------------------------------


def test_real_data_smoke_schema(real_run):
    """ut-smoke：真实 7 点拟合字段齐全（FR4-1 诚实报告字段存在性断言）。

    不硬断言 R² 阈值（FR4 允许「R²>0.9 **或** 诚实记录不达」），
    只断言 r_squared 数值已落盘、param_uncertainty 三字段与 method 在位、
    limitations 非空、kaplan_fit 三列表与 7 点对齐。
    """
    for key in TOP_LEVEL_KEYS:
        assert key in real_run, f"缺顶层键 {key}"

    meta = real_run["meta"]
    assert meta["tool"] == "fit_scaling"
    assert meta["device"] == "cpu"
    assert isinstance(meta["timestamp"], str) and meta["timestamp"]
    sources = meta["data_sources"]
    assert any("ablation-summary" in str(s) for s in sources)
    assert any("budget" in str(s) for s in sources)

    fit_quality = real_run["fit_quality"]
    assert fit_quality["n_points"] == 7
    assert isinstance(fit_quality["method"], str) and fit_quality["method"]
    assert isinstance(fit_quality["converged"], bool)
    assert isinstance(fit_quality["r_squared"], (int, float))
    assert math.isfinite(fit_quality["r_squared"]), "r_squared 数值必须落盘"
    uncertainty = fit_quality["param_uncertainty"]
    assert {"alpha_std", "A_std", "E_std", "method"} <= set(uncertainty)
    assert isinstance(uncertainty["method"], str) and uncertainty["method"]
    if fit_quality["converged"]:
        for key in ("alpha_std", "A_std", "E_std"):
            assert math.isfinite(float(uncertainty[key]))
    assert isinstance(fit_quality["limitations"], list)
    assert len(fit_quality["limitations"]) > 0

    kaplan = real_run["kaplan_fit"]
    for key in ("E", "A", "alpha", "r_squared", "params_m", "val_loss", "fitted"):
        assert key in kaplan, f"kaplan_fit 缺 {key}"
    assert len(kaplan["params_m"]) == 7
    assert len(kaplan["val_loss"]) == 7
    assert len(kaplan["fitted"]) == 7
    print("[OK] real 7-point fit smoke: full schema, r_squared on disk, limitations non-empty")


def test_identifiability_note_and_no_joint_params(real_run):
    """ut-identifiability：D 恒定（7 组同预算）→ identifiability_note 在位，
    且主拟合 kaplan_fit 不输出联合式 B/β 参数（不可辨识参数不得伪装成结果）。"""
    note = real_run["fit_quality"]["identifiability_note"]
    assert isinstance(note, str) and note.strip()

    kaplan = real_run["kaplan_fit"]
    for key in kaplan:
        normalized = key.lower()
        assert "beta" not in normalized, f"kaplan_fit 不应含 β 参数: {key}"
        assert normalized != "b" and not normalized.startswith(
            "b_"
        ), f"kaplan_fit 不应含 B 参数: {key}"
    print("[OK] identifiability note present; no B/beta params in kaplan_fit")


def test_chinchilla_synthetic_demo_recovery(real_run):
    """ut-syn-chinchilla：联合式 L(N,D) 合成演示 → recovered α_N/α_D 与
    true 相对误差 <15%（D3：联合式在合成数据上验证，键名宽松匹配）。"""
    demo = real_run["chinchilla_synthetic_demo"]
    true_params = demo["true_params"]
    recovered = demo["recovered_params"]
    assert isinstance(true_params, dict) and isinstance(recovered, dict)
    for norm in ("alphan", "alphad"):
        true_value = _get_by_norm(true_params, norm)
        recovered_value = _get_by_norm(recovered, norm)
        assert recovered_value == pytest.approx(
            true_value, rel=0.15
        ), f"{norm}: true={true_value}, recovered={recovered_value}"
    print("[OK] chinchilla synthetic demo: alpha_N/alpha_D recovered within 15%")


def test_predictions_have_positive_width_intervals(real_run):
    """ut-predictions：外推预测表非空，每例含 N/D/pred_loss/interval，
    区间为 [lo, hi] 且宽度 >0（教学级区间也必须非退化）。"""
    predictions = real_run["predictions"]
    assert isinstance(predictions, list) and len(predictions) > 0
    for pred in predictions:
        for key in ("name", "N", "D", "pred_loss", "interval"):
            assert key in pred, f"prediction 缺 {key}: {sorted(pred)}"
        low, high = pred["interval"]
        assert math.isfinite(low) and math.isfinite(high)
        assert high > low, f"区间宽度必须 >0: {pred['interval']}"
        assert math.isfinite(float(pred["pred_loss"]))
    print(f"[OK] {len(predictions)} predictions with positive-width intervals")


def test_budget_calculator_hand_check(real_run):
    """ut-budget-calc：FLOPs=6×50e6×1.24e9 算例手算对照（FR3-1）。

    Hoffmann 文献口径（α_N≈0.34/α_D≈0.28/D/N≈20）下：
    - 6·N*·D* ≈ flops_budget（预算守恒）；
    - D_over_N ≈ 20；
    - train_iters_at_repo_throughput ≈ D*/tokens_per_iter(=32768，budget.json)。
    每例必含 7 个契约字段（含两个训练时长字段，FR4）。
    """
    calculator = real_run["budget_calculator"]
    cases = calculator["cases"]
    assert isinstance(cases, list) and len(cases) > 0
    for case in cases:
        assert BUDGET_CASE_KEYS <= set(case), f"预算算例缺字段: {sorted(case)}"

    literature = calculator["literature_defaults"]
    assert literature["alpha_n"] == pytest.approx(0.34, abs=0.01)
    assert literature["alpha_d"] == pytest.approx(0.28, abs=0.01)
    assert literature["d_over_n"] == pytest.approx(20, rel=0.05)
    assert "6" in str(calculator["flops_per_token_formula"])

    matched = [
        case for case in cases if math.isclose(case["flops_budget"], REPO_FLOPS, rel_tol=0.01)
    ]
    assert matched, (
        f"应存在 flops_budget≈6*50e6*{CORPUS_TOKENS}={REPO_FLOPS:.3e} 的算例, "
        f"实际: {[case['flops_budget'] for case in cases]}"
    )
    case = matched[0]
    assert case["N_star"] > 0 and case["D_star"] > 0
    assert case["D_over_N"] == pytest.approx(20, rel=0.1)
    assert 6 * case["N_star"] * case["D_star"] == pytest.approx(case["flops_budget"], rel=0.02)
    assert case["train_iters_at_repo_throughput"] == pytest.approx(
        case["D_star"] / TOKENS_PER_ITER, rel=0.05
    )
    assert case["train_hours_at_repo_throughput"] > 0
    print(
        f"[OK] budget case at {REPO_FLOPS:.3e} FLOPs: D/N={case['D_over_N']}, "
        f"iters≈{case['train_iters_at_repo_throughput']:.0f}"
    )


def test_tokens_per_param_matches_repo_budget(real_run):
    """ut-tpp：tokens_per_param = 1244464437/50e6 ≈ 24.9（FR4-2，
    与手册复盘「50M/1.24B≈25:1 vs Chinchilla 20:1」同源）。"""
    review = real_run["repo_budget_review"]
    assert review["train_tokens"] == CORPUS_TOKENS
    assert review["params_m"] == REPO_PARAMS_M
    tokens_per_param = review["tokens_per_param"]
    assert tokens_per_param == pytest.approx(TOKENS_PER_PARAM, rel=1e-3)
    assert tokens_per_param == pytest.approx(24.9, abs=0.1)
    assert review["chinchilla_ref"] == pytest.approx(20, rel=0.05)
    assert isinstance(review["verdict"], str) and review["verdict"]
    print(f"[OK] tokens_per_param = {tokens_per_param:.2f} (vs Chinchilla 20)")


def test_bookkeeping_consumed_tokens(real_run):
    """ut-bookkeeping：consumed = 18000×32768 = 589,824,000；corpus = 1,244,464,437
    （记账口径三件套：0.59B 实际消耗 vs 1.24B 一纪元全量，FR3-3 溯源 budget.json）。"""
    bookkeeping = real_run["repo_budget_review"]["bookkeeping"]
    assert bookkeeping["corpus_tokens"] == CORPUS_TOKENS
    assert bookkeeping["consumed_tokens"] == CONSUMED_TOKENS
    assert CONSUMED_TOKENS == MAX_ITERS * TOKENS_PER_ITER == 589_824_000
    print(f"[OK] bookkeeping: corpus={CORPUS_TOKENS}, consumed={CONSUMED_TOKENS}")


# ---------------------------------------------------------------------------
# 数据源守卫（§4.3.1：--data/--budget 各自独立校验 → 退出 1 不写 JSON）
# ---------------------------------------------------------------------------


def test_data_flag_missing_exits_1(tmp_path):
    """ut-data-missing：仅 --data 指向不存在路径（--budget 在位）→ 退出 1、
    stderr 含实际探测路径、不写 JSON。"""
    budget_path = copy_budget_file(tmp_path / "budget.json")
    missing_data = tmp_path / "missing-data.json"
    out = tmp_path / "out.json"
    result = run_tool(
        "--data", str(missing_data), "--budget", str(budget_path), "--output-json", str(out)
    )
    assert result.returncode == 1
    assert "missing-data.json" in result.stderr, f"stderr 应含探测路径:\n{result.stderr}"
    assert not out.exists(), "数据源缺失时不得写 JSON"
    print("[OK] --data missing: exit 1, probed path in stderr, no JSON written")


def test_budget_flag_missing_exits_1(tmp_path):
    """ut-budget-missing：仅 --budget 缺失（--data 在位）→ 对称验证退出 1。"""
    data_path = write_ablation_file(tmp_path / "ablation.json", _synthetic_kaplan_groups())
    missing_budget = tmp_path / "missing-budget.json"
    out = tmp_path / "out.json"
    result = run_tool(
        "--data", str(data_path), "--budget", str(missing_budget), "--output-json", str(out)
    )
    assert result.returncode == 1
    assert "missing-budget.json" in result.stderr, f"stderr 应含探测路径:\n{result.stderr}"
    assert not out.exists(), "数据源缺失时不得写 JSON"
    print("[OK] --budget missing: exit 1, probed path in stderr, no JSON written")


# ---------------------------------------------------------------------------
# 不收敛与边界降级（§6.4/§6.5：converged=false 诚实降级，退出 0 仍产出 JSON）
# ---------------------------------------------------------------------------


def _assert_honest_degradation(result, out_path):
    assert result.returncode == 0, f"降级路径必须退出 0:\n{result.stderr}"
    assert out_path.exists(), "不收敛仍须产出 JSON（诚实降级而非失败）"
    data = json.loads(out_path.read_text(encoding="utf-8"))
    fit_quality = data["fit_quality"]
    assert fit_quality["converged"] is False
    assert isinstance(fit_quality["limitations"], list)
    assert len(fit_quality["limitations"]) > 0
    return data


def test_constant_loss_degrades_honestly(tmp_path):
    """ut-nonconv：病态数据（loss 全相同 → 零方差目标）→ converged=false、
    fit_quality 仍在、limitations 非空、退出 0（不收敛是诚实记录不是失败）。"""
    data_path = write_ablation_file(tmp_path / "flat-ablation.json", _constant_loss_groups())
    budget_path = copy_budget_file(tmp_path / "budget.json")
    out = tmp_path / "scaling_fit.json"
    result = run_tool(
        "--data", str(data_path), "--budget", str(budget_path), "--output-json", str(out)
    )
    _assert_honest_degradation(result, out)
    print("[OK] constant-loss degenerate data: converged=false, JSON written, exit 0")


def test_too_few_points_degrades(tmp_path):
    """ut-few-points：过滤后可用点 ≤2（3 组中 1 组 status=failed）→ 点数不足
    拟合 → 同不收敛降级路径：converged=false、退出 0（§6.4 边界行）。"""
    groups = _constant_loss_groups(3)[:2]
    groups.append({"name": "crashed", "status": "failed", "params_m": 10.0, "best_val_loss": None})
    data_path = write_ablation_file(tmp_path / "tiny-ablation.json", groups)
    budget_path = copy_budget_file(tmp_path / "budget.json")
    out = tmp_path / "scaling_fit.json"
    result = run_tool(
        "--data", str(data_path), "--budget", str(budget_path), "--output-json", str(out)
    )
    _assert_honest_degradation(result, out)
    print("[OK] <=2 usable points: converged=false, exit 0")


def test_empty_groups_degrades(tmp_path):
    """ut-empty-groups：空 groups 数组 → 同降级路径，不抛未捕获异常、退出 0
    （§6.4 边界行）。"""
    data_path = write_ablation_file(tmp_path / "empty-ablation.json", [])
    budget_path = copy_budget_file(tmp_path / "budget.json")
    out = tmp_path / "scaling_fit.json"
    result = run_tool(
        "--data", str(data_path), "--budget", str(budget_path), "--output-json", str(out)
    )
    _assert_honest_degradation(result, out)
    print("[OK] empty groups: converged=false, exit 0, no crash")
