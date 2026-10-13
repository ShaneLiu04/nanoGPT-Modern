"""tools/infra/run_roofline.py 屋顶线微实验的 TDD 回归锁（AR-v0.6.0 / T004，FR4）。

Red 先行：``tools/infra/run_roofline.py``（及 tools/infra/ 包）尚未实现，本模块在
收集阶段即因 import 失败而红——这是 TDD 的预期状态；实现合入
（design §6.1 roofline 表契约）后必须全绿。

GPU_SPECS 不猜分母原则：spec_bandwidth_gbs 只在卡名可子串匹配到已知规格表时
给出，否则为 None——绝不拿未知卡的测量值冒充规格值，也绝不用猜测的
bandwidth/ridge 点位对未知卡下结论。测量（effective_gbps）与规格
（spec_bandwidth_gbs）是两个口径，JSON 里分列。

本卡规格锚点（NVIDIA 专业卡，nvidia-smi 实测显存 16384 MiB）：
- Quadro RTX 5000：GDDR6 16 GiB / 带宽 448.0 GB/s（十进制）/ L2 4.0 MiB
  / FP32 ≈ 11.0 TFLOPS；
- 工作集跨越判据：64³·4B = 1 MiB < L2 4 MiB < 8192³·4B = 2 GiB
  ——SIZES_FULL 必须同时含 64 与 8192，才能画出「L2 内 → 跨 L2 → 显存」三段
  屋顶线（design §6.1 表的横轴覆盖）。

断言约定（同 tests/test_infra_math.py）：
- 整数锚点（FLOPs/字节数）用 ``==`` 精确断言；
- 浮点（GB/s 换算）用 ``pytest.approx``；
- 异常断言用 ``pytest.raises``；
- 涉及 torch.cuda 的用例一律 monkeypatch，不在 CI 上要求真实 GPU。
"""

import json

import pytest

import tools.infra.run_roofline as roofline
from tools.infra.run_roofline import (
    GPU_SPECS,
    SIZES_FULL,
    SIZES_QUICK,
    effective_bandwidth_gbs,
    gemm_flops,
    gemm_sweep,
    gemm_working_set_bytes,
    gemv_bandwidth,
    get_args,
    main,
    summarize,
)

# summarize 返回的顶层键集合（design §6.1 接口契约，精确断言）
SUMMARY_KEYS = {
    "version",
    "mode",
    "gpu_name",
    "spec_bandwidth_gbs",
    "segments",
}

# 本卡（Quadro RTX 5000）规格锚点：nvidia-smi 16384 MiB / 448 GB/s / L2 4 MiB
LOCAL_GPU_NAME = "Quadro RTX 5000"
LOCAL_SPEC_BANDWIDTH_GBS = 448.0


# ---------------------------------------------------------------------------
# SIZES_FULL / SIZES_QUICK 尺寸表
# ---------------------------------------------------------------------------


def test_sizes_quick_subset_full():
    """ut-sizes：SIZES_QUICK ⊆ SIZES_FULL；两端 64 与 8192 跨越 L2 4MiB 工作集。

    64³·4B = 1 MiB < L2 4 MiB（L2 内段）而 8192³·4B = 2 GiB（显存段），
    缺任一端点屋顶线就画不出三段结构；quick 表 ≥10 点保证 --quick 仍可跑全横轴。
    """
    assert set(SIZES_QUICK) <= set(SIZES_FULL)
    assert len(SIZES_QUICK) >= 10
    assert len(SIZES_FULL) >= 12
    assert min(SIZES_FULL) <= 64
    assert max(SIZES_FULL) >= 8192
    assert 64 in SIZES_FULL
    assert 8192 in SIZES_FULL


# ---------------------------------------------------------------------------
# 纯公式：gemm_flops / gemm_working_set_bytes / effective_bandwidth_gbs
# ---------------------------------------------------------------------------


def test_gemm_flops_and_working_set_formula():
    """ut-formula：GEMM FLOPs = 2·M·N·K；工作集 = (MN+NK+MK)·dtype_bytes。

    小矩阵手算逐值：gemm_flops(2,3,4) = 2·2·3·4 = 48；gemm_working_set_bytes(2,3,4)
    = (6+12+8)·4 = 104（默认 fp32 dtype_bytes=4）；
    1024³ 锚点 = 2·1024³ FLOPs（与 utils/infra_math 的强度公式共用口径）。
    """
    assert gemm_flops(2, 3, 4) == 2 * 2 * 3 * 4 == 48
    assert gemm_working_set_bytes(2, 3, 4) == (6 + 12 + 8) * 4 == 104
    assert gemm_flops(1024, 1024, 1024) == 2 * 1024**3
    assert gemm_working_set_bytes(1024, 1024, 1024) == 3 * 1024 * 1024 * 4


def test_effective_bandwidth_gbs():
    """ut-bw：有效带宽 = bytes/seconds/1e9（十进制 GB/s）；seconds ≤ 0 → ValueError。

    锚点：448e9 B 在 1.0 s 内搬完 = 448.0 GB/s（恰为本卡规格带宽——
    规格表的 448 也是十进制口径，两边口径一致）；1e9 B / 0.5 s = 2.0 GB/s。
    """
    assert effective_bandwidth_gbs(448e9, 1.0) == pytest.approx(448.0)
    assert effective_bandwidth_gbs(1e9, 0.5) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        effective_bandwidth_gbs(448e9, 0.0)
    with pytest.raises(ValueError):
        effective_bandwidth_gbs(448e9, -0.1)


# ---------------------------------------------------------------------------
# summarize：JSON 顶层 schema + GPU_SPECS 子串匹配（不猜分母）
# ---------------------------------------------------------------------------


def test_summarize_schema_and_spec_match():
    """ut-summarize：顶层键集合锁定；已知卡子串匹配 → 448.0；未知卡 → None。

    GPU_SPECS 不猜分母原则：spec_bandwidth_gbs 只能来自规格表的子串匹配
    （"Quadro RTX 5000" 命中），未知卡名一律 None，绝不用测量值或猜测值冒充。
    segments 必含 gemm/gemv/decode 三段键（decode 可为 None）。
    """
    spec = GPU_SPECS[LOCAL_GPU_NAME]
    assert spec["bandwidth_gbs"] == pytest.approx(LOCAL_SPEC_BANDWIDTH_GBS)
    assert spec["l2_cache_mib"] == pytest.approx(4.0)
    assert spec["fp32_tflops"] == pytest.approx(11.0, abs=1.0)

    s = summarize([], [], None, LOCAL_GPU_NAME, "full")
    assert set(s) == SUMMARY_KEYS
    assert s["spec_bandwidth_gbs"] == pytest.approx(LOCAL_SPEC_BANDWIDTH_GBS)
    assert set(s["segments"]) >= {"gemm", "gemv", "decode"}
    assert s["segments"]["decode"] is None

    unknown = summarize([], [], None, "Unknown GPU", "full")
    assert set(unknown) == SUMMARY_KEYS
    assert unknown["spec_bandwidth_gbs"] is None


# ---------------------------------------------------------------------------
# get_args：CLI 参数
# ---------------------------------------------------------------------------


def test_get_args_flags_and_sizes():
    """ut-args：--quick/--skip_decode 旗标、--sizes 逗号解析、默认值锁定。

    默认 out 指向 docs/results/roofline.json（结果可追溯目录），
    默认 iters=10、两旗标 False。
    """
    a = get_args(["--quick", "--skip_decode"])
    assert a.quick is True
    assert a.skip_decode is True

    b = get_args(["--sizes", "64,128,256"])
    assert b.sizes == [64, 128, 256]

    d = get_args([])
    assert d.quick is False
    assert d.skip_decode is False
    assert d.iters == 10
    assert "roofline.json" in d.out


# ---------------------------------------------------------------------------
# 测量段入参校验（空列表 → ValueError，不触发实际测量）
# ---------------------------------------------------------------------------


def test_segment_invalid_inputs():
    """ut-invalid-input：空 sizes / 空 shapes → ValueError（快速失败，不静默）。"""
    with pytest.raises(ValueError):
        gemm_sweep("cpu", [])
    with pytest.raises(ValueError):
        gemv_bandwidth("cpu", [])


def test_segment_invalid_device():
    """ut-invalid-device：非法 device → ValueError（design §4.3.2 异常契约）。

    不依赖 torch.device 的 RuntimeError：函数级白名单校验先于张量构建，
    三段函数一致（gemm/gemv/decode）。
    """
    with pytest.raises(ValueError, match="device"):
        gemm_sweep("bogus", [64])
    with pytest.raises(ValueError, match="device"):
        gemv_bandwidth("tpu", [(2, 2)])
    with pytest.raises(ValueError, match="device"):
        roofline.decode_step_bandwidth("mps")


# ---------------------------------------------------------------------------
# main：CPU 冒烟路径（monkeypatch torch.cuda，CI 无 GPU 也能验收）
# ---------------------------------------------------------------------------


def test_cpu_smoke_mode(tmp_path, monkeypatch):
    """ut-cpu-smoke：CUDA 不可用 → CPU 小矩阵冒烟、mode="cpu-smoke"、退出 0。

    monkeypatch torch.cuda.is_available → False：无 GPU 的 CI 环境照跑；
    JSON 必须落盘且 mode == "cpu-smoke"（与三段全量测量的 "full" 区分），
    gpu_name 键存在（CPU 时值为 "cpu" 或 None，宽断言）。
    """
    monkeypatch.setattr(roofline.torch.cuda, "is_available", lambda: False)
    out = tmp_path / "roofline.json"
    rc = main(["--out", str(out)])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "cpu-smoke"
    assert "gpu_name" in data
    assert data["gpu_name"] in ("cpu", None)


def test_measurement_abort_no_partial_json(tmp_path, monkeypatch):
    """ut-abort：测量段异常 → 退出 1 且不写部分 JSON（原子性验收）。

    monkeypatch cuda 可用 + gemm_sweep 抛 RuntimeError：main 必须整体失败
    （rc=1），绝不把半截测量结果落盘——部分 JSON 会污染可追溯的
    docs/results 结果链。
    """
    monkeypatch.setattr(roofline.torch.cuda, "is_available", lambda: True)
    # 防御：无真实 GPU 的机器上 get_device_name 不可调，一并打桩（不影响被测语义）
    monkeypatch.setattr(roofline.torch.cuda, "get_device_name", lambda *a, **k: LOCAL_GPU_NAME)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated gemm_sweep failure")

    monkeypatch.setattr(roofline, "gemm_sweep", boom)
    out = tmp_path / "roofline.json"
    rc = main(["--out", str(out)])
    assert rc == 1
    assert not out.exists()
