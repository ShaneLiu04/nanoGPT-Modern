"""tools/infra/triton_kernels.py Triton 微实验三件套的 TDD 回归锁（AR-v0.7.0 / T003，FR3+FR4）。

Red 先行：``tools/infra/triton_kernels.py`` 尚未实现，本模块在收集阶段即因
import 失败而红——这是 TDD 的预期状态；实现合入（design.md §4.3 接口契约：
三 kernel 包装 / eager 参照 / 强度公式 / bench+CLI）后必须全绿。

两层降级互不越界（design §4.4）：
- 模块级 ``pytest.importorskip("triton")``：无 triton 的 CI 环境整文件 skip
  （srs FR4-AC1「全部通过或 skip，零失败」），且必须先于被测模块导入——
  否则 Red 阶段的 ModuleNotFoundError 会在无 triton 的 CI 上变成收集失败而非
  合规 skip；
- CUDA 组用例再叠 ``skipif(not torch.cuda.is_available())``。

断言约定（同 tests/test_roofline_script.py / tests/test_infra_math.py）：
- 整数锚点（FLOPs/字节数）用 ``==`` 精确断言，预期值写成公式形式表达口径；
- fp32 一致性 atol=1e-5/rtol=1e-4，bf16 atol=1e-2/rtol=1e-2
  （srs FR3 两口径容差，design 裁定 D5：bf16 差异来自输入舍入与归约顺序）；
- 包装层异常契约（design §4.3.1）：形状/dtype/device 不符一律 ValueError；
- 涉及 torch.cuda 的用例一律 monkeypatch，不在 CI 上要求真实 GPU
  （test_bench_interrupt_no_partial_json 的可测路径选择见其 docstring）。
"""

import json

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

import tools.infra.triton_kernels as tk
from tools.infra.triton_kernels import (
    eager_gqa_decode,
    eager_rmsnorm,
    eager_swiglu,
    gqa_decode_bytes,
    gqa_decode_flops,
    main,
    rmsnorm_bytes,
    rmsnorm_flops,
    swiglu_bytes,
    swiglu_flops,
    triton_gqa_decode,
    triton_rmsnorm,
    triton_swiglu,
)

from model.modern_gpt import RMSNorm, _decode_gqa_attention

# 一致性两口径容差（srs FR3 / design 裁定 D5）
ATOL_FP32, RTOL_FP32 = 1e-5, 1e-4
ATOL_BF16, RTOL_BF16 = 1e-2, 1e-2

# CUDA 一致性用例小尺寸（design §4.4「小尺寸即可」）
SWIGLU_MNK = (128, 64, 32)  # M, N, K
RMSNORM_MN = (128, 64)  # M, N
GQA_B_H_KV_D_S = (2, 8, 4, 32, 256)  # B, n_head, n_kv, D, S

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _host_device() -> str:
    """形状校验负例的宿主设备：有 CUDA 时用 cuda 隔离出形状/dtype 子句，
    CPU-only CI 退回 cpu——此时 device 子句先触发，异常仍为 ValueError
    （契约里所有包装异常均为 ValueError），只是未隔离到具体子句。"""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# eager 参照正确性（CPU；srs FR4-AC2「无 GPU 可测」）
# ---------------------------------------------------------------------------


def test_eager_swiglu_matches_manual():
    """ut-eager-swiglu：F.silu(x@wg.T) * (x@wu.T) vs 手写 einsum 公式 allclose。"""
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    w_gate = torch.randn(6, 8)
    w_up = torch.randn(6, 8)
    got = eager_swiglu(x, w_gate, w_up)
    ref = F.silu(torch.einsum("mk,nk->mn", x, w_gate)) * torch.einsum("mk,nk->mn", x, w_up)
    assert got.shape == (4, 6)
    assert torch.allclose(got, ref, atol=ATOL_FP32, rtol=RTOL_FP32)


def test_eager_rmsnorm_matches_reference_class():
    """ut-eager-rmsnorm：与 model.modern_gpt.RMSNorm 前向 fp32 数值一致（srs FR4-AC2）。"""
    torch.manual_seed(0)
    x = torch.randn(5, 16)
    weight = torch.rand(16) + 0.5  # 非 1 权重，避免掩盖 weight 缩放项
    norm = RMSNorm(16, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(weight)
        ref = norm(x)
    got = eager_rmsnorm(x, weight, eps=1e-5)
    assert torch.allclose(got, ref, atol=ATOL_FP32, rtol=RTOL_FP32)
    # 默认 eps=1e-5（契约签名）与显式传参同值
    assert torch.allclose(eager_rmsnorm(x, weight), got)


def test_eager_gqa_decode_matches_sdpa():
    """ut-eager-gqa-sdpa：grouped 5 维 SDPA（q[B,n_kv,rep,1,D] × k/v[B,n_kv,1,S,D]
    广播）reshape 后 allclose——与 model._grouped_attention 同构的独立参照。"""
    torch.manual_seed(0)
    B, n_head, n_kv, D, S = 2, 8, 4, 32, 64
    rep = n_head // n_kv
    q = torch.randn(B, n_head, 1, D)
    k = torch.randn(B, n_kv, S, D)
    v = torch.randn(B, n_kv, S, D)
    got = eager_gqa_decode(q, k, v)
    ref = F.scaled_dot_product_attention(
        q.view(B, n_kv, rep, 1, D), k.unsqueeze(2), v.unsqueeze(2)
    ).reshape(B, n_head, 1, D)
    assert got.shape == (B, n_head, 1, D)
    assert torch.allclose(got, ref, atol=ATOL_FP32, rtol=RTOL_FP32)


@pytest.mark.parametrize("scale", [None, 0.13])
def test_eager_gqa_decode_matches_decode_helper(scale):
    """ut-eager-gqa-helper：与 _decode_gqa_attention（modern_gpt.py:245）逐式对齐，
    scale=None（默认 1/sqrt(D)）与显式 scale 两参数化。"""
    torch.manual_seed(0)
    B, n_head, n_kv, D, S = 2, 8, 4, 32, 64
    q = torch.randn(B, n_head, 1, D)
    k = torch.randn(B, n_kv, S, D)
    v = torch.randn(B, n_kv, S, D)
    got = eager_gqa_decode(q, k, v, scale=scale)
    ref = _decode_gqa_attention(q, k, v, n_kv, scale=scale)
    assert torch.allclose(got, ref, atol=ATOL_FP32, rtol=RTOL_FP32)


# ---------------------------------------------------------------------------
# 算术强度公式（CPU；design §4.3.1 口径，== 精确断言）
# ---------------------------------------------------------------------------


def test_intensity_formulas_small():
    """ut-intensity：六公式小形状手算逐值（M=3,N=4,K=5 / B=2,n_kv=2,rep=3,S=7,D=8）。"""
    M, N, K = 3, 4, 5
    # gate/up 两 GEMM 4MNK + silu·mul 逐元素 3MN = 240+36
    assert swiglu_flops(M, N, K) == 4 * M * N * K + 3 * M * N == 276
    # 读 x(MK) + wg/wu(2NK) + 写 y(MN) = 67 元素 × 4B
    assert swiglu_bytes(M, N, K, 4) == (M * K + 2 * N * K + M * N) * 4 == 268
    assert swiglu_bytes(M, N, K, 2) == (M * K + 2 * N * K + M * N) * 2  # dtype_size 参与换算
    # 平方和/均值/rsqrt/两次乘 ~5MN = 60
    assert rmsnorm_flops(M, N) == 5 * M * N == 60
    # 读 x(MN)+w(N) + 写 y(MN) = 28 元素 × 4B
    assert rmsnorm_bytes(M, N, 4) == (2 * M * N + N) * 4 == 112
    B, n_kv, rep, S, D = 2, 2, 3, 7, 8
    # scores + AV 两批 GEMM = 4·B·n_kv·rep·S·D = 2688
    assert gqa_decode_flops(B, n_kv, rep, S, D) == 4 * B * n_kv * rep * S * D == 2688
    # q(2·B·n_kv·rep·D) + k&v(2·B·n_kv·S·D) + y(B·n_kv·rep·D) = 736 元素 × 4B
    assert (
        gqa_decode_bytes(B, n_kv, rep, S, D, 4)
        == (2 * B * n_kv * rep * D + 2 * B * n_kv * S * D + B * n_kv * rep * D) * 4
        == 2944
    )


# ---------------------------------------------------------------------------
# 包装层形状校验（design §4.3.1 异常契约：一律 ValueError）
# ---------------------------------------------------------------------------


def _swiglu_case(case: str):
    """swiglu 负例构造：合法基线 x[4,8] / w[6,8] 上注入单一违规。"""
    dev = _host_device()
    x = torch.randn(4, 8, device=dev)
    w_gate = torch.randn(6, 8, device=dev)
    w_up = torch.randn(6, 8, device=dev)
    if case == "w_k_mismatch":
        w_up = torch.randn(6, 5, device=dev)  # K 与 x/w_gate 不一致
    elif case == "x_ndim_3":
        x = x.unsqueeze(0)
    elif case == "dtype_fp64":
        x, w_gate, w_up = (t.double() for t in (x, w_gate, w_up))
    elif case == "device_cpu":
        x, w_gate, w_up = (t.cpu() for t in (x, w_gate, w_up))
    return x, w_gate, w_up


@pytest.mark.parametrize("case", ["w_k_mismatch", "x_ndim_3", "dtype_fp64", "device_cpu"])
def test_triton_wrapper_invalid_shape_swiglu(case):
    """ut-swiglu-invalid：w 形状不匹配（K 不一致）/ x.ndim=3 / fp64 / CPU 输入
    → ValueError（§4.3.1 异常契约）。"""
    x, w_gate, w_up = _swiglu_case(case)
    with pytest.raises(ValueError):
        triton_swiglu(x, w_gate, w_up)


def _rmsnorm_case(case: str):
    """rmsnorm 负例构造：合法基线 x[4,8] / weight[8] 上注入单一违规。"""
    dev = _host_device()
    x = torch.randn(4, 8, device=dev)
    weight = torch.randn(8, device=dev)
    if case == "weight_len_mismatch":
        weight = torch.randn(9, device=dev)  # 长度 ≠ N
    elif case == "x_ndim_3":
        x = x.unsqueeze(0)
    elif case == "dtype_fp64":
        x, weight = x.double(), weight.double()
    elif case == "device_cpu":
        x, weight = x.cpu(), weight.cpu()
    return x, weight


@pytest.mark.parametrize("case", ["weight_len_mismatch", "x_ndim_3", "dtype_fp64", "device_cpu"])
def test_triton_wrapper_invalid_shape_rmsnorm(case):
    """ut-rmsnorm-invalid：weight 长度≠N / x.ndim=3 / fp64 / CPU 输入 → ValueError。"""
    x, weight = _rmsnorm_case(case)
    with pytest.raises(ValueError):
        triton_rmsnorm(x, weight)


def _gqa_case(case: str):
    """gqa 负例构造：合法基线 k/v [B=2,n_kv=2,S=16,D=8] 上注入单一违规。"""
    dev = _host_device()
    B, S, D = 2, 16, 8
    k = torch.randn(B, 2, S, D, device=dev)
    v = torch.randn(B, 2, S, D, device=dev)
    if case == "head_not_divisible":
        q = torch.randn(B, 5, 1, D, device=dev)  # n_head=5 不被 n_kv=2 整除
    elif case == "t_not_1":
        q = torch.randn(B, 4, 2, D, device=dev)  # T=2 ≠ 1（n_head=4 本可整除）
    return q, k, v


@pytest.mark.parametrize("case", ["head_not_divisible", "t_not_1"])
def test_triton_wrapper_invalid_shape_gqa(case):
    """ut-gqa-invalid：n_head 不被 n_kv 整除（5/2）/ q.shape[2]≠1（T=2）→ ValueError。"""
    q, k, v = _gqa_case(case)
    with pytest.raises(ValueError):
        triton_gqa_decode(q, k, v)


# ---------------------------------------------------------------------------
# bench / CLI 语义（design §4.3.3/§4.3.5：退出码 0/1/2；冒烟不写 JSON）
# ---------------------------------------------------------------------------


def test_bench_invalid_kernel_name():
    """ut-bench-name：按 kernel 名计时的入口收到非法名 → ValueError（§4.3.3）。

    接口契约只暴露 run_bench(quick, json_path)，未约定按名计时的可调用签名
    （design §4.4 备注）；实现若提供 ``bench_kernel(name, ...)`` 入口则此处直测，
    否则合规 skip——该防线由 test_cli_usage_error_exit_2（argparse exit 2）承接。
    """
    bench_kernel = getattr(tk, "bench_kernel", None)
    if bench_kernel is None:
        pytest.skip("接口契约未暴露按 kernel 名计时的入口（design §4.4 备注）")
    with pytest.raises(ValueError):
        bench_kernel("no_such_kernel")


def test_cpu_smoke_mode(tmp_path, monkeypatch, capsys):
    """ut-cpu-smoke：无 CUDA → 打印原因 + eager 冒烟（vs 手写公式）→ exit 0 不写 JSON。

    monkeypatch torch.cuda.is_available → False：无 GPU 的 CI 照跑；JSON 绝不
    落盘（无 GPU 计时数据，避免伪数据污染 docs/results 结果链，design 裁定 D7），
    stdout 明示 smoke（design §4.3.3「smoke mode, no JSON written」）。
    """
    monkeypatch.setattr(tk.torch.cuda, "is_available", lambda: False)
    out = tmp_path / "triton_kernels.json"
    rc = main(["--quick", "--json", str(out)])
    assert rc == 0
    assert not out.exists()
    assert "smoke" in capsys.readouterr().out.lower()


def test_cpu_smoke_assertion_failure(tmp_path, monkeypatch):
    """ut-smoke-fail：冒烟断言失败（eager 参照被 monkeypatch 成错误值）→ exit 1。

    同 test_cpu_smoke_mode 打桩 is_available → False 强制走冒烟路径（CUDA 机器
    上同样 deterministic）；错误值取 zeros_like——形状/dtype/device 全合法，
    只有数值错，恰好隔离「冒烟断言」这一行为点。
    """
    monkeypatch.setattr(tk.torch.cuda, "is_available", lambda: False)

    def wrong_rmsnorm(x, weight, eps=1e-5):
        return torch.zeros_like(x)

    monkeypatch.setattr(tk, "eager_rmsnorm", wrong_rmsnorm)
    out = tmp_path / "triton_kernels.json"
    rc = main(["--quick", "--json", str(out)])
    assert rc == 1
    assert not out.exists()


def test_bench_interrupt_no_partial_json(tmp_path, monkeypatch):
    """ut-abort：bench 路径测量异常 → rc=1 且不写部分 JSON（design 裁定 D4 原子性）。

    契约一致的可测路径（design §4.4 备注）：monkeypatch is_available→True 强制走
    bench 路径，再把三个包装（bench 一致性/计时的必经入口）换成中途抛
    RuntimeError——本机 CUDA 环境下 boom 确定性触发；无真实 GPU 的机器上，
    bench 路径随后任何 CUDA 张量构建失败同属「测量中断」，断言语义不变
    （rc=1、不落盘）。get_device_name 一并打桩防元信息采集先炸（同
    tests/test_roofline_script.py 的 test_measurement_abort_no_partial_json）。
    """
    monkeypatch.setattr(tk.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tk.torch.cuda, "get_device_name", lambda *a, **k: "Simulated GPU")

    def boom(*args, **kwargs):
        raise RuntimeError("simulated measurement failure")

    monkeypatch.setattr(tk, "triton_swiglu", boom)
    monkeypatch.setattr(tk, "triton_rmsnorm", boom)
    monkeypatch.setattr(tk, "triton_gqa_decode", boom)
    out = tmp_path / "triton_kernels.json"
    rc = main(["--quick", "--json", str(out)])
    assert rc == 1
    assert not out.exists()


def test_cli_help_smoke(capsys):
    """ut-cli-help：--help 打印用法并 SystemExit(0)（FR3 验收直测）。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "--json" in capsys.readouterr().out


def test_cli_usage_error_exit_2():
    """ut-cli-usage：未知 flag → argparse SystemExit(2)（design §4.3.5 退出码语义）。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["--unknown-flag"])
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# CUDA kernel 数值一致性（srs FR3 两口径容差；CI 无 CUDA 时整组 skip）
# ---------------------------------------------------------------------------


@requires_cuda
def test_triton_swiglu_allclose():
    """ut-triton-swiglu：包装 vs eager 参照 fp32 allclose（atol=1e-5/rtol=1e-4）。"""
    torch.manual_seed(0)
    M, N, K = SWIGLU_MNK
    dev = torch.device("cuda")
    x = torch.randn(M, K, device=dev)
    w_gate = torch.randn(N, K, device=dev)
    w_up = torch.randn(N, K, device=dev)
    got = triton_swiglu(x, w_gate, w_up)
    ref = eager_swiglu(x, w_gate, w_up)
    assert got.shape == (M, N)
    assert got.dtype == x.dtype
    assert got.device.type == "cuda"
    assert torch.allclose(got, ref, atol=ATOL_FP32, rtol=RTOL_FP32)


@requires_cuda
def test_triton_rmsnorm_allclose():
    """ut-triton-rmsnorm：包装 vs eager 参照 fp32 allclose（默认 eps=1e-5 顺带锁定）。"""
    torch.manual_seed(0)
    M, N = RMSNORM_MN
    dev = torch.device("cuda")
    x = torch.randn(M, N, device=dev)
    weight = torch.rand(N, device=dev) + 0.5
    got = triton_rmsnorm(x, weight)
    ref = eager_rmsnorm(x, weight)
    assert got.shape == (M, N)
    assert got.dtype == x.dtype
    assert torch.allclose(got, ref, atol=ATOL_FP32, rtol=RTOL_FP32)


@requires_cuda
@pytest.mark.parametrize("scale", [None, 0.37])
def test_triton_gqa_decode_allclose(scale):
    """ut-triton-gqa：包装 vs eager 参照 fp32 allclose；scale None/显式 两参数化。"""
    torch.manual_seed(0)
    B, n_head, n_kv, D, S = GQA_B_H_KV_D_S
    dev = torch.device("cuda")
    q = torch.randn(B, n_head, 1, D, device=dev)
    k = torch.randn(B, n_kv, S, D, device=dev)
    v = torch.randn(B, n_kv, S, D, device=dev)
    got = triton_gqa_decode(q, k, v, scale=scale)
    ref = eager_gqa_decode(q, k, v, scale=scale)
    assert got.shape == (B, n_head, 1, D)
    assert got.dtype == q.dtype
    assert torch.allclose(got, ref, atol=ATOL_FP32, rtol=RTOL_FP32)


@requires_cuda
def test_triton_consistency_bf16():
    """ut-triton-bf16：三包装 bf16 输入（kernel fp32 累加）vs eager bf16，
    atol=1e-2/rtol=1e-2（design 裁定 D5）。输出 dtype 契约未锁定，宽断言。

    量级选择：两口径差主要来自 eager 参照链的中间 bf16 舍入（gate/up matmul
    出栈、silu 出栈、乘积出栈，复合 ~1% 相对，给定输入即固定、对任何
    fp32 累加的 kernel 侧等价），N(0,1) 输入下 swiglu 乘积幅度可达 ~10²
    使 rtol 项贴边越限（实测 viol≈1.33）。swiglu 输入取 ×0.25（|y|≲1，
    atol 主导，裕度 ~3.5×）；rmsnorm/gqa 幅度天然 O(1)（裕度 0.62/0.33）。
    判别力不受影响：bf16 直算（无 fp32 累加）或公式错误的实现误差远超容差。
    """
    torch.manual_seed(0)
    dev = torch.device("cuda")
    dt = torch.bfloat16

    M, N, K = SWIGLU_MNK
    x = torch.randn(M, K, device=dev, dtype=dt) * 0.25
    w_gate = torch.randn(N, K, device=dev, dtype=dt) * 0.25
    w_up = torch.randn(N, K, device=dev, dtype=dt) * 0.25
    got = triton_swiglu(x, w_gate, w_up)
    assert got.dtype in (dt, torch.float32)
    assert torch.allclose(got, eager_swiglu(x, w_gate, w_up), atol=ATOL_BF16, rtol=RTOL_BF16)

    M, N = RMSNORM_MN
    x = torch.randn(M, N, device=dev, dtype=dt)
    weight = torch.rand(N, device=dev, dtype=dt) + 0.5
    assert torch.allclose(
        triton_rmsnorm(x, weight), eager_rmsnorm(x, weight), atol=ATOL_BF16, rtol=RTOL_BF16
    )

    B, n_head, n_kv, D, S = GQA_B_H_KV_D_S
    q = torch.randn(B, n_head, 1, D, device=dev, dtype=dt)
    k = torch.randn(B, n_kv, S, D, device=dev, dtype=dt)
    v = torch.randn(B, n_kv, S, D, device=dev, dtype=dt)
    assert torch.allclose(
        triton_gqa_decode(q, k, v),
        eager_gqa_decode(q, k, v),
        atol=ATOL_BF16,
        rtol=RTOL_BF16,
    )


@requires_cuda
def test_bench_json_roundtrip(tmp_path):
    """ut-bench-json：--quick --json tmp → rc=0，JSON schema 键齐全（design §4.3.4）。

    measurements 元素含 kernel/size/triton_ms/eager_ms/speedup/
    intensity_flops_per_byte；consistency 元素含 dtype/atol/rtol/allclose，
    且 rc=0 蕴含全部 allclose 通过（§4.3.3 退出码语义），fp32/bf16 两口径
    （裁定 D5）都必须出现。
    """
    out = tmp_path / "triton_kernels.json"
    rc = main(["--quick", "--json", str(out)])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"meta", "measurements", "consistency"}
    assert {"device", "torch", "triton"} <= set(data["meta"])
    assert len(data["measurements"]) >= 3  # 三 kernel（--quick 每 kernel 1 档）
    for m in data["measurements"]:
        assert {
            "kernel",
            "size",
            "triton_ms",
            "eager_ms",
            "speedup",
            "intensity_flops_per_byte",
        } <= set(m)
    assert len(data["consistency"]) >= 3
    for c in data["consistency"]:
        assert {"dtype", "atol", "rtol", "allclose"} <= set(c)
        assert c["allclose"], "rc=0 蕴含一致性全部通过（design §4.3.3 退出码语义）"
    dtypes = {str(c["dtype"]).replace("torch.", "") for c in data["consistency"]}
    assert {"float32", "bfloat16"} <= dtypes
