"""Triton 微实验三件套：fused SwiGLU / fused RMSNorm / GQA decode（AR-v0.7.0 / T003，FR3）。

教学级手写 kernel + PyTorch eager 参照 + A/B 计时 + 数值一致性（fp32/bf16 两口径）
+ 算术强度标注。独立微实验，不接入模型（用户裁定）。

分层降级（design.md §4.3.3 / 裁定 D7）：
- 本机 CUDA + triton：全量 bench → ``docs/results/triton_kernels.json``（原子落盘，
  测量中断绝不写部分 JSON，裁定 D4）+ stdout Markdown 摘要，退出码 0；
- 无 CUDA 或无 triton：打印原因 → CPU eager 参照冒烟（vs 手写公式断言）→
  成功退出码 0（不写 JSON），失败退出码 1；
- argparse 用法错误退出码 2（内置）。

一致性容差（srs FR3 / design 裁定 D5）：fp32 atol=1e-5/rtol=1e-4；
bf16 atol=1e-2/rtol=1e-2（kernel 一律 fp32 累加，差异来自输入舍入与归约顺序）。

CLI：``python tools/infra/triton_kernels.py [--quick] [--json PATH] [--selftest]``。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:  # 无 triton 的环境照常 import：CLI 走 CPU 冒烟路径（design §4.3.3）
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - 环境相关
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

ATOL_FP32, RTOL_FP32 = 1e-5, 1e-4
ATOL_BF16, RTOL_BF16 = 1e-2, 1e-2

DEFAULT_JSON = Path(__file__).resolve().parents[2] / "docs" / "results" / "triton_kernels.json"

_ALLOWED_DTYPES = (torch.float32, torch.bfloat16)


# ---------------------------------------------------------------------------
# eager 参照（纯 PyTorch，CPU 可跑；亦是冒烟对象与一致性基准）
# ---------------------------------------------------------------------------


def eager_swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """F.silu(x @ w_gate.T) * (x @ w_up.T)，对齐 MLP 非 MoE 路径。"""
    return F.silu(x @ w_gate.transpose(-1, -2)) * (x @ w_up.transpose(-1, -2))


def eager_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) * weight，公式对齐 model/modern_gpt.py RMSNorm。"""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def eager_gqa_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """T==1 GQA decode，逻辑逐式对齐 model/modern_gpt.py::_decode_gqa_attention。"""
    B, n_head, T, D = q.shape
    n_kv = k.shape[1]
    rep = n_head // n_kv
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    q_g = q.view(B, n_kv, rep, D)
    att = (q_g @ k.transpose(-1, -2)) * scale
    att = torch.softmax(att, dim=-1)
    y = att @ v
    return y.reshape(B, n_head, T, D)


# ---------------------------------------------------------------------------
# 算术强度公式（口径与 utils/infra_math 一致：fp32 元素 4B，乘加计 2 FLOPs）
# ---------------------------------------------------------------------------


def swiglu_flops(M: int, N: int, K: int) -> int:
    """gate/up 两 GEMM 4MNK + silu·mul 逐元素 3MN。"""
    return 4 * M * N * K + 3 * M * N


def swiglu_bytes(M: int, N: int, K: int, dtype_size: int) -> int:
    """读 x(MK) + wg/wu(2NK) + 写 y(MN)。"""
    return (M * K + 2 * N * K + M * N) * dtype_size


def rmsnorm_flops(M: int, N: int) -> int:
    """平方和/均值/rsqrt/两次乘 ~5MN。"""
    return 5 * M * N


def rmsnorm_bytes(M: int, N: int, dtype_size: int) -> int:
    """读 x(MN)+w(N) + 写 y(MN)。"""
    return (2 * M * N + N) * dtype_size


def gqa_decode_flops(B: int, n_kv: int, rep: int, S: int, D: int) -> int:
    """scores + AV 两批 GEMM：4·B·n_kv·rep·S·D。"""
    return 4 * B * n_kv * rep * S * D


def gqa_decode_bytes(B: int, n_kv: int, rep: int, S: int, D: int, dtype_size: int) -> int:
    """q(2·B·n_kv·rep·D) + k&v(2·B·n_kv·S·D) + y(B·n_kv·rep·D)。"""
    return (2 * B * n_kv * rep * D + 2 * B * n_kv * S * D + B * n_kv * rep * D) * dtype_size


# ---------------------------------------------------------------------------
# Triton kernels（仅基础特性；sm75 无 TMA/warp spec，教学定位）
# ---------------------------------------------------------------------------

if triton is not None and tl is not None:

    @triton.jit  # type: ignore[misc]
    def _swiglu_kernel(
        x_ptr,
        wg_ptr,
        wu_ptr,
        y_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_wg_n,
        stride_wu_n,
        stride_ym,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # 逐 (BLOCK_M, BLOCK_N) tile：acc_g/acc_u fp32 累加，y = silu(g) * u。
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            x_blk = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + kk[None, :],
                mask=(offs_m[:, None] < M) & (kk[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            wg_blk = tl.load(
                wg_ptr + offs_n[:, None] * stride_wg_n + kk[None, :],
                mask=(offs_n[:, None] < N) & (kk[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            wu_blk = tl.load(
                wu_ptr + offs_n[:, None] * stride_wu_n + kk[None, :],
                mask=(offs_n[:, None] < N) & (kk[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            acc_g += tl.dot(x_blk, tl.trans(wg_blk), input_precision="ieee")
            acc_u += tl.dot(x_blk, tl.trans(wu_blk), input_precision="ieee")
        y = acc_g * tl.sigmoid(acc_g) * acc_u
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit  # type: ignore[misc]
    def _rmsnorm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        N,
        eps,
        stride_xm,
        stride_ym,
        BLOCK_N: tl.constexpr,
    ):
        # 每程序一行：ms = mean(row*row)（掩码补零参与求和，除真实 N）；
        # y = row * rsqrt(ms + eps) * w。
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        x_blk = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        ms = tl.sum(x_blk * x_blk, axis=0) / N
        rstd = 1.0 / tl.sqrt(ms + eps)
        w_blk = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x_blk * rstd * w_blk
        tl.store(y_ptr + row * stride_ym + offs, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit  # type: ignore[misc]
    def _gqa_decode_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        y_ptr,
        S,
        scale,
        n_kv,
        stride_qb,
        stride_qh,
        stride_kb,
        stride_kh,
        stride_vb,
        stride_vh,
        stride_yb,
        stride_yh,
        REP: tl.constexpr,
        D: tl.constexpr,
        BLOCK_S: tl.constexpr,
        REP_PAD: tl.constexpr,
    ):
        # grid=(B*n_kv,)：每程序一个 (b, kv_head)。q 块 [REP_PAD, D] 常驻寄存器
        # （行掩码，pad 行不落盘）；沿 S 分块在线 softmax（m/l/acc 寄存器递推，
        # 与 ring_attention 推导同构——attention-kernels.md §4）。
        pid = tl.program_id(0)
        b = pid // n_kv
        kvh = pid % n_kv
        offs_r = tl.arange(0, REP_PAD)
        offs_d = tl.arange(0, D)
        r_mask = offs_r < REP
        q_blk = tl.load(
            q_ptr + b * stride_qb + (kvh * REP + offs_r)[:, None] * stride_qh + offs_d[None, :],
            mask=r_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        m = tl.full((REP_PAD,), float("-inf"), dtype=tl.float32)
        lse = tl.zeros((REP_PAD,), dtype=tl.float32)
        acc = tl.zeros((REP_PAD, D), dtype=tl.float32)
        for s0 in range(0, S, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            s_mask = offs_s < S
            k_blk = tl.load(
                k_ptr + b * stride_kb + kvh * stride_kh + offs_s[:, None] * D + offs_d[None, :],
                mask=s_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            v_blk = tl.load(
                v_ptr + b * stride_vb + kvh * stride_vh + offs_s[:, None] * D + offs_d[None, :],
                mask=s_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            s_blk = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee") * scale
            m_new = tl.maximum(m, tl.max(s_blk, axis=1))
            # 全 -inf 行（pad）的 exp(-inf - -inf) 会 NaN：替零防染毒
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(s_blk - m_safe[:, None])
            alpha = tl.exp(m - m_safe)
            lse = lse * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, v_blk, input_precision="ieee")
            m = m_new
        l_safe = tl.where(lse == 0.0, 1.0, lse)
        y = acc / l_safe[:, None]
        tl.store(
            y_ptr + b * stride_yb + (kvh * REP + offs_r)[:, None] * stride_yh + offs_d[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=r_mask[:, None],
        )


# ---------------------------------------------------------------------------
# Python 包装：形状/dtype/device 校验（异常一律 ValueError）+ grid + launch
# ---------------------------------------------------------------------------


def _require_2d(t: torch.Tensor, name: str) -> None:
    if t.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got ndim={t.ndim}")


def _require_dtype(t: torch.Tensor, name: str) -> None:
    if t.dtype not in _ALLOWED_DTYPES:
        raise ValueError(f"{name} dtype must be float32/bfloat16, got {t.dtype}")


def _require_cuda(t: torch.Tensor, name: str) -> None:
    if t.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA, got {t.device}")


def _require_triton() -> None:
    if triton is None:
        raise RuntimeError("triton not installed; only CPU smoke path is available")


def triton_swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """x [M,K]、w_gate/w_up [N,K] → y [M,N]。"""
    _require_2d(x, "x")
    _require_2d(w_gate, "w_gate")
    _require_2d(w_up, "w_up")
    _require_dtype(x, "x")
    _require_dtype(w_gate, "w_gate")
    _require_dtype(w_up, "w_up")
    _require_cuda(x, "x")
    _require_cuda(w_gate, "w_gate")
    _require_cuda(w_up, "w_up")
    if w_gate.shape != w_up.shape or w_gate.shape[1] != x.shape[1]:
        raise ValueError(
            f"shape mismatch: x {tuple(x.shape)}, w_gate {tuple(w_gate.shape)}, "
            f"w_up {tuple(w_up.shape)}"
        )
    _require_triton()
    M, K = x.shape
    N = w_gate.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))
    _swiglu_kernel[grid](
        x,
        w_gate,
        w_up,
        y,
        M,
        N,
        K,
        x.stride(0),
        w_gate.stride(0),
        w_up.stride(0),
        y.stride(0),
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=16,
    )
    return y


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """x [M,N]、weight [N] → y [M,N]。"""
    _require_2d(x, "x")
    if weight.ndim != 1:
        raise ValueError(f"weight must be 1-D, got ndim={weight.ndim}")
    _require_dtype(x, "x")
    _require_dtype(weight, "weight")
    _require_cuda(x, "x")
    _require_cuda(weight, "weight")
    if weight.shape[0] != x.shape[1]:
        raise ValueError(f"weight length {weight.shape[0]} != x N {x.shape[1]}")
    _require_triton()
    M, N = x.shape
    y = torch.empty_like(x)
    grid = (M,)
    _rmsnorm_kernel[grid](
        x,
        weight,
        y,
        N,
        eps,
        x.stride(0),
        y.stride(0),
        BLOCK_N=triton.next_power_of_2(N),
    )
    return y


def triton_gqa_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """q [B,n_head,1,D]、k/v [B,n_kv,S,D] → y [B,n_head,1,D]。"""
    for name, t, ndim in (("q", q, 4), ("k", k, 4), ("v", v, 4)):
        if t.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}-D, got ndim={t.ndim}")
        _require_dtype(t, name)
        _require_cuda(t, name)
    if q.shape[2] != 1:
        raise ValueError(f"q T must be 1, got {q.shape[2]}")
    if k.shape != v.shape:
        raise ValueError(f"k/v shape mismatch: {tuple(k.shape)} vs {tuple(v.shape)}")
    B, n_head, _, D = q.shape
    n_kv, S = k.shape[1], k.shape[2]
    if q.shape[0] != k.shape[0] or k.shape[3] != D:
        raise ValueError(f"batch/head_dim mismatch: q {tuple(q.shape)}, k {tuple(k.shape)}")
    if n_head % n_kv != 0:
        raise ValueError(f"n_head {n_head} not divisible by n_kv_head {n_kv}")
    _require_triton()
    rep = n_head // n_kv
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    y = torch.empty((B, n_head, 1, D), device=q.device, dtype=q.dtype)
    grid = (B * n_kv,)
    _gqa_decode_kernel[grid](
        q,
        k,
        v,
        y,
        S,
        scale,
        n_kv,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        y.stride(0),
        y.stride(1),
        REP=rep,
        D=D,
        BLOCK_S=64,
        REP_PAD=16,
    )
    return y


# ---------------------------------------------------------------------------
# bench：尺寸表 + A/B 计时（cuda.Event，10 预热 50 迭代取中位数）+ 一致性两口径
# ---------------------------------------------------------------------------

_FULL_SIZES: dict[str, list[tuple[int, ...]]] = {
    "swiglu": [(1024, 1024, 512), (4096, 1024, 512), (16384, 1024, 512)],
    "rmsnorm": [(1024, 1024), (4096, 1024), (16384, 1024)],
    "gqa_decode": [(8, 16, 8, 64, 512), (8, 16, 8, 64, 2048), (8, 16, 8, 64, 8192)],
}
_QUICK_SIZES: dict[str, list[tuple[int, ...]]] = {
    "swiglu": [_FULL_SIZES["swiglu"][0]],
    "rmsnorm": [_FULL_SIZES["rmsnorm"][0]],
    "gqa_decode": [_FULL_SIZES["gqa_decode"][0]],
}

_N_WARMUP, N_ITER = 10, 50

# bf16 口径下 eager 参照链的中间舍入复合 ~1% 相对误差：swiglu 乘积幅度随 K 增大
# （N(0,σ²K)），×0.0625 把 |y| 压回 ≲1 让 atol 主导（Green 阶段实测校准：
# ×0.25 在 K=512 下 |y| 尾部 ~16，负 gate × 大 up 的舍入抵消使 rtol 失守）。
_SWIGLU_INPUT_SCALE = 0.0625


def _dtype_size(dt: torch.dtype) -> int:
    return 4 if dt == torch.float32 else 2


def bench_kernel(name: str, *args: Any, **kwargs: Any) -> Any:
    """按 kernel 名计时的统一入口；非法名 ValueError（design §4.3.3）。"""
    table: dict[str, Any] = {
        "swiglu": triton_swiglu,
        "rmsnorm": triton_rmsnorm,
        "gqa_decode": triton_gqa_decode,
    }
    if name not in table:
        raise ValueError(f"unknown kernel name: {name!r} (choose from {sorted(table)})")
    return table[name](*args, **kwargs)


def _cuda_inputs(name: str, size: tuple[int, ...], dt: torch.dtype) -> tuple[Any, ...]:
    torch.manual_seed(0)
    dev = torch.device("cuda")
    scale = _SWIGLU_INPUT_SCALE if name == "swiglu" else 1.0
    if name == "swiglu":
        M, N, K = size
        return (
            torch.randn(M, K, device=dev, dtype=dt) * scale,
            torch.randn(N, K, device=dev, dtype=dt) * scale,
            torch.randn(N, K, device=dev, dtype=dt) * scale,
        )
    if name == "rmsnorm":
        M, N = size
        return (
            torch.randn(M, N, device=dev, dtype=dt),
            torch.rand(N, device=dev, dtype=dt) + 0.5,
        )
    B, n_head, n_kv, D, S = size
    return (
        torch.randn(B, n_head, 1, D, device=dev, dtype=dt),
        torch.randn(B, n_kv, S, D, device=dev, dtype=dt),
        torch.randn(B, n_kv, S, D, device=dev, dtype=dt),
    )


def _eager_ref(name: str, *inputs: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if name == "swiglu":
        return eager_swiglu(inputs[0], inputs[1], inputs[2])
    if name == "rmsnorm":
        return eager_rmsnorm(inputs[0], inputs[1], eps=eps)
    return eager_gqa_decode(inputs[0], inputs[1], inputs[2])


def _triton_ref(name: str, *inputs: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if name == "swiglu":
        return triton_swiglu(inputs[0], inputs[1], inputs[2])
    if name == "rmsnorm":
        return triton_rmsnorm(inputs[0], inputs[1], eps=eps)
    return triton_gqa_decode(inputs[0], inputs[1], inputs[2])


def _intensity(name: str, size: tuple[int, ...], dt: torch.dtype) -> float:
    ds = _dtype_size(dt)
    if name == "swiglu":
        M, N, K = size
        return swiglu_flops(M, N, K) / swiglu_bytes(M, N, K, ds)
    if name == "rmsnorm":
        M, N = size
        return rmsnorm_flops(M, N) / rmsnorm_bytes(M, N, ds)
    B, n_head, n_kv, D, S = size
    rep = n_head // n_kv
    return gqa_decode_flops(B, n_kv, rep, S, D) / gqa_decode_bytes(B, n_kv, rep, S, D, ds)


def _time_ms(fn: Any, *args: Any) -> list[float]:
    for _ in range(_N_WARMUP):
        fn(*args)
    times = []
    for _ in range(N_ITER):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _bench_one(name: str, size: tuple[int, ...], dt: torch.dtype) -> dict[str, Any]:
    inputs = _cuda_inputs(name, size, dt)
    triton_ms = _time_ms(_triton_ref, name, *inputs)
    eager_ms = _time_ms(_eager_ref, name, *inputs)
    got, ref = _triton_ref(name, *inputs), _eager_ref(name, *inputs)
    atol, rtol = (ATOL_FP32, RTOL_FP32) if dt == torch.float32 else (ATOL_BF16, RTOL_BF16)
    t_mid = statistics.median(triton_ms)
    e_mid = statistics.median(eager_ms)
    return {
        "measurement": {
            "kernel": name,
            "size": dict(zip(_size_keys(name), size)),
            "dtype": str(dt).replace("torch.", ""),
            "triton_ms": t_mid,
            "eager_ms": e_mid,
            "speedup": e_mid / t_mid,
            "triton_ms_mean": statistics.mean(triton_ms),
            "eager_ms_mean": statistics.mean(eager_ms),
            "intensity_flops_per_byte": _intensity(name, size, dt),
        },
        "consistency": {
            "kernel": name,
            "size": dict(zip(_size_keys(name), size)),
            "dtype": str(dt).replace("torch.", ""),
            "allclose": bool(torch.allclose(got, ref, atol=atol, rtol=rtol)),
            "max_abs_err": float((got.float() - ref.float()).abs().max()),
            "atol": atol,
            "rtol": rtol,
        },
    }


def _size_keys(name: str) -> tuple[str, ...]:
    if name == "swiglu":
        return ("M", "N", "K")
    if name == "rmsnorm":
        return ("M", "N")
    return ("B", "n_head", "n_kv", "D", "S")


def run_bench(quick: bool = False, json_path: str | Path = DEFAULT_JSON) -> int:
    """全量 bench：三 kernel × 尺寸档 × fp32/bf16 两口径 → 原子落盘 JSON。

    任一一致性断言失败或测量异常 → 退出码 1 且不写 JSON（裁定 D4/D7）。
    """
    table = _QUICK_SIZES if quick else _FULL_SIZES
    measurements: list[dict[str, Any]] = []
    consistency: list[dict[str, Any]] = []
    for name in ("swiglu", "rmsnorm", "gqa_decode"):
        for size in table[name]:
            for dt in (torch.float32, torch.bfloat16):
                result = _bench_one(name, size, dt)
                measurements.append(result["measurement"])
                consistency.append(result["consistency"])
    if not all(c["allclose"] for c in consistency):
        print("consistency assertion failed; see rows above", file=sys.stderr)
        for c in consistency:
            print(
                f"  {c['kernel']} {c['dtype']} allclose={c['allclose']} "
                f"max_abs_err={c['max_abs_err']:.3e}",
                file=sys.stderr,
            )
        return 1
    payload = {
        "meta": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "triton": getattr(triton, "__version__", "n/a"),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quick": quick,
            "timing": f"cuda-event median of {N_ITER} after {_N_WARMUP} warmup",
        },
        "measurements": measurements,
        "consistency": consistency,
    }
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print_markdown_summary(measurements, consistency)
    return 0


def _print_markdown_summary(
    measurements: list[dict[str, Any]], consistency: list[dict[str, Any]]
) -> None:
    print("| kernel | size | dtype | eager_ms | triton_ms | speedup | intensity | allclose |")
    print("|---|---|---|---|---|---|---|---|")
    by_key = {(c["kernel"], json.dumps(c["size"]), c["dtype"]): c["allclose"] for c in consistency}
    for m in measurements:
        size = json.dumps(m["size"])
        print(
            f"| {m['kernel']} | {size} | {m['dtype']} | {m['eager_ms']:.4f} "
            f"| {m['triton_ms']:.4f} | {m['speedup']:.2f}x "
            f"| {m['intensity_flops_per_byte']:.1f} "
            f"| {by_key.get((m['kernel'], size, m['dtype']), '')} |"
        )


# ---------------------------------------------------------------------------
# CPU 冒烟（无 CUDA/无 triton：eager 参照 vs 手写公式，成功 exit 0 不写 JSON）
# ---------------------------------------------------------------------------


def _cpu_smoke() -> int:
    torch.manual_seed(0)
    print(
        f"triton available: {triton is not None}; " f"CUDA available: {torch.cuda.is_available()}"
    )
    x = torch.randn(4, 8)
    w_gate, w_up = torch.randn(6, 8), torch.randn(6, 8)
    ok = torch.allclose(
        eager_swiglu(x, w_gate, w_up),
        F.silu(x @ w_gate.T) * (x @ w_up.T),
        atol=ATOL_FP32,
        rtol=RTOL_FP32,
    )
    xn, wn = torch.randn(5, 16), torch.rand(16) + 0.5
    ok = ok and torch.allclose(
        eager_rmsnorm(xn, wn),
        xn * torch.rsqrt(xn.pow(2).mean(-1, keepdim=True) + 1e-5) * wn,
        atol=ATOL_FP32,
        rtol=RTOL_FP32,
    )
    B, n_head, n_kv, D, S = 2, 8, 4, 32, 64
    q = torch.randn(B, n_head, 1, D)
    k, v = torch.randn(B, n_kv, S, D), torch.randn(B, n_kv, S, D)
    rep = n_head // n_kv
    manual = (
        torch.softmax((q.view(B, n_kv, rep, D) @ k.transpose(-1, -2)) / math.sqrt(D), dim=-1) @ v
    )
    ok = ok and torch.allclose(
        eager_gqa_decode(q, k, v), manual.reshape(B, n_head, 1, D), atol=ATOL_FP32, rtol=RTOL_FP32
    )
    if not ok:
        print("CPU smoke assertion failed", file=sys.stderr)
        return 1
    print("smoke mode, no JSON written")
    return 0


def _selftest() -> int:
    """单档快速一致性检查（CUDA），不写 JSON；无 CUDA 走冒烟。"""
    if not torch.cuda.is_available():
        return _cpu_smoke()
    for name, size in (
        ("swiglu", _QUICK_SIZES["swiglu"][0]),
        ("rmsnorm", _QUICK_SIZES["rmsnorm"][0]),
        ("gqa_decode", _QUICK_SIZES["gqa_decode"][0]),
    ):
        result = _bench_one(name, size, torch.float32)
        c = result["consistency"]
        print(f"selftest {name}: allclose={c['allclose']} max_abs_err={c['max_abs_err']:.3e}")
        if not c["allclose"]:
            return 1
    return 0


# ---------------------------------------------------------------------------
# CLI（argparse SystemExit 原样传播：--help→0，用法错误→2）
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triton_kernels",
        description="Triton micro-benchmarks: fused SwiGLU / RMSNorm / GQA decode",
    )
    parser.add_argument("--quick", action="store_true", help="one size per kernel")
    parser.add_argument("--json", type=str, default=str(DEFAULT_JSON), help="output JSON path")
    parser.add_argument(
        "--selftest", action="store_true", help="single-size consistency check (no JSON)"
    )
    args = parser.parse_args(argv)
    if not torch.cuda.is_available() or triton is None:
        return _cpu_smoke()
    if args.selftest:
        return _selftest()
    try:
        return run_bench(quick=args.quick, json_path=args.json)
    except Exception as exc:  # 测量中断：不写部分 JSON（裁定 D4）
        print(f"measurement aborted: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
