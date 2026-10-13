"""roofline 本机微实验（AR-v0.6.0-ai-infra-foundations / FR4，T004）。

三段式（design §4.2.2）：

1. **GEMM 尺寸扫描**：64 -> 8192 跨越 L2 4 MiB 工作集（64^3*4B = 1 MiB 在
   L2 内，8192^3*4B = 2 GiB 走显存），测 GFLOPs/s 并观察 cache-bound 区间；
2. **GEMV 对照**：矩阵 x 向量，强度约 1，验证 memory-bound 方向；
3. **decode 步反推**：tiny ModernGPT 单 token 前向计时，有效带宽 = 权重字节
   /步时，锚定手册「decode 必然 memory-bound（强度约 0.5 远低于 ridge）」主张。

GPU_SPECS 不猜分母原则：规格值只来自下表子串匹配（nvidia-smi 卡名 + 官方
规格表），未知卡 ``spec_bandwidth_gbs = None``，利用率只在规格已知时给出
——测量（effective_gbps）与规格（spec_bandwidth_gbs）是两个口径，分列。

产物：``docs/results/roofline.json``（schema 见 summarize）。
本卡规格（nvidia-smi 实测 16384 MiB）：Quadro RTX 5000 桌面版，16 GiB
GDDR6 / 448 GB/s（十进制）/ L2 4 MiB / sm75 / FP32 约 11 TFLOPS。
注：docs/MIGRATION.md §8 曾误载「8GB」，已随本任务修正（T004 门控 W5）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import torch

# 以文件方式运行时把仓库根加入 sys.path（tools/profiling/profile_inference.py L24 先例）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 已知卡规格表：键为卡名子串（get_device_name(0) 含它即命中），值为官方规格。
GPU_SPECS: dict[str, dict[str, float]] = {
    "Quadro RTX 5000": {"bandwidth_gbs": 448.0, "l2_cache_mib": 4.0, "fp32_tflops": 11.0},
}

# 完整横轴：1 MiB（L2 内）-> 4 MiB（L2 边界）-> 2 GiB（显存段）三段结构
SIZES_FULL: list[int] = [
    64,
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
]
# --quick：⊆ FULL 且 ≥10 点（FR4 验收：quick 产物仍覆盖全横轴）
SIZES_QUICK: list[int] = [64, 128, 256, 512, 1024, 1536, 2048, 4096, 6144, 8192]

GEMV_SHAPES: list[tuple[int, int]] = [(2048, 2048), (4096, 4096), (8192, 8192)]
DECODE_N_LAYERS = 4
DECODE_N_EMBD = 128
DECODE_N_HEAD = 4


def gemm_flops(m: int, n: int, k: int) -> int:
    """GEMM [M,N,K] FLOPs = 2*M*N*K（与 utils.infra_math 强度公式共用口径）。"""
    return 2 * m * n * k


def gemm_working_set_bytes(m: int, n: int, k: int, dtype_bytes: int = 4) -> int:
    """GEMM 工作集 = (输出 + A + B) 元素数 x dtype 字节。"""
    return (m * n + n * k + m * k) * dtype_bytes


def effective_bandwidth_gbs(num_bytes: float, seconds: float) -> float:
    """有效带宽（十进制 GB/s）= bytes / seconds / 1e9。"""
    if seconds <= 0:
        raise ValueError(f"seconds 必须为正，收到 {seconds}")
    return num_bytes / seconds / 1e9


def _time_ms(fn, device: str, warmup: int = 3, iters: int = 10) -> float:
    """计时辅助：CUDA 用事件，CPU 用 perf_counter；返回均值毫秒。"""
    for _ in range(warmup):
        fn()
    times_ms: list[float] = []
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times_ms.append((time.perf_counter() - t0) * 1e3)
    return sum(times_ms) / len(times_ms)


def gemm_sweep(device: str, sizes: list[int], iters: int = 10) -> list[dict]:
    """段1：GEMM 尺寸扫描（fp32 方阵），每尺寸返回耗时/算力/工作集。"""
    if not sizes:
        raise ValueError("sizes 不能为空（快速失败，不静默）")
    dev = torch.device(device)
    rows: list[dict] = []
    for s in sizes:
        a = torch.randn(s, s, device=dev)
        b = torch.randn(s, s, device=dev)
        ms = _time_ms(partial(torch.matmul, a, b), device, iters=iters)
        rows.append(
            {
                "size": s,
                "ms": round(ms, 4),
                "gflops_per_s": round(gemm_flops(s, s, s) / (ms * 1e6), 2),
                "working_set_bytes": gemm_working_set_bytes(s, s, s),
            }
        )
    return rows


def gemv_bandwidth(device: str, shapes: list[tuple[int, int]], iters: int = 10) -> list[dict]:
    """段2：GEMV [N,K]x[K] 对照（强度约 1，memory-bound 方向验证）。"""
    if not shapes:
        raise ValueError("shapes 不能为空（快速失败，不静默）")
    dev = torch.device(device)
    rows: list[dict] = []
    for n, k in shapes:
        matrix = torch.randn(n, k, device=dev)
        vec = torch.randn(k, device=dev)
        ms = _time_ms(partial(torch.matmul, matrix, vec), device, iters=iters)
        bytes_moved = (n * k + k + n) * 4
        rows.append(
            {
                "n": n,
                "k": k,
                "ms": round(ms, 4),
                "bytes_moved": bytes_moved,
                "gbps": round(effective_bandwidth_gbs(bytes_moved, ms / 1e3), 2),
            }
        )
    return rows


def decode_step_bandwidth(device: str, n_layer: int = DECODE_N_LAYERS) -> dict:
    """段3：tiny ModernGPT 单 token 前向计时，反推 decode 有效带宽。

    batch=1 时每个权重只参与一次 MAC：强度 = 2N/权重字节 约 0.5（远低于
    ridge），bound 恒为 memory（方向判定；小模型绝对利用率另见手册 §6.1 注）。
    """
    from model.modern_gpt import ModernGPT, ModernGPTConfig

    dev = torch.device(device)
    cfg = ModernGPTConfig(
        vocab_size=50257,
        block_size=64,
        n_layer=n_layer,
        n_head=DECODE_N_HEAD,
        n_embd=DECODE_N_EMBD,
        dropout=0.0,
    )
    model = ModernGPT(cfg).to(dev).eval()
    n_params = model.get_num_params(non_embedding=False)
    weight_bytes = n_params * 4
    idx = torch.zeros(1, 1, dtype=torch.long, device=dev)
    with torch.no_grad():
        ms = _time_ms(partial(model, idx), device, warmup=5, iters=50)
    return {
        "n_params": int(n_params),
        "weight_bytes": weight_bytes,
        "step_ms": round(ms, 4),
        "effective_gbps": round(effective_bandwidth_gbs(weight_bytes, ms / 1e3), 2),
        "bound": "memory",
    }


def summarize(
    gemm: list[dict],
    gemv: list[dict],
    decode: dict | None,
    gpu_name: str | None,
    mode: str,
) -> dict:
    """组装 roofline.json（version/mode/gpu_name/spec_bandwidth_gbs/segments）。

    spec_bandwidth_gbs 只在卡名子串命中 GPU_SPECS 时给出（不猜分母）。
    """
    spec_bandwidth_gbs: float | None = None
    if gpu_name:
        for key, val in GPU_SPECS.items():
            if key in gpu_name:
                spec_bandwidth_gbs = val["bandwidth_gbs"]
                break
    return {
        "version": 1,
        "mode": mode,
        "gpu_name": gpu_name,
        "spec_bandwidth_gbs": spec_bandwidth_gbs,
        "segments": {"gemm": gemm, "gemv": gemv, "decode": decode},
    }


def _print_report(payload: dict) -> None:
    """stdout Markdown 摘要（表格/JSON 替代绘图，design D5：gitee PNG 预算）。"""
    print("# roofline 微实验报告")
    print()
    print(f"- GPU：{payload['gpu_name']}（mode={payload['mode']}）")
    spec = payload["spec_bandwidth_gbs"]
    spec_text = spec if spec is not None else "未知（不猜分母）"
    print(f"- 规格带宽：{spec_text} GB/s")
    print()
    print("## 段1 GEMM 扫描（fp32 方阵）")
    print()
    print("| size | ms | GFLOPs/s | 工作集 |")
    print("|------|----|----------|--------|")
    for r in payload["segments"]["gemm"]:
        ws_mib = r["working_set_bytes"] / 2**20
        print(f"| {r['size']} | {r['ms']} | {r['gflops_per_s']} | {ws_mib:.1f} MiB |")
    gflops = [r["gflops_per_s"] for r in payload["segments"]["gemm"]]
    if gflops:
        peak = max(gflops)
        small = payload["segments"]["gemm"][0]["gflops_per_s"]
        ratio = peak / small if small else float("nan")
        if ratio > 1.5:
            verdict = f"小->大 GFLOPs/s 提升 {ratio:.1f}x：cache-bound 区间可见"
        else:
            verdict = "小尺寸与大尺寸差距不明显（受驱动/频率/计时粒度影响）"
        print()
        print(f"- 峰值 {peak:.1f} GFLOPs/s；{verdict}")
    print()
    print("## 段2 GEMV 对照")
    print()
    print("| NxK | ms | 有效 GB/s |" + (" 利用率 vs 规格 |" if spec else ""))
    print("|------|----|-----------|" + ("------|" if spec else ""))
    for r in payload["segments"]["gemv"]:
        util = f" {r['gbps'] / spec * 100:.1f}% |" if spec else ""
        print(f"| {r['n']}x{r['k']} | {r['ms']} | {r['gbps']} |{util}")
    dec = payload["segments"]["decode"]
    if dec:
        print()
        print("## 段3 decode 步反推")
        print()
        n_params = dec["n_params"]
        w_mib = dec["weight_bytes"] / 2**20
        print(f"- tiny 模型 {n_params:,} 参数（fp32 权重 {w_mib:.1f} MiB）")
        util = ""
        if spec:
            util = f"（规格利用率 {dec['effective_gbps'] / spec * 100:.1f}%）"
        print(f"- 单步 {dec['step_ms']} ms -> 有效带宽 {dec['effective_gbps']} GB/s{util}")
        print(f"- bound = {dec['bound']}（batch=1 强度约 0.5，远低于 ridge）")


def _parse_sizes(raw: str) -> list[int]:
    sizes = [int(s) for s in raw.split(",") if s.strip()]
    if not sizes:
        raise ValueError(f"--sizes 解析为空：{raw!r}")
    return sizes


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python tools/infra/run_roofline.py",
        description="roofline 微实验（三段式：GEMM 扫描/GEMV 对照/decode 反推）",
    )
    parser.add_argument("--quick", action="store_true", help="截短尺寸表（>=10 点）")
    parser.add_argument("--skip_decode", action="store_true", help="跳过段3")
    parser.add_argument("--sizes", type=_parse_sizes, default=None, help="GEMM 尺寸覆盖")
    parser.add_argument("--out", default="docs/results/roofline.json", help="结果 JSON 路径")
    parser.add_argument("--iters", type=int, default=10, help="每尺寸计时迭代数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = get_args(argv)
    out_path = Path(args.out)

    if not torch.cuda.is_available():
        print("[roofline] CUDA 不可用：走 CPU 冒烟（仅验证逻辑，数值无意义）")
        try:
            gemm = gemm_sweep("cpu", [64, 128, 256], iters=2)
        except Exception as exc:  # 脚本级兜底：退出码语义化
            print(f"[roofline] CPU 冒烟失败：{exc}")
            return 1
        payload = summarize(gemm, [], None, "cpu", "cpu-smoke")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[roofline] cpu-smoke 完成 -> {out_path}")
        return 0

    sizes = args.sizes if args.sizes is not None else (SIZES_QUICK if args.quick else SIZES_FULL)
    try:
        gemm = gemm_sweep("cuda", sizes, iters=args.iters)
        gemv = gemv_bandwidth("cuda", GEMV_SHAPES, iters=args.iters)
        decode = None if args.skip_decode else decode_step_bandwidth("cuda")
    except Exception as exc:  # 测量中断：不写部分 JSON，退出 1
        print(f"[roofline] 测量失败（不写部分 JSON）：{exc}")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    mode = "quick" if args.quick else "full"
    payload = summarize(gemm, gemv, decode, gpu_name, mode)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(payload)
    print()
    print(f"[roofline] 完成 -> {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
