"""显存/FLOPs 心算计算器（AR-v0.6.0-ai-infra-foundations / FR3）。

纯函数模块：不依赖 torch，公式与 ``docs/ai-infra/memory-flops-arithmetic.md``
互锁（同一公式两处实现，由 ``tests/test_infra_math.py`` 锚点断言钉死）。

口径约定（与 model/modern_gpt.py 逐项对齐，详见手册）：

- **参数量**：现代侧 Linear 全 bias=False（L335-338/911-913）；SwiGLU 中间维
  I = align128(8d/3)（L1282）；GELU I = 4d 不对齐（L1278）；QK-Norm 为
  RMSNorm(head_dim) 跨头共享 gamma → 每层 2·head_dim（L346-347）；RMSNorm
  每 norm 仅 gamma（d），LayerNorm 加 weight+bias（2d）；MoE = (E-1)·FFN +
  router d·E（L932）；tied lm_head 不重复计（L1624 张量共享 + L1905 id 去重）。
- **训练静态显存（precision 三口径）**：
  fp32 / amp（本仓 autocast 默认，参数即 fp32 主权重）= 4+4+8 = 16 B/param；
  bf16-pure（纯权重流派，Adam 状态仍 fp32）= 2+2+8 = 12 B/param。
  核心教学点：autocast 不省静态显存，省的是激活（trainer_base.build_amp_context）。
- **FLOPs**：训练 ≈ 6N + 12·L·T·d（N = 非 embedding 参数；tied lm_head 的
  vocab 投影 matmul 照样发生，include_embedding_matmul=True 时计入）。
- **KV**：2·L·n_kv·hd·dtype；SDPA 无原生 GQA 内核时的逐层瞬时物化与层数无关
  （MIGRATION §8.2：实测斜率 12,335 ≈ 单层瞬时 12,288 B/token）。

CLI::

    python -m utils.infra_math --n_layer 12 --n_embd 512 --n_head 8 \\
        --n_kv_head 2 --vocab_size 50257
"""

from __future__ import annotations

import argparse
import math
import sys

Gib = 1024**3

# 精度口径表：{参数, 梯度, Adam m+v} 字节/param（手册 §2.1 / design §4.1）
PRECISION_BYTES: dict[str, tuple[int, int, int]] = {
    "fp32": (4, 4, 8),
    "amp": (4, 4, 8),
    "bf16-pure": (2, 2, 8),
}

FFN_VARIANTS = ("swiglu", "gelu")
NORM_VARIANTS = ("rms", "layer")
POSITION_ENCODINGS = ("rope", "learned")


def _align_up(value: float, multiple_of: int) -> int:
    """向上对齐到 multiple_of 的整数倍（L1282：SwiGLU 中间维 LLaMA 式对齐）。"""
    return int(math.ceil(value / multiple_of)) * multiple_of


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} 必须为正整数，收到 {value}")


def _intermediate_size(n_embd: int, ffn_variant: str, override: int | None) -> int:
    if override is not None:
        _validate_positive("intermediate_size", override)
        return override
    if ffn_variant == "swiglu":
        return _align_up(8 * n_embd / 3, 128)
    return 4 * n_embd


def param_count(
    vocab_size: int,
    n_layer: int,
    n_embd: int,
    n_head: int,
    n_kv_head: int | None = None,
    ffn_variant: str = "swiglu",
    intermediate_size: int | None = None,
    num_experts: int = 1,
    tie_weights: bool = True,
    position_encoding: str = "rope",
    block_size: int = 1024,
    qk_norm: bool = False,
    use_bias: bool = False,
    norm_variant: str = "rms",
) -> int:
    """按 GPT 结构推总可训练参数量（口径见模块 docstring，锚点精确闭合）。

    锚点（EXPERIMENTS.md §6.1 E-A）：modern-full == 59,560,960、
    baseline legacy（gelu/MHA/bias/LayerNorm/learned PE）== 64,085,504、
    MoE(E=4) 增量 == 77,881,344、QK-Norm 增量 == 2·head_dim·L。
    """
    if ffn_variant not in FFN_VARIANTS:
        raise ValueError(f"ffn_variant 必须是 {FFN_VARIANTS} 之一，收到 {ffn_variant!r}")
    if norm_variant not in NORM_VARIANTS:
        raise ValueError(f"norm_variant 必须是 {NORM_VARIANTS} 之一，收到 {norm_variant!r}")
    if position_encoding not in POSITION_ENCODINGS:
        raise ValueError(
            f"position_encoding 必须是 {POSITION_ENCODINGS} 之一，收到 {position_encoding!r}"
        )
    _validate_positive("vocab_size", vocab_size)
    _validate_positive("n_layer", n_layer)
    _validate_positive("n_embd", n_embd)
    _validate_positive("n_head", n_head)
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd ({n_embd}) 必须能被 n_head ({n_head}) 整除")
    if num_experts < 1:
        raise ValueError(f"num_experts 必须 >= 1，收到 {num_experts}")

    kv = n_head if n_kv_head is None else n_kv_head
    if kv > n_head:
        raise ValueError(f"n_kv_head ({kv}) 不能超过 n_head ({n_head})")
    if n_head % kv != 0:
        raise ValueError(f"n_head ({n_head}) 必须能被 n_kv_head ({kv}) 整除")

    head_dim = n_embd // n_head
    kv_dim = kv * head_dim
    intermediate = _intermediate_size(n_embd, ffn_variant, intermediate_size)
    norm_params = n_embd * 2 if norm_variant == "layer" else n_embd

    # embedding（tied 时不重复计 lm_head；untied 恰好多一张 vocab×d）
    total = vocab_size * n_embd
    if not tie_weights:
        total += vocab_size * n_embd
    if position_encoding == "learned":
        _validate_positive("block_size", block_size)
        total += block_size * n_embd

    # 每层
    attn = n_embd * n_embd * 2 + n_embd * kv_dim * 2  # q/o + k/v
    ffn = n_embd * intermediate * 2 if ffn_variant == "gelu" else n_embd * intermediate * 3
    per_layer = attn + ffn + 2 * norm_params
    if use_bias:
        per_layer += n_embd * 2 + kv_dim * 2 + intermediate + n_embd
    if qk_norm:
        per_layer += 2 * head_dim
    total += n_layer * per_layer

    # MoE：每层 (E-1) 份额外专家 FFN + router d×E（num_experts=1 走稠密路径）
    if num_experts > 1:
        if ffn_variant != "swiglu":
            raise ValueError("MoE 仅支持 ffn_variant='swiglu'（config_models L121 同款约束）")
        total += n_layer * ((num_experts - 1) * ffn + n_embd * num_experts)

    # 最终 norm
    total += norm_params
    return total


def _activation_bytes_per_token_per_layer(
    n_embd: int,
    n_head: int,
    n_kv_head: int | None,
    ffn_variant: str,
    intermediate_size: int | None,
    num_experts: int,
    dtype_bytes: int,
    seq_len: int = 1024,
    materialize_scores: bool = False,
) -> float:
    """逐张量激活模型（design §4.1 表，fp32 口径给出、dtype 缩放）。

    覆盖：norm 输入/输出（2d）+ q/k/v 投影（d + 2·n_kv·hd）+ RoPE 旋转副本
    （d + n_kv·hd）+ SDPA 输出（d）+ o_proj/残差（2d）+ 第二个 norm（2d）
    + FFN 中间（SwiGLU 3I / GELU 8d，h+gelu(h) 两张量）+ FFN 输出/残差（2d）。

    attn scores 两口径（design §4.1）：fused SDPA（flash/mem-efficient 后端，
    本仓训练默认）不物化 [B,H,T,T]，不计；``materialize_scores=True`` 走
    math 后端口径，计 softmax 前后各一份 = 2·n_head·T B/token/层。
    """
    kv_dim = (n_head if n_kv_head is None else n_kv_head) * (n_embd // n_head)
    intermediate = _intermediate_size(n_embd, ffn_variant, intermediate_size)
    ffn_bytes = 3 * intermediate if ffn_variant == "swiglu" else 8 * n_embd
    per_element = (
        2 * n_embd  # ln_1 in/out
        + n_embd
        + 2 * kv_dim  # q/k/v proj
        + n_embd
        + kv_dim  # rope 副本
        + n_embd  # sdpa out
        + 2 * n_embd  # o_proj out + residual
        + 2 * n_embd  # ln_2 in/out
        + ffn_bytes
        + 2 * n_embd  # ffn out + residual
    )
    if num_experts > 1:
        per_element += num_experts  # router logits [B,T,E]（top-1 激活近似不变）
    if materialize_scores:
        per_element += 2 * n_head * seq_len  # scores [B,H,T,T] 前后各一，per token
    return per_element * dtype_bytes


def training_memory_breakdown(
    n_params: int,
    n_layer: int,
    n_embd: int,
    n_head: int,
    n_kv_head: int | None = None,
    seq_len: int = 1024,
    batch: int = 1,
    precision: str = "amp",
    grad_ckpt: bool = False,
    ffn_variant: str = "swiglu",
    num_experts: int = 1,
    dtype_bytes: int = 4,
    vocab_size: int = 50257,
    materialize_scores: bool = False,
) -> dict[str, float]:
    """训练显存五项分解（GB；口径表见模块 docstring / 手册 §2-§3）。

    返回键：params_gb / gradients_gb / optimizer_gb / activations_gb /
    kv_eval_gb / total_static_gb / total_gb。
    grad_ckpt=True 时激活 = 层边界（L·d·B·T）+ 单层重算峰值 + logits。
    materialize_scores=True 时激活额外计 attn scores [B,H,T,T]（softmax
    前后各一 = 2·n_head·T B/token/层）；默认 False 对应本仓 fused SDPA
    训练路径（不物化 scores，两口径见 design §4.1）。
    """
    if precision not in PRECISION_BYTES:
        raise ValueError(f"precision 必须是 {sorted(PRECISION_BYTES)} 之一，收到 {precision!r}")
    _validate_positive("n_params", n_params)
    _validate_positive("seq_len", seq_len)
    _validate_positive("batch", batch)

    p_bytes, g_bytes, opt_bytes = PRECISION_BYTES[precision]
    params_gb = n_params * p_bytes / Gib
    gradients_gb = n_params * g_bytes / Gib
    optimizer_gb = n_params * opt_bytes / Gib
    total_static_gb = params_gb + gradients_gb + optimizer_gb

    tokens = seq_len * batch
    per_layer = _activation_bytes_per_token_per_layer(
        n_embd,
        n_head,
        n_kv_head,
        ffn_variant,
        None,
        num_experts,
        dtype_bytes,
        seq_len=seq_len,
        materialize_scores=materialize_scores,
    )
    if grad_ckpt:
        # 只存层边界 hidden + 反向重算单层峰值；logits 仍需保留
        act_bytes = n_layer * n_embd * dtype_bytes * tokens + per_layer * tokens
    else:
        act_bytes = n_layer * per_layer * tokens
    act_bytes += vocab_size * tokens * dtype_bytes  # logits [B,T,V]
    activations_gb = act_bytes / Gib

    head_dim = n_embd // n_head
    kv = n_head if n_kv_head is None else n_kv_head
    kv_eval_gb = 2 * n_layer * kv * head_dim * tokens * dtype_bytes / Gib

    return {
        "params_gb": params_gb,
        "gradients_gb": gradients_gb,
        "optimizer_gb": optimizer_gb,
        "activations_gb": activations_gb,
        "kv_eval_gb": kv_eval_gb,
        "total_static_gb": total_static_gb,
        "total_gb": total_static_gb + activations_gb + kv_eval_gb,
    }


def inference_kv_bytes(
    n_layer: int,
    n_kv_head: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """推理 KV cache 字节：total = slope × T（GQA 用 n_kv_head，省的本质）。"""
    _validate_positive("n_layer", n_layer)
    _validate_positive("n_kv_head", n_kv_head)
    _validate_positive("head_dim", head_dim)
    _validate_positive("seq_len", seq_len)
    _validate_positive("batch", batch)
    _validate_positive("dtype_bytes", dtype_bytes)
    slope = 2 * n_layer * n_kv_head * head_dim * batch * dtype_bytes
    return {"total_bytes": float(slope * seq_len), "per_token_slope_bytes": float(slope)}


def gqa_kv_materialization_slope(
    n_head: int, head_dim: int, dtype_bytes: int = 4, tensors: int = 3
) -> float:
    """SDPA 无原生 GQA 内核时的逐 token 峰值瞬时物化斜率（与层数无关）。

    = tensors·n_head·head_dim·dtype_bytes；MIGRATION §8.2 锚点：
    (16, 64, 4, 3) == 12,288 B/token（24 层实测 12,335，层数无关——逐层串行
    执行 + 分配器复用使峰值 ≈ 单层瞬时）。
    """
    _validate_positive("n_head", n_head)
    _validate_positive("head_dim", head_dim)
    _validate_positive("dtype_bytes", dtype_bytes)
    if tensors < 1:
        raise ValueError(f"tensors 必须 >= 1，收到 {tensors}")
    return float(tensors * n_head * head_dim * dtype_bytes)


def _non_embedding_params(n_layer: int, n_embd: int, vocab_size: int, lm_head_tied: bool) -> int:
    """FLOPs 口径的 N：默认架构（MHA 注意力 4d² + SwiGLU + RMSNorm）非 embedding 参数。

    与 param_count(n_kv_head=None) 互锁（tests: 6N 项差值法）。
    """
    intermediate = _intermediate_size(n_embd, "swiglu", None)
    per_layer = 4 * n_embd * n_embd + 2 * n_embd + 3 * n_embd * intermediate
    total = n_layer * per_layer + n_embd  # 最终 norm
    if not lm_head_tied:
        total += n_embd * vocab_size
    return total


def flops_per_token(
    n_layer: int,
    n_embd: int,
    vocab_size: int,
    seq_len: int,
    backward: bool = True,
    include_embedding_matmul: bool = False,
    lm_head_tied: bool = True,
) -> float:
    """每 token FLOPs ≈ passes·2N + passes·4·L·T·d（+ lm_head matmul 项）。

    passes = 3（backward，前向 1 + 反向 2）或 1（纯前向）；attention 项
    backward 时即手册的 12·L·T·d。
    """
    _validate_positive("n_layer", n_layer)
    _validate_positive("n_embd", n_embd)
    _validate_positive("vocab_size", vocab_size)
    _validate_positive("seq_len", seq_len)
    passes = 3 if backward else 1
    n_params = _non_embedding_params(n_layer, n_embd, vocab_size, lm_head_tied)
    flops = 2 * n_params * passes + 4 * n_layer * seq_len * n_embd * passes
    if include_embedding_matmul:
        flops += 2 * n_embd * vocab_size * passes
    return float(flops)


def flops_per_step(
    n_layer: int,
    n_embd: int,
    vocab_size: int,
    seq_len: int,
    batch: int,
    backward: bool = True,
    include_embedding_matmul: bool = False,
    lm_head_tied: bool = True,
) -> float:
    """每优化步 FLOPs = flops_per_token × batch。"""
    _validate_positive("batch", batch)
    return float(
        batch
        * flops_per_token(
            n_layer,
            n_embd,
            vocab_size,
            seq_len,
            backward=backward,
            include_embedding_matmul=include_embedding_matmul,
            lm_head_tied=lm_head_tied,
        )
    )


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """算术强度 = FLOPs / 访存字节数（roofline 判据，手册 §6）。"""
    if bytes_moved <= 0:
        raise ValueError(f"bytes_moved 必须为正，收到 {bytes_moved}")
    return flops / bytes_moved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m utils.infra_math",
        description="显存/FLOPs 心算计算器（口径与 docs/ai-infra 手册互锁）",
    )
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_head", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--ffn_variant", choices=FFN_VARIANTS, default="swiglu")
    parser.add_argument("--num_experts", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    # precision 不用 argparse choices：非法值须以 ValueError 报出（ut-cli 契约），
    # 由 training_memory_breakdown 校验
    parser.add_argument("--precision", default="amp")
    parser.add_argument("--grad_ckpt", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：输出 Markdown 显存五项分解 + FLOPs 报告。"""
    args = _build_parser().parse_args(argv)

    n = param_count(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        ffn_variant=args.ffn_variant,
        num_experts=args.num_experts,
    )
    mem = training_memory_breakdown(
        n,
        args.n_layer,
        args.n_embd,
        args.n_head,
        n_kv_head=args.n_kv_head,
        seq_len=args.seq_len,
        batch=args.batch,
        precision=args.precision,
        grad_ckpt=args.grad_ckpt,
        ffn_variant=args.ffn_variant,
        num_experts=args.num_experts,
        vocab_size=args.vocab_size,
    )
    f_tok = flops_per_token(
        args.n_layer, args.n_embd, args.vocab_size, args.seq_len, include_embedding_matmul=True
    )

    print("# 显存/FLOPs 心算报告")
    print()
    print(f"- 参数量 N = {n:,}（{n / 1e6:.3f} M）")
    print(
        f"- 精度口径：{args.precision}（静态 " f"{sum(PRECISION_BYTES[args.precision])} B/param）"
    )
    print()
    print("| 项 | GB |")
    print("|----|----|")
    print(f"| 参数 | {mem['params_gb']:.3f} |")
    print(f"| 梯度 | {mem['gradients_gb']:.3f} |")
    print(f"| 优化器 m+v | {mem['optimizer_gb']:.3f} |")
    print(f"| **静态合计** | **{mem['total_static_gb']:.3f}** |")
    print(f"| 激活 | {mem['activations_gb']:.3f} |")
    print(f"| KV (eval) | {mem['kv_eval_gb']:.3f} |")
    print(f"| **总计** | **{mem['total_gb']:.3f}** |")
    print()
    print(f"- FLOPs/token ≈ {f_tok / 1e6:.1f} M（6N + 12·L·T·d，含 lm_head 项）")
    f_step = flops_per_step(
        args.n_layer,
        args.n_embd,
        args.vocab_size,
        args.seq_len,
        args.batch,
        include_embedding_matmul=True,
    )
    print(f"- FLOPs/step（batch={args.batch}）≈ {f_step / 1e9:.2f} G")
    print()
    print("口径与公式推导见 docs/ai-infra/memory-flops-arithmetic.md")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
