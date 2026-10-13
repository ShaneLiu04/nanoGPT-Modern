"""utils/infra_math.py 纯函数计算器的 TDD 回归锁（AR-v0.6.0-ai-infra-foundations / T003）。

Red 先行：``utils/infra_math.py`` 尚未实现，本模块在收集阶段即因 import 失败而红——
这是 TDD 的预期状态；实现合入（design.md §4.3.1 契约）后必须全绿。

锚点数据来源（可追溯到 docs/results/*.json）：
- docs/EXPERIMENTS.md §6.1 E-A 消融表（RTX 6000D 云端实测，18000 step）：
  baseline(legacy) 64.086M / modern-full 59.561M / moe(E=4) 总参数 137.442M
  （moe − modern-full 组间差 = 77.881M；qk-norm 59.562M）；
- docs/MIGRATION.md §8.2：SDPA 无原生 GQA 内核时的逐层瞬时物化
  12,288 B/token = 3 × 16头 × 64维 × 4B（24 层模型实测斜率 12,335，与层数无关）；
- docs/ai-infra/memory-flops-arithmetic.md §1.2/§2.2 的逐项精确账
  （modern-full = 25,731,584 + 12×2,819,072 + 512 = 59,560,960，分毫不差）。

断言约定：
- 整数锚点用 ``==`` 精确断言；
- 浮点（GB 换算）用 ``pytest.approx``，预期值一律写成 ``n_params * 16 / 1024**3``
  形式表达口径（不硬编码换算常数）；
- 异常断言用 ``pytest.raises``，并检查中文 message 含非法值本身。
"""

import pytest
from utils.infra_math import (
    arithmetic_intensity,
    flops_per_step,
    flops_per_token,
    gqa_kv_materialization_slope,
    inference_kv_bytes,
    main,
    param_count,
    training_memory_breakdown,
)

# 本仓标准 50M 模型（config/pretrain_cloud.yaml：modern-full）
VOCAB = 50257
L_STD, D_STD, H_STD, KV_STD = 12, 512, 8, 2
HEAD_DIM = D_STD // H_STD  # 64

# EXPERIMENTS.md §6.1 E-A 锚点（精确整数）
MODERN_FULL_PARAMS = 59_560_960  # 报告值 59.561M
BASELINE_LEGACY_PARAMS = 64_085_504  # 报告值 64.086M
MOE_DELTA_PARAMS = 77_881_344  # 137.442M − 59.561M

# training_memory_breakdown 的返回键集合（design §4.3.1，精确断言）
MEMORY_KEYS = {
    "params_gb",
    "gradients_gb",
    "optimizer_gb",
    "activations_gb",
    "kv_eval_gb",
    "total_static_gb",
    "total_gb",
}

GIB = 1024**3


# ---------------------------------------------------------------------------
# param_count
# ---------------------------------------------------------------------------


def test_param_count_modern_full_anchor():
    """ut-param-modern-full：modern-full 精确锚点 59,560,960（E-A 报告 59.561M）。

    逐项账：wte 25,731,584 + 12×(attn 655,360 + norms 1,024 + SwiGLU 2,162,688)
    + 最终 norm 512 = 59,560,960（tied lm_head 不重复计、RoPE 无参数表）。
    """
    got = param_count(
        vocab_size=VOCAB,
        n_layer=L_STD,
        n_embd=D_STD,
        n_head=H_STD,
        n_kv_head=KV_STD,
        ffn_variant="swiglu",
    )
    assert got == MODERN_FULL_PARAMS


def test_param_count_baseline_legacy_anchor():
    """ut-param-baseline：legacy 基线精确锚点 64,085,504（E-A 报告 64.086M）。

    逐项账：wte 25,731,584 + wpe 524,288 + 12×(MHA attn 1,048,576
    + GELU FFN 2,097,152 + LayerNorm 2,048 + bias 4,608) + ln_f 1,024。
    legacy 全 Linear 带 bias（共 55,296）是「现代架构去 bias」的教学对照组。
    """
    got = param_count(
        vocab_size=VOCAB,
        n_layer=L_STD,
        n_embd=D_STD,
        n_head=H_STD,
        n_kv_head=H_STD,
        ffn_variant="gelu",
        use_bias=True,
        norm_variant="layer",
        position_encoding="learned",
    )
    assert got == BASELINE_LEGACY_PARAMS


def test_param_count_moe_delta_anchor():
    """ut-param-moe-delta：E=4 时相对 modern-full 增量 = 77,881,344（精确）。

    MoE 口径 = (E−1)×FFN + router d×E = 3×2,162,688 + 2,048（每层 6,490,112 ×12）。
    """
    dense = param_count(
        vocab_size=VOCAB,
        n_layer=L_STD,
        n_embd=D_STD,
        n_head=H_STD,
        n_kv_head=KV_STD,
        ffn_variant="swiglu",
    )
    moe = param_count(
        vocab_size=VOCAB,
        n_layer=L_STD,
        n_embd=D_STD,
        n_head=H_STD,
        n_kv_head=KV_STD,
        ffn_variant="swiglu",
        num_experts=4,
    )
    assert dense == MODERN_FULL_PARAMS
    assert moe - dense == MOE_DELTA_PARAMS


def test_param_count_qk_norm_delta():
    """ut-param-qk-norm：qk_norm=True 增量 = 2·head_dim·L = 1,536（精确）。

    RMSNorm(head_dim) 跨头共享 gamma → 每层 2×64，×12 层（锚定 qk-norm 组 59.562M）。
    """
    without = param_count(
        vocab_size=VOCAB,
        n_layer=L_STD,
        n_embd=D_STD,
        n_head=H_STD,
        n_kv_head=KV_STD,
    )
    with_qk = param_count(
        vocab_size=VOCAB,
        n_layer=L_STD,
        n_embd=D_STD,
        n_head=H_STD,
        n_kv_head=KV_STD,
        qk_norm=True,
    )
    assert with_qk - without == 2 * HEAD_DIM * L_STD == 1536


def test_param_count_untied_lm_head():
    """ut-param-untied：tie_weights=False 恰好多一张独立 lm_head 权重 vocab×d。"""
    tied = param_count(
        vocab_size=VOCAB,
        n_layer=L_STD,
        n_embd=D_STD,
        n_head=H_STD,
        n_kv_head=KV_STD,
    )
    untied = param_count(
        vocab_size=VOCAB,
        n_layer=L_STD,
        n_embd=D_STD,
        n_head=H_STD,
        n_kv_head=KV_STD,
        tie_weights=False,
    )
    assert untied - tied == VOCAB * D_STD


def test_param_count_invalid_ffn_variant():
    """ut-param-invalid：ffn_variant 非法 → ValueError，中文 message 含非法值。"""
    with pytest.raises(ValueError) as excinfo:
        param_count(VOCAB, L_STD, D_STD, H_STD, KV_STD, ffn_variant="x")
    assert "x" in str(excinfo.value)


def test_param_count_invalid_norm_variant():
    """ut-param-invalid：norm_variant 非法 → ValueError，中文 message 含非法值。"""
    with pytest.raises(ValueError) as excinfo:
        param_count(VOCAB, L_STD, D_STD, H_STD, KV_STD, norm_variant="x")
    assert "x" in str(excinfo.value)


def test_param_count_invalid_kv_head_relation():
    """ut-param-invalid：n_kv_head > n_head 或不整除 n_head → ValueError。"""
    with pytest.raises(ValueError):
        param_count(VOCAB, L_STD, D_STD, H_STD, n_kv_head=H_STD * 2)
    with pytest.raises(ValueError):
        param_count(VOCAB, L_STD, D_STD, H_STD, n_kv_head=3)  # 8 % 3 != 0


def test_param_count_invalid_dimensions():
    """ut-param-invalid：零/负维度 → ValueError。"""
    with pytest.raises(ValueError):
        param_count(VOCAB, n_layer=0, n_embd=D_STD, n_head=H_STD)
    with pytest.raises(ValueError):
        param_count(VOCAB, L_STD, n_embd=0, n_head=H_STD)
    with pytest.raises(ValueError):
        param_count(VOCAB, L_STD, D_STD, n_head=-8)
    with pytest.raises(ValueError):
        param_count(vocab_size=-1, n_layer=L_STD, n_embd=D_STD, n_head=H_STD)


# ---------------------------------------------------------------------------
# training_memory_breakdown
# ---------------------------------------------------------------------------


def test_training_memory_amp_static_16_bytes_per_param():
    """ut-mem-amp-16b：AMP autocast 口径 = 4+4+8 = 16 B/param。

    autocast 不省静态显存（参数即 fp32 主权重），只省激活——口径表锁定。
    顺带精确断言返回 dict 键集合（design §6.2 接口契约）。
    """
    m = training_memory_breakdown(
        MODERN_FULL_PARAMS,
        L_STD,
        D_STD,
        H_STD,
        n_kv_head=KV_STD,
        precision="amp",
    )
    assert set(m) == MEMORY_KEYS
    assert m["params_gb"] == pytest.approx(MODERN_FULL_PARAMS * 4 / GIB)
    assert m["gradients_gb"] == pytest.approx(MODERN_FULL_PARAMS * 4 / GIB)
    assert m["optimizer_gb"] == pytest.approx(MODERN_FULL_PARAMS * 8 / GIB)
    assert m["total_static_gb"] == pytest.approx(MODERN_FULL_PARAMS * 16 / GIB)
    assert m["total_gb"] >= m["total_static_gb"]


def test_training_memory_fp32_row_items():
    """ut-mem-fp32-16b：fp32 纯精度行项 4+4+8，静态合计同为 16 B/param。

    与 amp 同账不同语义（fp32 无 bf16 激活折减，手册 §2.1 注明）。
    """
    m = training_memory_breakdown(
        100_000_000,
        L_STD,
        D_STD,
        H_STD,
        n_kv_head=KV_STD,
        precision="fp32",
    )
    assert m["params_gb"] == pytest.approx(100_000_000 * 4 / GIB)
    assert m["gradients_gb"] == pytest.approx(100_000_000 * 4 / GIB)
    assert m["optimizer_gb"] == pytest.approx(100_000_000 * 8 / GIB)
    assert m["total_static_gb"] == pytest.approx(
        m["params_gb"] + m["gradients_gb"] + m["optimizer_gb"]
    )
    assert m["total_static_gb"] == pytest.approx(100_000_000 * 16 / GIB)


def test_training_memory_bf16_pure_static_12_bytes_per_param():
    """ut-mem-bf16-pure-12b：bf16 纯权重口径 = 2+2+8 = 12 B/param（状态仍 fp32）。"""
    m = training_memory_breakdown(
        MODERN_FULL_PARAMS,
        L_STD,
        D_STD,
        H_STD,
        n_kv_head=KV_STD,
        precision="bf16-pure",
    )
    assert m["params_gb"] == pytest.approx(MODERN_FULL_PARAMS * 2 / GIB)
    assert m["gradients_gb"] == pytest.approx(MODERN_FULL_PARAMS * 2 / GIB)
    assert m["optimizer_gb"] == pytest.approx(MODERN_FULL_PARAMS * 8 / GIB)
    assert m["total_static_gb"] == pytest.approx(MODERN_FULL_PARAMS * 12 / GIB)


def test_training_memory_moe_static_delta_anchor():
    """ut-mem-moe-anchor：+77,881,344 参数 → 静态增量 = 77,881,344×16 B ≈ 1.16 GiB。

    AR-v0.4.0 记录的「moe 组优化器态 +1.25 GB（十进制）」实测锚点，AMP 口径闭环。
    """
    base = training_memory_breakdown(
        MODERN_FULL_PARAMS,
        L_STD,
        D_STD,
        H_STD,
        n_kv_head=KV_STD,
        precision="amp",
    )
    moe = training_memory_breakdown(
        MODERN_FULL_PARAMS + MOE_DELTA_PARAMS,
        L_STD,
        D_STD,
        H_STD,
        n_kv_head=KV_STD,
        precision="amp",
    )
    assert moe["total_static_gb"] - base["total_static_gb"] == pytest.approx(
        MOE_DELTA_PARAMS * 16 / GIB
    )


def test_training_memory_grad_ckpt_reduces_activations():
    """ut-mem-grad-ckpt：grad_ckpt=True 激活严格小于全量保存（L>1）。"""
    common = dict(
        n_layer=L_STD,
        n_embd=D_STD,
        n_head=H_STD,
        n_kv_head=KV_STD,
        seq_len=1024,
        batch=32,
        precision="amp",
    )
    full = training_memory_breakdown(MODERN_FULL_PARAMS, grad_ckpt=False, **common)
    ckpt = training_memory_breakdown(MODERN_FULL_PARAMS, grad_ckpt=True, **common)
    assert full["activations_gb"] > ckpt["activations_gb"] > 0


def test_training_memory_invalid_precision():
    """ut-mem-invalid：precision 非法 → ValueError，中文 message 含非法值。"""
    with pytest.raises(ValueError) as excinfo:
        training_memory_breakdown(
            MODERN_FULL_PARAMS,
            L_STD,
            D_STD,
            H_STD,
            n_kv_head=KV_STD,
            precision="x",
        )
    assert "x" in str(excinfo.value)


# ---------------------------------------------------------------------------
# inference_kv_bytes / gqa_kv_materialization_slope
# ---------------------------------------------------------------------------


def test_inference_kv_bytes_slope_and_total():
    """ut-kv-slope：slope = 2·L·n_kv·hd·B·dtype，total = slope×T（手算逐值对照）。

    50M 模型 bf16 KV：2×12×2×64×1×2 = 6,144 B/token，T=1024 → 6,291,456 B。
    """
    kv = inference_kv_bytes(
        n_layer=L_STD,
        n_kv_head=KV_STD,
        head_dim=HEAD_DIM,
        seq_len=1024,
        batch=1,
        dtype_bytes=2,
    )
    assert set(kv) == {"total_bytes", "per_token_slope_bytes"}
    assert kv["per_token_slope_bytes"] == 2 * L_STD * KV_STD * HEAD_DIM * 1 * 2 == 6144
    assert kv["total_bytes"] == 6144 * 1024


def test_inference_kv_bytes_invalid_seq_len():
    """ut-kv-invalid：seq_len 非正 → ValueError。"""
    with pytest.raises(ValueError):
        inference_kv_bytes(L_STD, KV_STD, HEAD_DIM, seq_len=0)
    with pytest.raises(ValueError):
        inference_kv_bytes(L_STD, KV_STD, HEAD_DIM, seq_len=-1)


def test_gqa_kv_materialization_slope_anchor():
    """ut-kv-materialization：SDPA GQA 扩展物化斜率 = tensors·n_head·hd·dtype。

    MIGRATION §8.2 锚点：3×16×64×4 = 12,288 B/token（24 层模型实测 12,335，
    与层数无关——逐层串行执行 + 分配器复用使峰值 ≈ 单层瞬时；签名无层数参数
    本身就是该语义的接口表达）。
    """
    assert gqa_kv_materialization_slope(16, 64, dtype_bytes=4, tensors=3) == 12288


def test_gqa_kv_materialization_invalid_args():
    """ut-gqa-invalid：n_head/head_dim/dtype_bytes 非正或 tensors<1 → ValueError。"""
    with pytest.raises(ValueError):
        gqa_kv_materialization_slope(n_head=0, head_dim=64)
    with pytest.raises(ValueError):
        gqa_kv_materialization_slope(n_head=16, head_dim=-1)
    with pytest.raises(ValueError):
        gqa_kv_materialization_slope(n_head=16, head_dim=64, dtype_bytes=0)
    with pytest.raises(ValueError):
        gqa_kv_materialization_slope(n_head=16, head_dim=64, tensors=0)


# ---------------------------------------------------------------------------
# flops_per_token / flops_per_step
# ---------------------------------------------------------------------------


def test_flops_6n_term_interlocks_with_param_count():
    """ut-flops-6nd：6N 项与 param_count 公式互锁（不同 n_layer 差值法）。

    flops(L=13) − flops(L=12) = 6·(单层非 embedding 参数) + 12·(13−12)·T·d：
    embedding/lm_head/最终 norm 等与 L 无关的项在差值中全部相消。
    单层参数用 param_count 差值求得——两公式在 N 的口径上互锁
    （默认架构：swiglu/RMSNorm/RoPE/bias=False；n_kv_head=None 时 k/v 投影为
    d×d，与 n_head 取值无关）。
    """
    seq_len = 1024
    single_layer = param_count(VOCAB, 13, D_STD, H_STD) - param_count(VOCAB, 12, D_STD, H_STD)
    expected_delta = 6 * single_layer + 12 * (13 - 12) * seq_len * D_STD
    got_delta = flops_per_token(13, D_STD, VOCAB, seq_len) - flops_per_token(
        12, D_STD, VOCAB, seq_len
    )
    assert got_delta == pytest.approx(expected_delta)


def test_flops_attention_term_exact():
    """ut-flops-attn-term：attention 项 = 12·L·T·d（两配置差值，精确断言）。

    6N 与 lm_head 项均与 T 无关，差值只留 attention 项的 T 增量。
    """
    f_1024 = flops_per_token(L_STD, D_STD, VOCAB, 1024)
    f_512 = flops_per_token(L_STD, D_STD, VOCAB, 512)
    assert f_1024 - f_512 == 12 * L_STD * (1024 - 512) * D_STD


def test_flops_per_step_scales_with_batch():
    """ut-flops-step：flops_per_step = flops_per_token × batch（同参对照）。"""
    per_token = flops_per_token(L_STD, D_STD, VOCAB, 1024)
    assert flops_per_step(L_STD, D_STD, VOCAB, 1024, 32) == pytest.approx(per_token * 32)
    assert flops_per_step(L_STD, D_STD, VOCAB, 1024, 1) == pytest.approx(per_token)


def test_flops_per_step_invalid_batch():
    """ut-flops-step：batch 非正 → ValueError。"""
    with pytest.raises(ValueError):
        flops_per_step(L_STD, D_STD, VOCAB, 1024, 0)
    with pytest.raises(ValueError):
        flops_per_step(L_STD, D_STD, VOCAB, 1024, -1)


def test_flops_lm_head_matmul_term():
    """ut-flops-lmhead：include_embedding_matmul 差值随 backward 系数缩放。

    tied lm_head 权重即 embedding（不计入 N），但 vocab 投影 matmul 照样发生：
    前向 2·d·vocab/次；backward=True 时按「前向 1 + 反向 2 = 3 次通过」乘 3
    → 差值 = 6·d·vocab（此处按 backward 系数口径断言）。
    """
    with_matmul = flops_per_token(L_STD, D_STD, VOCAB, 1024, include_embedding_matmul=True)
    without = flops_per_token(L_STD, D_STD, VOCAB, 1024, include_embedding_matmul=False)
    assert with_matmul - without == pytest.approx(6 * D_STD * VOCAB)

    fwd_with = flops_per_token(
        L_STD, D_STD, VOCAB, 1024, backward=False, include_embedding_matmul=True
    )
    fwd_without = flops_per_token(
        L_STD, D_STD, VOCAB, 1024, backward=False, include_embedding_matmul=False
    )
    assert fwd_with - fwd_without == pytest.approx(2 * D_STD * VOCAB)


def test_flops_untied_lm_head_counts_in_n():
    """ut-flops-lmhead（tied 分支）：lm_head_tied=False 时 N 含独立 lm_head 权重。

    与 ut-param-untied 互锁：untied 比 tied 恰多 d·vocab 参数 → 6N 项差值
    = 6·d·vocab（include_embedding_matmul=False 下 lm_head matmul 项不另计）。
    """
    tied = flops_per_token(L_STD, D_STD, VOCAB, 1024, lm_head_tied=True)
    untied = flops_per_token(L_STD, D_STD, VOCAB, 1024, lm_head_tied=False)
    assert untied - tied == pytest.approx(6 * D_STD * VOCAB)


def test_flops_invalid_seq_len():
    """ut-flops-invalid：seq_len ≤ 0 → ValueError。"""
    with pytest.raises(ValueError):
        flops_per_token(L_STD, D_STD, VOCAB, 0)
    with pytest.raises(ValueError):
        flops_per_token(L_STD, D_STD, VOCAB, -1)


# ---------------------------------------------------------------------------
# arithmetic_intensity
# ---------------------------------------------------------------------------


def test_arithmetic_intensity_gemm_exact():
    """ut-intensity：GEMM [M,N,K] 强度 = 2MNK / 2(MN+NK+MK)（精确断言）。

    立方 GEMM（M=N=K）强度 = K/3——大 GEMM 强度 ≈ K 的手册 §6.1 判据。
    """
    m = n = k = 1024
    flops = 2 * m * n * k
    bytes_moved = 2 * (m * n + n * k + m * k)
    assert arithmetic_intensity(flops, bytes_moved) == pytest.approx(flops / bytes_moved)
    assert arithmetic_intensity(flops, bytes_moved) == pytest.approx(1024 / 3)


def test_arithmetic_intensity_invalid_bytes():
    """ut-intensity：bytes_moved ≤ 0 → ValueError。"""
    with pytest.raises(ValueError):
        arithmetic_intensity(2_147_483_648, 0)
    with pytest.raises(ValueError):
        arithmetic_intensity(2_147_483_648, -1)


# ---------------------------------------------------------------------------
# CLI（python -m utils.infra_math）
# ---------------------------------------------------------------------------


def test_cli_main_renders_markdown(capsys):
    """ut-cli：标准 modern-full 参数 → 返回 0，stdout 为 Markdown 报告。"""
    rc = main(
        [
            "--n_layer",
            "12",
            "--n_embd",
            "512",
            "--n_head",
            "8",
            "--n_kv_head",
            "2",
            "--vocab_size",
            "50257",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "GB" in out or "参数" in out


def test_cli_invalid_precision_raises():
    """ut-cli：非法 --precision → ValueError（不静默兜底）。"""
    with pytest.raises(ValueError):
        main(
            [
                "--n_layer",
                "12",
                "--n_embd",
                "512",
                "--n_head",
                "8",
                "--n_kv_head",
                "2",
                "--vocab_size",
                "50257",
                "--precision",
                "x",
            ]
        )


def test_cli_help_exits_zero(capsys):
    """ut-cli：--help 打印用法并 SystemExit(0)（FR3 验收直测）。"""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "--n_layer" in capsys.readouterr().out
