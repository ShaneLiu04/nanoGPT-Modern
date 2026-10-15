# AI Infra 学习路径（Learning Track）

> 面向想深入「大模型系统工程」的读者：以本仓 nanoGPT-Modern 的**真实代码、真实实验数据**为锚点，把 AI infra 的核心主题学透。
> 原则：**每个公式都能在本仓找到交叉验证**——不是博客式转载，而是「推导 → 工具复算 → 实测对照」三段式。

## 如何使用本路径

1. 先读 [显存/算力心算手册](./memory-flops-arithmetic.md)（主题 1，本期已交付），建立量化直觉；
2. 用计算器复算手册每个数字：`python -m utils.infra_math --help`；
3. 本机跑 roofline 微实验：`python tools/infra/run_roofline.py --quick`；
4. 按下方主题地图逐主题深入，每主题先看「本仓锚点」再看延伸阅读；
5. 完整 GPU 实验按 [RUNNING_EXPERIMENTS.md](../RUNNING_EXPERIMENTS.md) F 组 SOP 执行（需高算力 GPU）。

## 主题地图（10 主题 → 6 个 AR）

| # | 主题 | 交付 AR | 状态 | 本仓锚点 | 延伸阅读 |
|---|------|---------|------|----------|----------|
| 1 | **显存与算力心算**：参数/梯度/优化器/激活/KV 五项分解、6ND FLOPs、roofline 判据 | AR-v0.6.0 | **已交付** | [心算手册](./memory-flops-arithmetic.md) · [计算器 utils/infra_math.py](../../utils/infra_math.py) · [roofline 微实验](../../tools/infra/run_roofline.py) · [EXPERIMENTS E-A 显存表](../EXPERIMENTS.md) · [MIGRATION §8.2 KV 斜率](../MIGRATION.md) · [ADR-004 KV 量化](../adr/adr-004-kv-cache-quantization.md) | Transformer Math 101（Kipply）· Megatron-LM 论文（Shoeybi et al., 2019）· Roofline 模型（Williams et al., 2009）· Adam（Kingma & Ba, 2014） |
| 2 | **Attention kernel 深潜**：SDPA/GQA 内存布局、flash 注意力分块、scores [B,H,T,T] 物化代价 | AR-v0.7.0 / **AR-v0.8.5 深化** | **已交付** | [attention 手册](./attention-kernels.md) · [ADR-001 GQA grouped broadcast](../adr/adr-001-gqa-grouped-broadcast.md) · model/modern_gpt.py `CausalSelfAttention`（is_causal 折叠路径）· model/ring_attention.py（在线 softmax 参照）· [MIGRATION §8.2 KV 物化全案](../MIGRATION.md) · tests/test_decode_gqa_attention.py 等 5 组 · **深化档（v0.8.5）**：[长上下文手册](./long-context.md)（RoPE 外推四档/滑窗驱逐/needle/MLA 理论档）+ [MoE 系统手册](./moe-systems.md)（路由/容量/负载/专家并行）+ [EXPERIMENTS §6.11](../EXPERIMENTS.md) | FlashAttention（Dao et al., 2022）· GQA（Ainslie et al., 2023）· YaRN（Peng et al., 2023）· Switch Transformer（Fedus et al., 2021）· DeepSeek-V2（MLA） |
| 3 | **torch.compile / Triton**：图捕获、算子融合、动态 shape、自定义 kernel | AR-v0.7.0 | **已交付** | [compile 手册](./compile-stack.md) · [Triton 微实验 tools/infra/triton_kernels.py](../../tools/infra/triton_kernels.py)（教学级手写 kernel，非生产库）· [MIGRATION.md 性能章节](../MIGRATION.md) · E-C bf16 vs fp32 实测（2.10x）· [EXPERIMENTS §6.9 Triton 三件套](../EXPERIMENTS.md) · [E-F-4 inductor 云端 SOP](../RUNNING_EXPERIMENTS.md) | torch.compile 教程 · Triton 官方 tutorial · cudagraphs |
| 4 | **分布式训练**：DP/DDP/FSDP/ZeRO、shard 与 all-gather 通信量、MFU 核算 | AR-v0.8.0 | 计划中 | [ARCHITECTURE.md 分布式节](../ARCHITECTURE.md) · tests/test_ddp_shard.py · tests/test_fsdp_checkpoint.py · E-C FSDP 单卡 148.0k tok/s 实测 · RUNNING_EXPERIMENTS F 组 SOP（E-F-1 通信带宽 / E-F-2 多卡 MFU） | ZeRO（Rajbhandari et al., 2020）· FSDP（Zhao et al., 2023）· Megatron 并行策略 |
| 5 | **弹性容错**：断点续训、幂等 resume、NCCL 超时处置 | AR-v0.8.0 | 计划中 | tests/test_resume_integrity.py · training/train_pretrain.py resume 路径 · [RUNNING_EXPERIMENTS.md](../RUNNING_EXPERIMENTS.md) 状态表 | TorchElastic · Babylon（Microsoft）|
| 6 | **推理服务**：KV cache 生命周期、batch 调度、speculative decoding、量化 | AR-v0.9.0 | 计划中 | model/modern_gpt.py `generate(draft_model/draft_tokens)`+`_verify`（speculative 路径，CLI 未暴露）· tools/experiments/run_inference_suite.py（E-B-3 suite：自动训 draft+扫 draft_len）· api_server.py · [ADR-004](../adr/adr-004-kv-cache-quantization.md) · E-B/E-C decode 吞吐实测 | vLLM/PagedAttention（Kwon et al., 2023）· speculative decoding（Leviathan et al., 2023）|
| 7 | **RLHF 系统工程**：rollout 并发、GRPO chunked loss、DPO 偏好数据管道、reward 服务化 | AR-v0.9.0 | 计划中 | [GRPO_NOTES.md](../GRPO_NOTES.md) · [ADR-002 GRPO vs PPO](../adr/adr-002-grpo-vs-ppo.md) · training/train_grpo.py（B>256 no_grad 分块）· training/train_dpo.py（logprob 对齐） | TRL · DeepSpeed-Chat · RLHF book（HuggingFace）|
| 8 | **数值精度**：四精度表示学、AMP autocast 逐行、fp32 分块累加、loss scaling | AR-v0.7.5 | **本期已交付** | [精度手册](./numerical-precision.md) · [precision_probe 探针](../../tools/infra/precision_probe.py)（四实验纯 CPU）· [EXPERIMENTS §6.10 精度表](../EXPERIMENTS.md) · training/trainer_base.py `build_amp_context`/`enable_tf32` · E-C bf16 2.10× 实测 · [triton_kernels.json bf16 容差链](../results/triton_kernels.json) | mixed-precision 论文（Micikevicius et al., 2018）· bf16 vs fp16（Intel 白皮书）· fp8（Sun et al., 2024，展望）|
| 9 | **Scaling laws**：Kaplan vs Chinchilla、B_crit、compute-optimal 记账口径、诚实拟合报告 | AR-v0.7.5 | **本期已交付** | [scaling 手册](./scaling-laws.md) · [fit_scaling 拟合器](../../tools/infra/fit_scaling.py)（多初值+病态诚实降级）· [ablation-summary.json 七组](../results/ablation-summary.json) · [budget.json 记账三件套](../results/budget.json) · [EXPERIMENTS §6.10 拟合记录](../EXPERIMENTS.md) | Kaplan et al., 2020 · Hoffmann et al., 2022（Chinchilla）· WSD（Hu et al., 2024）|
| 10 | **数据管道**：语料/去重/打包/污染防御全链路、EOT/doc 统计判据 | AR-v0.7.5 | **本期已交付** | [数据手册](./data-pipeline.md) · [data_probe 探针](../../tools/infra/data_probe.py)（真实剖析+合成实验）· data/prepare.py（dedup/packing 生产路径）· data/dedup.py `MinHashDeduplicator` · [MIGRATION §3 判据](../MIGRATION.md) · [EXPERIMENTS §6.10 数据表](../EXPERIMENTS.md) | The Pile（Gao et al., 2020）· MinHash LSH（Leskovec et al. 讲义）· 质量过滤（C4/RRF）|

> 状态说明：**已交付** = 文档+工具+微实验可用；**计划中** = 后续 AR 交付，当前仅有既有代码/实验锚点可读。

## 路线图（6 AR）

```text
AR-v0.6.0 已交付   主题1 显存/算力心算 —— 手册 + 计算器 + roofline 微实验 + F 组 SOP
AR-v0.7.0 已交付   主题2 Attention kernel 深潜 + 主题3 torch.compile/Triton —— 两本手册 + Triton 微实验三件套
AR-v0.7.5 已交付   主题8 数值精度 + 主题9 Scaling laws + 主题10 数据管道 —— 三手册 + 三纯 CPU 工具 + EXPERIMENTS §6.10
AR-v0.8.5 已交付   MoE 系统工程 + 长上下文（主题 2 深化档）—— long-context/moe-systems 两手册 + compile-stack §0 前置章 + rope_scaling 四档 + E-A-8/E-A-9/E-C-6 实验 + EXPERIMENTS §6.11
AR-v0.8.0          主题4 分布式 + 主题5 弹性容错（含 F 组完整 GPU 实验）
AR-v0.9.0          主题6 推理服务 + 主题7 RLHF 系统工程（v0.8.5 之后顺延）
```

拆分理由：全做但分批保质量——每个 AR 聚焦 1-3 个主题，工具与实验先行，防止「文档大山」。

## 全局阅读顺序建议

- **第 0 步**：[README.md](../../README.md) → [ARCHITECTURE.md](../ARCHITECTURE.md)（建立仓貌认知）
- **第 1 步**：本目录主题 1（手册 + 计算器 + roofline）
- **第 2 步**：主题 8/9/10 三手册（[数值精度](./numerical-precision.md) · [Scaling laws](./scaling-laws.md) · [数据管道](./data-pipeline.md)）+ 配套三探针 `python tools/infra/{precision_probe,fit_scaling,data_probe}.py`，实测解读见 [EXPERIMENTS §6.10](../EXPERIMENTS.md)
- **第 2.5 步**：v0.8.5 深化两手册（[长上下文](./long-context.md) · [MoE 系统](./moe-systems.md)）+ 配套工具（`tools/infra/moe_probe.py`、`tools/experiments/run_context_eval.py`/`run_needle_eval.py`），实测解读见 [EXPERIMENTS §6.11](../EXPERIMENTS.md)
- **第 3 步**：[TECH_REPORT.md](../TECH_REPORT.md) + [EXPERIMENTS.md](../EXPERIMENTS.md)（看实测数字如何支撑设计决策）
- **第 4 步**：[MIGRATION.md](../MIGRATION.md)（从 nanoGPT 到现代架构的每一处改动及理由）
- **第 5 步**：[adr/](../adr/)（6 篇架构决策记录：GQA/KV 量化/GRPO/MoE grouped GEMM 等）
- **第 6 步**：[RUNNING_EXPERIMENTS.md](../RUNNING_EXPERIMENTS.md)（亲手复现：A-D 组已执行，F 组待高算力 GPU）

## 贡献约定

- 新增主题文档放本目录，并在主题地图登记 AR 与锚点；
- 所有数值断言优先引用 `docs/results/*.json` 或 EXPERIMENTS.md 表格，禁止凭记忆写数字；
- 未交付主题一律标「计划中」并给出 AR 编号。
