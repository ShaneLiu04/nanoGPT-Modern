# AI Infra 学习路径（Learning Track）

> 面向想深入「大模型系统工程」的读者：以本仓 nanoGPT-Modern 的**真实代码、真实实验数据**为锚点，把 AI infra 的核心主题学透。
> 原则：**每个公式都能在本仓找到交叉验证**——不是博客式转载，而是「推导 → 工具复算 → 实测对照」三段式。

## 如何使用本路径

1. 先读 [显存/算力心算手册](./memory-flops-arithmetic.md)（主题 1，本期已交付），建立量化直觉；
2. 用计算器复算手册每个数字：`python -m utils.infra_math --help`；
3. 本机跑 roofline 微实验：`python tools/infra/run_roofline.py --quick`；
4. 按下方主题地图逐主题深入，每主题先看「本仓锚点」再看延伸阅读；
5. 完整 GPU 实验按 [RUNNING_EXPERIMENTS.md](../RUNNING_EXPERIMENTS.md) F 组 SOP 执行（需高算力 GPU）。

## 主题地图（7 主题 → 4 个 AR）

| # | 主题 | 交付 AR | 状态 | 本仓锚点 | 延伸阅读 |
|---|------|---------|------|----------|----------|
| 1 | **显存与算力心算**：参数/梯度/优化器/激活/KV 五项分解、6ND FLOPs、roofline 判据 | AR-v0.6.0 | **本期已交付** | [心算手册](./memory-flops-arithmetic.md) · [计算器 utils/infra_math.py](../../utils/infra_math.py) · [roofline 微实验](../../tools/infra/run_roofline.py) · [EXPERIMENTS E-A 显存表](../EXPERIMENTS.md) · [MIGRATION §8.2 KV 斜率](../MIGRATION.md) · [ADR-004 KV 量化](../adr/adr-004-kv-cache-quantization.md) | Transformer Math 101（Kipply）· Megatron-LM 论文（Shoeybi et al., 2019）· Roofline 模型（Williams et al., 2009）· Adam（Kingma & Ba, 2014） |
| 2 | **Attention kernel 深潜**：SDPA/GQA 内存布局、flash 注意力分块、scores [B,H,T,T] 物化代价 | AR-v0.7.0 | **本期已交付** | [attention 手册](./attention-kernels.md) · [ADR-001 GQA grouped broadcast](../adr/adr-001-gqa-grouped-broadcast.md) · model/modern_gpt.py `CausalSelfAttention`（is_causal 折叠路径）· model/ring_attention.py（在线 softmax 参照）· [MIGRATION §8.2 KV 物化全案](../MIGRATION.md) · tests/test_decode_gqa_attention.py 等 5 组 | FlashAttention（Dao et al., 2022）· GQA（Ainslie et al., 2023） |
| 3 | **torch.compile / Triton**：图捕获、算子融合、动态 shape、自定义 kernel | AR-v0.7.0 | **本期已交付** | [compile 手册](./compile-stack.md) · [Triton 微实验 tools/infra/triton_kernels.py](../../tools/infra/triton_kernels.py)（教学级手写 kernel，非生产库）· [MIGRATION.md 性能章节](../MIGRATION.md) · E-C bf16 vs fp32 实测（2.10x）· [EXPERIMENTS §6.9 Triton 三件套](../EXPERIMENTS.md) · [E-F-4 inductor 云端 SOP](../RUNNING_EXPERIMENTS.md) | torch.compile 教程 · Triton 官方 tutorial · cudagraphs |
| 4 | **分布式训练**：DP/DDP/FSDP/ZeRO、shard 与 all-gather 通信量、MFU 核算 | AR-v0.8.0 | 计划中 | [ARCHITECTURE.md 分布式节](../ARCHITECTURE.md) · tests/test_ddp_shard.py · tests/test_fsdp_checkpoint.py · E-C FSDP 单卡 148.0k tok/s 实测 · RUNNING_EXPERIMENTS F 组 SOP（E-F-1 通信带宽 / E-F-2 多卡 MFU） | ZeRO（Rajbhandari et al., 2020）· FSDP（Zhao et al., 2023）· Megatron 并行策略 |
| 5 | **弹性容错**：断点续训、幂等 resume、NCCL 超时处置 | AR-v0.8.0 | 计划中 | tests/test_resume_integrity.py · training/train_pretrain.py resume 路径 · [RUNNING_EXPERIMENTS.md](../RUNNING_EXPERIMENTS.md) 状态表 | TorchElastic · Babylon（Microsoft）|
| 6 | **推理服务**：KV cache 生命周期、batch 调度、speculative decoding、量化 | AR-v0.9.0 | 计划中 | model/modern_gpt.py `generate(draft_model/draft_tokens)`+`_verify`（speculative 路径，CLI 未暴露）· tools/experiments/run_inference_suite.py（E-B-3 suite：自动训 draft+扫 draft_len）· api_server.py · [ADR-004](../adr/adr-004-kv-cache-quantization.md) · E-B/E-C decode 吞吐实测 | vLLM/PagedAttention（Kwon et al., 2023）· speculative decoding（Leviathan et al., 2023）|
| 7 | **RLHF 系统工程**：rollout 并发、GRPO chunked loss、DPO 偏好数据管道、reward 服务化 | AR-v0.9.0 | 计划中 | [GRPO_NOTES.md](../GRPO_NOTES.md) · [ADR-002 GRPO vs PPO](../adr/adr-002-grpo-vs-ppo.md) · training/train_grpo.py（B>256 no_grad 分块）· training/train_dpo.py（logprob 对齐） | TRL · DeepSpeed-Chat · RLHF book（HuggingFace）|

> 状态说明：**本期已交付** = 文档+工具+微实验可用；**计划中** = 后续 AR 交付，当前仅有既有代码/实验锚点可读。

## 4-AR 路线图

```text
AR-v0.6.0 已交付   主题1 显存/算力心算 —— 手册 + 计算器 + roofline 微实验 + F 组 SOP
AR-v0.7.0（本期）  主题2 Attention kernel 深潜 + 主题3 torch.compile/Triton —— 两本手册 + Triton 微实验三件套
AR-v0.8.0         主题4 分布式 + 主题5 弹性容错（含 F 组完整 GPU 实验）
AR-v0.9.0         主题6 推理服务 + 主题7 RLHF 系统工程
```

拆分理由：全做但分批保质量——每个 AR 聚焦 1-3 个主题，工具与实验先行，防止「文档大山」。

## 全局阅读顺序建议

- **第 0 步**：[README.md](../../README.md) → [ARCHITECTURE.md](../ARCHITECTURE.md)（建立仓貌认知）
- **第 1 步**：本目录主题 1（手册 + 计算器 + roofline）
- **第 2 步**：[TECH_REPORT.md](../TECH_REPORT.md) + [EXPERIMENTS.md](../EXPERIMENTS.md)（看实测数字如何支撑设计决策）
- **第 3 步**：[MIGRATION.md](../MIGRATION.md)（从 nanoGPT 到现代架构的每一处改动及理由）
- **第 4 步**：[adr/](../adr/)（6 篇架构决策记录：GQA/KV 量化/GRPO/MoE grouped GEMM 等）
- **第 5 步**：[RUNNING_EXPERIMENTS.md](../RUNNING_EXPERIMENTS.md)（亲手复现：A-D 组已执行，F 组待高算力 GPU）

## 贡献约定

- 新增主题文档放本目录，并在主题地图登记 AR 与锚点；
- 所有数值断言优先引用 `docs/results/*.json` 或 EXPERIMENTS.md 表格，禁止凭记忆写数字；
- 未交付主题一律标「计划中」并给出 AR 编号。
