# nanoGPT-Modern v0.3.0

一个**端到端的轻量级大语言模型训练-推理-对齐全栈框架**，基于 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 思想构建，在 **~50M 参数** 规模下完整验证现代 Transformer 组件的架构增益与效率 trade-off。v0.3.0 是一次全维度升级，覆盖模型架构、训练系统、数据管道、推理服务、评估观测与工程 DevOps 六大领域共 **42 项优化**。

> 📊 技术白皮书：`docs/TECH_REPORT.md`
> 📝 变更日志与改进清单：`docs/CHANGELOG.md`
> 🏗️ 架构决策记录：`docs/adr/`
> 🧪 实验记录与复现命令：`docs/EXPERIMENTS.md`

---

## 核心特性

### 双轨制架构对比
- 同一仓库实现 **GPT-2 经典架构**（BaselineGPT）与 **LLaMA/Gemma 风格现代架构**（ModernGPT），共享数据顺序、随机种子与超参，唯一变量是架构差异。

### 现代组件全集成（v0.3.0 增强）
- **归一化**: RMSNorm（fused kernel 自动探测）
- **激活**: SwiGLU (8d/3) / MoE top-1 gating（分组 GEMM 优化）
- **位置编码**: RoPE + NTK-aware 长度外推
- **注意力**: MHA / GQA（零拷贝 grouped broadcast）/ FlashAttention / SDPA 自动后端 / Sliding Window / QK-Norm / Attention Temperature
- **KV Cache**: 静态环形缓存 + Paged KV Cache + **INT8/FP8 量化缓存**
- **推理加速**: torch.compile / **投机解码（batch > 1）** / **连续批处理（Continuous Batching）**
- **多 Token 预测**: MTP (n_future heads)
- **序列并行**: Ring Attention (纯 PyTorch blockwise)

### 三阶段对齐流水线 + 偏好学习
- **预训练**: OpenWebText 因果语言建模
- **SFT**: 监督微调（算术 + **Chain-of-Thought 推理格式**）
- **GRPO**: 强化学习对齐（组相对策略优化，**G8/G16 扩展**，**KL 自适应**）
- **DPO / IPO / KTO**: 偏好对齐（**从 GRPO 自动生成偏好对**，完整训练循环）
- **Iterative GRPO**: 拒绝采样 + EMA ref 更新循环

### 工程化训练设施（v0.3.0 增强）
- **统一训练抽象**: `BaseTrainer` 模板方法（预训练/SFT/GRPO/DPO/Iterative GRPO 共享基础设施）
- **分布式**: DDP / FSDP / **张量并行骨架（TP）** / **序列并行（SP）**
- **混合精度**: AMP bf16/fp16 + GradScaler
- **梯度累积**: 所有训练阶段支持
- **学习率调度**: cosine / linear / WSD / constant + warmup
- **EMA**: 影子权重保存/恢复
- **Early Stopping**: 基于 val loss 的耐心机制
- **Checkpoint 生命周期**: `keep_last_n` 自动清理 + 完整状态恢复（模型 + 优化器 + scaler + scheduler + EMA + RNG + resume_offset）
- **监控告警**: **Loss Spike 自动检测**（>3σ 触发 LR 降低）、**梯度范数分层监控**、**显存 Profiler**、**吞吐实时监控**
- **配置校验**: **Pydantic 结构化校验**（`n_head % n_kv_head == 0` 等规则自动验证）

### 数据管道（v0.3.0 全面增强）
- **全局 Shuffle**: 预计算随机索引 `.shuffle.idx`，训练时全局随机化
- **质量评分**: `QualityScoreFilter` 基于规则/FastText 评分，支持分层采样
- **增量去重**: `MinHashDeduplicator` 支持增量签名持久化与追加比对
- **多语言**: `MultilingualTokenizer`（tiktoken + SentencePiece）+ `LanguageDetector` + 按语言权重采样
- **代码数据**: `CodeDataset` 支持本地目录/The Stack 加载 + AST 语法过滤 + 按语言分组采样
- **数学增强**: `ChainOfThoughtDataset` 生成 GSM8K 风格逐步推理数据（`<reasoning>` + `<answer>`）
- **对话模板**: `ChatTemplate` 抽象（chatml / llama-2 / gemma 格式）+ system prompt 注入
- **文档边界**: DocBoundaryDataset / PackingDataset（跨文档 mask + `document_ids`）

### 高效推理（v0.3.0 全面增强）
- **CUDA Events** 精确计时，prefill/decode 分离
- **KV Cache**: 环形缓存 + Paged KV Cache + **INT8/FP8 量化**（显存节省 50%）
- **投机解码**: draft-then-verify，支持 **batch > 1**（tree-based），接受率自适应降级
- **连续批处理**: `RequestQueue` + `ContinuousBatchScheduler` + 动态 batch 拼接，Prefix Cache 共享
- **torch.compile**: 自动探测与回退
- **Batch 推理**: finished mask 逐序列提前终止

### 评估与可观测性（v0.3.0 新增）
- **标准 Benchmark**: `benchmark_suite.py` 预配置 MMLU / ARC / HellaSwag / Winogrande / HumanEval / GSM8K / **Needle-in-Haystack（长上下文）**
- **对齐评估**: 本地 PPL + 可选 lm-eval 下游任务 + **胜率对比（Win Rate）**
- **注意力可视化**: `AttentionVisualizer` 输出 heatmap + `LogitLens` 各层 top-k 预测词
- **性能 Profiler**: Chrome trace 导出 + 显存分层报告
- **消融自动化**: `run_ablations.py` 训练/推理矩阵一键运行

### 生态兼容与部署（v0.3.0 新增）
- **HuggingFace**: 格式导出/加载（`config.json` + `model.safetensors`）
- **GGUF**: 导出（f32/f16/q8_0），扩展支持 Q4_0 / Q5_0 / Q6_K 类型标注
- **ONNX**: `export_to_onnx.py` 动态轴导出 + ONNX Runtime 验证
- **量化**: INT8 / bitsandbytes 8-bit/4-bit / **SmoothQuant / AWQ 集成**（骨架）
- **API 服务**: `api_server.py` FastAPI 兼容 OpenAI API（`/v1/completions` + `/v1/chat/completions`）
- **容器化**: Dockerfile + docker-compose（训练 + TensorBoard + API 服务）
- **预提交**: `.pre-commit-config.yaml`（black + ruff + mypy + pytest）

### 质量保障
- **回归测试**: 234 项 + 新增覆盖，核心模块通过 `mypy` 类型检查
- **CI/CD**: GitHub Actions（pytest + mypy + lint + 短训练 smoke test）
- **确定性**: `test_determinism.py` 固定种子重复训练 loss 曲线完全一致

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 可选：评估、量化、多语言扩展
pip install -e ".[eval,quant, multilingual]"
```

### 2. 准备数据

```bash
# 全局 shuffle 数据（推荐，v0.3.0）
python data/prepare.py --split train
python data/prepare.py --split val
python data/validate.py data/openwebtext/train.bin

# 生成全局随机索引（可选，提升训练质量）
python -c "from data.openwebtext import generate_shuffle_index; generate_shuffle_index('data/openwebtext/train.bin', seed=1337)"
```

### 3. 快速验证预训练（1000 步，约 5 分钟）

```bash
python training/train_pretrain.py --config config/pretrain.yaml --max_iters 1000 --n_kv_head 2
```

### 4. 推理

```bash
# 标准推理
python inference/generate.py \
  --config config/generate.yaml \
  --checkpoint out/pretrain/best_ckpt.pt \
  --max_new_tokens 200

# 连续批处理推理（v0.3.0）
python inference/continuous_batching.py \
  --checkpoint out/pretrain/best_ckpt.pt \
  --prompts "The future of AI is" "Explain quantum computing"
```

### 5. 运行回归测试

```bash
python -m pytest tests/ -q
```

### 6. 启动 API 服务（v0.3.0）

```bash
python api_server.py \
  --checkpoint out/pretrain/best_ckpt.pt \
  --model modern \
  --port 8000

# 测试
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The future of AI is", "max_tokens": 50}'
```

---

## 完整三阶段流水线 + 偏好对齐

```bash
# 预训练（推荐启用 EMA 和全局 shuffle）
python training/train_pretrain.py --config config/pretrain.yaml --use_ema --keep_last_n 3

# SFT（注入 CoT 格式推理能力）
python training/train_sft.py --config config/sft.yaml --init_from out/pretrain/best_ckpt.pt

# GRPO 对齐（G4/G8 可选，KL 自适应）
python training/train_grpo.py --config config/grpo.yaml \
  --init_from out/sft/best_sft-only.pt \
  --ref_from out/sft/best_sft-only.pt \
  --group_size 8

# DPO 偏好对齐（从 GRPO 自动生成偏好对）
python training/train_dpo.py \
  --init_from out/sft/best_sft-only.pt \
  --preference_source grpo \
  --grpo_checkpoint out/grpo/best_grpo_g8.pt \
  --win_rate_eval

# 评估
python evaluation/eval_alignment.py \
  --checkpoint out/grpo/best_grpo_g8.pt \
  --ref_checkpoint out/sft/best_sft-only.pt

# 标准 Benchmark（v0.3.0）
python evaluation/benchmark_suite.py \
  --checkpoint out/pretrain/best_ckpt.pt \
  --tasks mmlu,arc,hellaswag,gsm8k \
  --output_json out/benchmark_results.json

# 长上下文测试（Needle-in-Haystack）
python evaluation/needle_in_haystack.py \
  --checkpoint out/pretrain/best_ckpt.pt \
  --context_lengths 1024,2048,4096,8192
```

---

## Hydra / OmegaConf 入口（可选）

```bash
python training/train_pretrain_hydra.py
python training/train_pretrain_hydra.py batch_size=16 n_layer=2
python training/train_sft_hydra.py init_from=out/pretrain/best_ckpt.pt
python training/train_grpo_hydra.py init_from=out/sft/best_sft-only.pt ref_from=out/sft/best_sft-only.pt
python training/train_dpo_hydra.py init_from=out/sft/best_sft-only.pt preference_source=grpo
python inference/generate_hydra.py checkpoint=out/pretrain/best_ckpt.pt prompt="Hello"
```

---

## 项目结构

```
nanogpt-modern/
├── config/                  # YAML 配置 + Hydra 配置组合
├── data/                    # 数据准备、加载、过滤、去重、混合、多语言、代码、CoT
│   ├── prepare.py           # 多进程 shard 化预处理
│   ├── filter.py            # 质量过滤（Length/Repetition/Regex/FastText）
│   ├── dedup.py             # MinHash+LSH 去重（增量模式）
│   ├── mixer.py             # 多源混合
│   ├── openwebtext.py       # Memmap/Packing/DocBoundary + 全局 Shuffle
│   ├── quality_filter.py    # 质量评分与分层采样（v0.3.0）
│   ├── multilingual.py      # 多语言 tokenizer + 检测 + 采样（v0.3.0）
│   ├── code.py              # 代码数据管道 + AST 过滤（v0.3.0）
│   ├── arithmetic.py        # 算术数据集 + Chain-of-Thought（v0.3.0）
│   └── chat_templates.py    # 对话模板系统（v0.3.0）
├── model/                   # BaselineGPT / ModernGPT / KV Cache / 量化 / HF 兼容
│   ├── modern_gpt.py        # ModernGPT（RMSNorm + SwiGLU + RoPE + GQA + MoE grouped GEMM）
│   ├── baseline_gpt.py      # BaselineGPT（GPT-2 风格）
│   ├── attention_utils.py   # SDPA 后端切换 + GQA 广播探测
│   ├── flash_attention.py   # FlashAttention 封装
│   ├── kv_cache_utils.py    # 环形缓存 + Paged KV Cache + **INT8/FP8 量化**（v0.3.0）
│   ├── quantization.py      # INT8 / bitsandbytes 量化
│   ├── paged_kv_cache.py    # 块表 KV Cache
│   ├── ring_attention.py    # 序列并行 Ring Attention
│   └── hf_model.py          # HuggingFace 兼容
├── training/                # BaseTrainer + 预训练/SFT/GRPO/DPO/Iterative GRPO
│   ├── trainer_base.py      # 统一训练抽象（DDP/FSDP/AMP/EMA/Early Stop）
│   ├── train_pretrain.py    # 预训练
│   ├── train_sft.py         # 监督微调
│   ├── train_grpo.py        # GRPO 强化学习（批量化 + 梯度累积）
│   ├── train_dpo.py         # DPO/IPO/KTO 偏好对齐（**完整训练 + 胜率评估**，v0.3.0）
│   └── iterative_grpo.py  # Iterative GRPO
├── inference/               # 生成与 benchmark 工具
│   ├── generate.py          # 推理脚本（KV Cache / torch.compile / 投机解码）
│   ├── generate_utils.py    # 采样工具
│   ├── continuous_batching.py  # **连续批处理 + Prefix Cache**（v0.3.0）
│   └── visualize_attention.py  # **注意力可视化 + Logit Lens**（v0.3.0）
├── evaluation/              # 对齐评估与标准化 benchmark
│   ├── eval_alignment.py    # 对齐质量评估
│   ├── eval_benchmark.py    # 本地 PPL + lm-eval
│   ├── benchmark_suite.py   # **标准 Benchmark 集成**（v0.3.0）
│   ├── eval_win_rate.py     # **胜率评估**（v0.3.0）
│   └── needle_in_haystack.py  # **长上下文 Needle-in-Haystack**（v0.3.0）
├── rewards/                 # 规则奖励函数 + 格式/过程/正确性评分
├── utils/                   # 配置、日志、checkpoint、LR 调度、RL/DPO 工具
│   ├── config.py            # YAML + argparse + **Pydantic 校验**（v0.3.0）
│   ├── config_models.py     # Pydantic 配置模型（v0.3.0）
│   ├── checkpoint.py        # 完整 checkpoint 管理
│   ├── logging.py           # wandb + TensorBoard 多后端日志
│   ├── lr_scheduler.py      # cosine / linear / WSD / constant
│   ├── rl_utils.py          # GRPO 工具
│   ├── dpo_utils.py         # DPO/IPO/KTO 损失
│   ├── monitoring.py        # **Loss Spike / 梯度范数 / 显存 / 吞吐监控**（v0.3.0）
│   └── profiler.py          # **PyTorch Profiler + Chrome trace**（v0.3.0）
├── tests/                   # 回归测试（234 passed + 新增）
├── docs/                    # 技术报告、变更日志、实验记录、架构决策记录
│   ├── TECH_REPORT.md       # 完整技术白皮书
│   ├── CHANGELOG.md         # v0.2.0 → v0.3.0 变更日志、改进清单、已知限制、未来方向
│   ├── EXPERIMENTS.md       # 实验记录与复现命令
│   └── adr/                 # **架构决策记录**（v0.3.0）
├── export_to_hf.py          # nanoGPT -> HuggingFace
├── load_from_hf.py          # HuggingFace -> nanoGPT
├── export_gguf.py           # 导出 GGUF
├── export_to_onnx.py        # **ONNX 导出 + 验证**（v0.3.0）
├── api_server.py            # **FastAPI OpenAI 兼容服务**（v0.3.0）
├── profile_training.py      # **训练性能 Profiler**（v0.3.0）
├── profile_inference.py     # **推理性能 Profiler**（v0.3.0）
├── run_ablations.py         # 训练/推理消融自动化
├── run_full_evaluation.py   # 全维度评估自动化
├── Dockerfile               # **多阶段容器**（v0.3.0）
├── docker-compose.yml       # **训练 + TensorBoard + API**（v0.3.0）
├── .github/workflows/       # **CI/CD**（v0.3.0）
│   └── ci.yml
├── .pre-commit-config.yaml  # **预提交钩子**（v0.3.0）
├── pyproject.toml           # 项目配置 + 可选依赖组
└── README.md                # 本文档
```

---

## 模型架构对比

| 维度 | BaselineGPT | ModernGPT |
|------|-------------|-----------|
| 归一化 | LayerNorm | RMSNorm（fused kernel） |
| FFN | GELU (4×) | SwiGLU (8d/3) / **MoE grouped GEMM** |
| 位置编码 | 可学习绝对 Embedding | RoPE + **NTK-aware 外推** |
| Attention | MHA / SDPA/manual | MHA / **GQA grouped broadcast** / FlashAttention / Sliding Window / QK-Norm |
| KV Cache | — | 原生 + ring/paged + **INT8/FP8 量化** |
| MoE | — | 可选 top-1 gating + **分组 GEMM 优化** |
| 预训练长度 | block_size | block_size + Ring Attention |
| Pre/Post-Norm | 支持 | 支持 |
| EMA | — | 内置 |

### 参数量对齐（12L/512D）

| 配置 | 总参数量 | KV Cache (B/tok) |
|------|----------|------------------|
| BaselineGPT | 54.6M | N/A |
| ModernGPT MHA (n_kv=8) | 54.0M | 18,432 |
| ModernGPT GQA-4KV | 51.7M | 9,216 (-50%) |
| ModernGPT GQA-2KV | 50.5M | 4,608 (-75%) |
| ModernGPT GQA-2KV + KV-INT8 | 50.5M | 2,304 (-87.5%) |

---

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/TECH_REPORT.md` | 完整技术白皮书：架构设计、详细模块、性能容量、Roadmap |
| `docs/CHANGELOG.md` | v0.2.0 → v0.3.0 变更日志、改进清单、已知限制、未来方向 |
| `docs/EXPERIMENTS.md` | 实验记录、复现命令、关键结论 |
| `docs/adr/` | 架构决策记录（ADR）：为何选择 GRPO、grouped-broadcast、Hydra 等 |

---

## 常用命令速查

### 训练

```bash
# 现代模型 + GQA-2KV + EMA
python training/train_pretrain.py --model modern --n_kv_head 2 --use_ema

# 基线模型（对照）
python training/train_pretrain.py --model baseline

# 梯度累积（effective batch = 48）
python training/train_pretrain.py --model modern --gradient_accumulation_steps 4

# FSDP 多卡
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=4 training/train_pretrain.py --model modern --fsdp

# 断点续训
python training/train_pretrain.py --model modern --resume out/pretrain/latest_ckpt.pt

# 启用监控（Loss Spike 检测 + 吞吐监控）
python training/train_pretrain.py --model modern --use_monitoring
```

### 推理与评估

```bash
# 推理消融
python inference/generate.py --checkpoint out/pretrain/best_ckpt.pt \
  --max_new_tokens 200 400 600 --num_samples 30 --output_json out/ablation.json

# 注意力可视化
python inference/visualize_attention.py --checkpoint out/pretrain/best_ckpt.pt \
  --prompt "The future of AI is" --output_dir out/viz/

# 标准 Benchmark
python evaluation/benchmark_suite.py --checkpoint out/pretrain/best_ckpt.pt \
  --tasks mmlu,arc,hellaswag,gsm8k --output_json out/benchmark.json

# 胜率评估
python evaluation/eval_win_rate.py --policy out/grpo/best_grpo_g8.pt \
  --reference out/sft/best_sft-only.pt --num_samples 100
```

### 导出

```bash
# 导出 HuggingFace 格式
python export_to_hf.py --checkpoint out/pretrain/best_ckpt.pt --out_dir hf/nanogpt-modern

# 导出 GGUF
python export_gguf.py --checkpoint out/pretrain/best_ckpt.pt \
  --out out/pretrain/best_ckpt.q8_0.gguf --quant q8_0

# 导出 ONNX
python export_to_onnx.py --checkpoint out/pretrain/best_ckpt.pt \
  --out out/pretrain/model.onnx --model modern
```

### 容器化

```bash
# 构建镜像
docker build -t nanogpt-modern:latest .

# 启动训练 + TensorBoard + API
docker-compose up

# 仅训练
docker-compose up train
```

---

## 参考文献

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Andrej Karpathy
- [RMSNorm] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
- [SwiGLU] GLU Variants Improve Transformer (Shazeer, 2020)
- [RoPE] RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
- [GQA] GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)
- [GRPO] Group Relative Policy Optimization (DeepSeekMath, Shao et al., 2024)
- [LLaMA] LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
- [DPO] Direct Preference Optimization (Rafailov et al., 2023)
- [KIVI] KV Cache Quantization (Liu et al., 2024)
- [vLLM] Efficient Memory Management for LLM Serving (Kwon et al., 2023)

---

## 许可

基于 nanoGPT 思想构建，仅供研究与学习使用。MIT License。

---

*版本: v0.3.0 | 最后更新: 2026-06-30*
