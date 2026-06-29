# nanoGPT-Modern

一个**端到端的轻量级大语言模型训练-推理-对齐全栈框架**，基于 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 思想构建，在 **~50M 参数** 规模下完整验证现代 Transformer 组件的架构增益与效率 trade-off。

> 📊 技术白皮书：`docs/TECH_REPORT.md`  
> 📝 变更日志与改进清单：`docs/CHANGELOG.md`  
> 🧪 实验记录与复现命令：`docs/EXPERIMENTS.md`

---

## 核心特性

- **双轨制架构对比**：同一仓库实现 GPT-2 经典架构（BaselineGPT）与 LLaMA/Gemma 风格现代架构（ModernGPT），共享数据顺序、随机种子与超参，唯一变量是架构差异。
- **现代组件全集成**：RMSNorm、SwiGLU、RoPE、GQA、FlashAttention、QK-Norm、Attention Temperature、NTK-aware 外推、Sliding Window Attention、MoE、MTP、Paged KV Cache、KV Cache、EMA。
- **三阶段对齐流水线**：预训练 → 监督微调（SFT）→ GRPO 强化学习对齐，另含 DPO/IPO/KTO 偏好对齐入口。
- **工程化训练设施**：AMP、GradScaler、梯度累积、多模式 LR Scheduler、Early Stopping、DDP/FSDP、完整 checkpoint 状态恢复。
- **高效推理**：CUDA Events 精确计时、prefill/decode 分离、cache/no-cache 消融、`torch.compile` 加速、Speculative Decoding。
- **生态兼容**：HuggingFace 格式导出/加载、GGUF 导出（f32/f16/q8_0）、INT8/bitsandbytes 量化、lm-eval 下游评估。
- **质量保障**：234 项回归测试，核心模块通过 `mypy` 类型检查。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
python data/prepare.py --split train
python data/prepare.py --split val
python data/validate.py data/openwebtext/train.bin

# 3. 快速验证预训练（1000 步，约 5 分钟）
python training/train_pretrain.py --config config/pretrain.yaml --max_iters 1000 --n_kv_head 2

# 4. 推理
python inference/generate.py \
  --config config/generate.yaml \
  --checkpoint out/pretrain/best_ckpt.pt \
  --max_new_tokens 200

# 5. 运行回归测试
python -m pytest tests/ -q
```

---

## 完整三阶段流水线

```bash
# 预训练
python training/train_pretrain.py --config config/pretrain.yaml --use_ema --keep_last_n 3

# SFT
python training/train_sft.py --config config/sft.yaml --init_from out/pretrain/best_ckpt.pt

# GRPO 对齐
python training/train_grpo.py --config config/grpo.yaml \
  --init_from out/sft/best_sft-only.pt \
  --ref_from out/sft/best_sft-only.pt

# 评估
python evaluation/eval_alignment.py \
  --checkpoint out/grpo/best_grpo_g4.pt \
  --ref_checkpoint out/sft/best_sft-only.pt

# 预训练 Benchmark（本地 PPL + 可选 lm-eval）
python evaluation/eval_benchmark.py \
  --checkpoint out/pretrain/best_ckpt.pt \
  --data_dir data/openwebtext --split val
```

---

## Hydra / OmegaConf 入口（可选）

```bash
python training/train_pretrain_hydra.py
python training/train_pretrain_hydra.py batch_size=16 n_layer=2
python training/train_sft_hydra.py init_from=out/pretrain/best_ckpt.pt
python training/train_grpo_hydra.py init_from=out/sft/best_sft-only.pt ref_from=out/sft/best_sft-only.pt
python inference/generate_hydra.py checkpoint=out/pretrain/best_ckpt.pt prompt="Hello"
```

---

## 项目结构

```
nanogpt-modern/
├── config/                 # YAML 配置 + Hydra 配置组合
├── data/                   # 数据准备、加载、过滤、去重、混合
├── model/                  # BaselineGPT / ModernGPT / KV Cache / 量化 / HF 兼容
├── training/               # BaseTrainer + 预训练/SFT/GRPO/DPO/Iterative GRPO
├── inference/              # 生成与 benchmark 工具
├── evaluation/             # 对齐评估与标准化 benchmark
├── rewards/                # 规则奖励函数
├── utils/                  # 配置、日志、checkpoint、LR 调度、RL/DPO 工具
├── tests/                  # 回归测试（234 passed, 2 skipped）
├── docs/                   # 技术报告、变更日志、实验记录
├── export_to_hf.py         # nanoGPT -> HuggingFace
├── load_from_hf.py         # HuggingFace -> nanoGPT
├── export_gguf.py          # 导出 GGUF
├── run_ablations.py        # 训练/推理消融自动化
└── run_full_evaluation.py  # 全维度评估自动化
```

---

## 模型架构对比

| 维度 | BaselineGPT | ModernGPT |
|------|-------------|-----------|
| 归一化 | LayerNorm | RMSNorm |
| FFN | GELU (4×) | SwiGLU (8d/3) |
| 位置编码 | 可学习绝对 Embedding | RoPE |
| Attention | MHA / SDPA/manual | MHA / GQA / FlashAttention |
| KV Cache | — | 原生支持 + ring/paged manager |
| MoE | — | 可选 top-1 gating |
| Pre/Post-Norm | 支持 | 支持 |
| EMA | — | 内置 |

### 参数量对齐（12L/512D）

| 配置 | 总参数量 | KV Cache (B/tok) |
|------|----------|------------------|
| BaselineGPT | 54.6M | N/A |
| ModernGPT MHA (n_kv=8) | 54.0M | 18,432 |
| ModernGPT GQA-4KV | 51.7M | 9,216 (-50%) |
| ModernGPT GQA-2KV | 50.5M | 4,608 (-75%) |

---

## 文档索引

| 文档 | 内容 |
|------|------|
| `docs/TECH_REPORT.md` | 完整技术白皮书：架构设计、详细模块、性能容量、Roadmap |
| `docs/CHANGELOG.md` | v0.2.0 变更日志、改进清单、已知限制 |
| `docs/EXPERIMENTS.md` | 实验记录、复现命令、关键结论 |

---

## 常用命令速查

```bash
# 现代模型 + GQA-2KV
python training/train_pretrain.py --model modern --n_kv_head 2 --use_ema

# 基线模型（对照）
python training/train_pretrain.py --model baseline

# 梯度累积（effective batch = 48）
python training/train_pretrain.py --model modern --gradient_accumulation_steps 4

# FSDP 多卡
torchrun --nproc_per_node=4 training/train_pretrain.py --model modern --fsdp

# 断点续训
python training/train_pretrain.py --model modern --resume out/pretrain/latest_ckpt.pt

# 推理消融
python inference/generate.py --checkpoint out/pretrain/best_ckpt.pt \
  --max_new_tokens 200 400 600 --num_samples 30 --output_json out/ablation.json

# 导出 HuggingFace 格式
python export_to_hf.py --checkpoint out/pretrain/best_ckpt.pt --out_dir hf/nanogpt-modern

# 导出 GGUF
python export_gguf.py --checkpoint out/pretrain/best_ckpt.pt \
  --out out/pretrain/best_ckpt.q8_0.gguf --quant q8_0
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

---

## 许可

基于 nanoGPT 思想构建，仅供研究与学习使用。
