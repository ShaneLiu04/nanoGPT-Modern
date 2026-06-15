# nanoGPT-Modern

一个**端到端的轻量级大语言模型训练-推理-对齐全栈框架**，基于 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 思想构建，在 **~50M 参数** 规模下完整验证现代 Transformer 组件的架构增益与效率 trade-off。

> :clipboard: 完整的系统改进清单见 [IMPROVEMENT_CHECKLIST.md](IMPROVEMENT_CHECKLIST.md)  
> :bar_chart: 深度诊断与优化报告见 [IMPROVEMENT_REPORT.md](IMPROVEMENT_REPORT.md)  
> :white_check_mark: 关键阻塞 Bug 已修复，新增回归测试 `tests/test_bugfixes.py`

---

## 为什么做这个项目？

### 动机

当前大模型研究被封锁在“黑盒 API + 闭源权重”的范式中，研究者很难回答一个基础问题：

> **每一个现代 Transformer 组件（RMSNorm、SwiGLU、RoPE、GQA、KV Cache…）到底带来了多少真实收益？**

nanoGPT-Modern 的设计目标是在**可完整复现的轻量级规模**（~50M 参数）下，构建一条从预训练、SFT 到 RL 对齐的完整流水线，并对现代架构进行**受控对比实验**：

- **单一变量原则**：BaselineGPT 与 ModernGPT 共享相同的数据顺序、随机种子、训练超参，确保对比结果仅反映架构差异。
- **全栈可观测**：训练 loss、推理吞吐、KV Cache 显存、对齐准确率等指标可在同一套代码中横向对比。
- **可复现基座**：固定种子、CUDA Events 精确计时、完整 checkpoint 状态保存、回归测试覆盖核心路径。

### 核心创新点

1. **双轨制架构对比**：在同一仓库中实现 GPT-2 经典架构与 LLaMA/Gemma 风格现代架构，可直接切换、公平对比。
2. **现代组件全集成**：RMSNorm、SwiGLU、RoPE、GQA、KV Cache、MoE（实验性）、EMA 全部可配置。
3. **三阶段对齐流水线**：预训练 → 监督微调（SFT）→ GRPO 强化学习对齐，覆盖大模型后训练完整链路。
4. **工程化训练设施**：AMP、GradScaler、梯度累积、多模式 LR Scheduler、Early Stopping、DDP/FSDP、完整 checkpoint 状态恢复。
5. **严谨的性能测量**：CUDA Events 分离 prefill/decode 阶段，cache/no-cache 双路径消融，KV Cache 显存精确量化。

---

## 快速开始 (Quick Start)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（流式处理，不写 Python list 到内存）
python data/prepare.py --split train
python data/prepare.py --split val
python data/validate.py data/openwebtext/train.bin

# 3. 快速验证预训练 (1000步, ~5分钟)
python training/train_pretrain.py --config config/pretrain.yaml --max_iters 1000 --n_kv_head 2

# 4. 推理
python inference/generate.py --config config/generate.yaml --checkpoint out/pretrain/best_ckpt.pt --max_new_tokens 200

# 5. 完整三阶段流水线（YAML 配置 + 命令行覆盖）
python training/train_pretrain.py --config config/pretrain.yaml --use_ema
python training/train_sft.py --config config/sft.yaml --init_from out/pretrain/best_ckpt.pt
python training/train_grpo.py --config config/grpo.yaml --init_from out/sft/best_sft-only.pt --ref_from out/sft/best_sft-only.pt
python evaluation/eval_alignment.py --checkpoint out/grpo/best_ckpt.pt --ref_checkpoint out/sft/best_sft-only.pt

# 6. 运行回归测试
python tests/test_bugfixes.py
```

---

## 系统全景：各组件职能与协作关系

本项目围绕 "预训练 → 监督微调 → RL 对齐" 三阶段流水线组织，每层由独立模块构成，通过配置对象串接。

### 1. 数据管道层 --- 生产训练数据

| 模块                  | 职能                                                         |
| --------------------- | ------------------------------------------------------------ |
| `data/prepare.py`     | 流式下载 OpenWebText → 批量 BPE tokenize → 写入固定大小 binary shards + 全局 `.bin` + `.idx` 元数据 |
| `data/openwebtext.py` | `MemmapDataset` 从二进制文件流式读取；`DocBoundaryDataset` 按文档边界切分，支持断点续训 offset |
| `data/arithmetic.py`  | 合成算术数据集生成器：`easy` (单步四则)、`medium` (2-3步混合+括号)、`hard` (5种多样化模板) |
| `data/validate.py`    | 数据质检：token 范围检查、词表覆盖率、EOT 频率统计、随机 decode 采样 |

### 2. 模型层 --- 双轨制架构对比

| 模块                      | 职能                                                         |
| ------------------------- | ------------------------------------------------------------ |
| `model/baseline_gpt.py`   | GPT-2 经典架构：LayerNorm + GELU FFN + 绝对位置编码。支持 SDPA/manual 双后端切换、Pre/Post-Norm |
| `model/modern_gpt.py`     | ModernGPT：RMSNorm + SwiGLU + RoPE + GQA + MoE(可选) + EMA。支持 KV Cache 原生推理 |
| `model/kv_cache_utils.py` | `KVCacheManager`：管理逐层 past KV 张量，支持滑动窗口截断、批量重置、GQA 适配 |

**两模型的关系**：共享相同的训练超参、数据顺序、随机种子，确保对比实验中 **唯一变量是架构差异**。

### 3. 训练系统层 --- 三阶段流水线

| 模块                         | 职能                         | 核心特性                                                     |
| ---------------------------- | ---------------------------- | ------------------------------------------------------------ |
| `training/train_pretrain.py` | 语言建模预训练 (OpenWebText) | AMP + GradScaler + 梯度累积 + LR Scheduler (cosine/linear/wsd/constant) + EMA + EarlyStopping + DDP/FSDP + 完整训练状态 checkpoint |
| `training/train_sft.py`      | 监督微调 (算术数据)          | LR Scheduler + 数据混叠 (easy+medium+hard)                   |
| `training/train_grpo.py`     | GRPO 对齐 (算术任务)         | PPO Clip + KL Penalty + Old/New Policy 分离 + Dropout Guard  |
| `training/iterative_grpo.py` | 迭代 RLHF                    | 周期性更新参考模型 + Rejection Sampling SFT                  |

**三阶段关系**：预训练 → SFT → GRPO 是串行依赖链。预训练产出基座模型，SFT 在算术数据上注入任务格式，GRPO 通过规则奖励进一步优化正确率。

### 4. 推理系统层 --- 生成与性能测量

| 模块                    | 职能                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `inference/generate.py` | CUDA Events 精确计时 + prefill/decode 阶段分离 + cache/no-cache 消融对比 |

**推理管线**：`ModernGPT.generate()` 支持两种路径：

- **No-cache 路径**：每次生成一个 token 都完整 forward 全序列 (O(T^2) 复杂度)
- **Cache 路径**：prefill 阶段一次编码全 prompt → decode 阶段逐 token forward 仅新 token，复用 cached KV (O(T) 复杂度)

### 5. 评估与奖励系统层

| 模块                           | 职能                                                         |
| ------------------------------ | ------------------------------------------------------------ |
| `evaluation/eval_alignment.py` | 全维度评估：accuracy / format_pass_rate / invalid_rate / reward / KL_divergence |
| `rewards/rule_reward.py`       | 规则奖励函数：格式分 (是否包含 `<answer>...</answer>` + 数值) + 正确性分 (数值误差 < 1e-4) |

### 6. 工具与基础设施层

| 模块                    | 职能                                                         |
| ----------------------- | ------------------------------------------------------------ |
| `config/`               | YAML 配置文件 (pretrain.yaml, sft.yaml, grpo.yaml, generate.yaml)，统一管理超参 |
| `utils/lr_scheduler.py` | 统一 LR Scheduler：cosine / linear / wsd / constant 四种模式 |
| `utils/logging.py`      | 日志系统：wandb + TensorBoard + console 三后端，支持 auto-degrade fallback |
| `utils/checkpoint.py`   | Checkpoint 持久化：模型 + 优化器 + iter + config + RNG + scaler + scheduler + EMA + resume_offset，支持 FSDP Full State Dict |

### 数据流全景图

```
OpenWebText (~1.13B tokens)
    |
    v
prepare.py ---> train.bin / val.bin
    |
    v
train_pretrain.py (BaselineGPT / ModernGPT) ---> 基座模型 checkpoints
    |
    |  算术数据集 (easy/medium/hard)
    |      |
    |      v
    |  train_sft.py ---> SFT checkpoints
    |      |
    |      v
    |  train_grpo.py ---> GRPO aligned checkpoints
    |      |
    |      v
    |  eval_alignment.py ---> 评估报告
    |
    v
generate.py ---> 推理消融 (cache vs no-cache)
```

---

## 模型架构设计

### 双轨制对比

| 维度          | BaselineGPT                      | ModernGPT                            |
| ------------- | -------------------------------- | ------------------------------------ |
| 归一化        | LayerNorm (含 bias)              | **RMSNorm** (无 bias)                |
| 前馈网络      | GELU (4x 扩展, 8d^2 参数)        | **SwiGLU** (gate/up/down, 3d*hidden) |
| 位置编码      | 可学习绝对位置 Embedding         | **RoPE** 旋转位置编码                |
| Attention     | SDPA (默认) / manual causal mask | SDPA (FlashAttention 自动分发)       |
| KV Cache      | ---                              | 原生支持 + KVCacheManager + 滑动窗口 |
| GQA           | ---                              | n_kv_head {2,4,8}                    |
| MoE FFN       | ---                              | num_experts >= 1, top-1 gating       |
| Pre/Post-Norm | 支持                             | 支持                                 |
| Weight Tying  | wte <-> lm_head                  | wte <-> lm_head                      |
| EMA           | ---                              | 内置 shadow weights                  |

### GQA (Grouped Query Attention) 详解

GQA 的核心思想：多个 Query head 共享同一组 Key/Value head，减少 KV Cache 的显存占用。

```
MHA (n_kv_head=8):           GQA-4KV (n_kv_head=4):        GQA-2KV (n_kv_head=2):
Q: [8 heads]                 Q: [8 heads]                  Q: [8 heads]
K: [8 heads]                 K: [4 heads] x repeat 2       K: [2 heads] x repeat 4
V: [8 heads]                 V: [4 heads] x repeat 2       V: [2 heads] x repeat 4
KV Cache: 18,432 B/tok       KV Cache: 9,216 B/tok (-50%)   KV Cache: 4,608 B/tok (-75%)
参数: 54.0M                  参数: 51.7M                    参数: 50.5M
```

实现方式：`CausalSelfAttention` 中 K/V 投影到 `n_kv_head * head_dim` (窄投影)，reshape 后通过 `repeat_interleave(self.n_rep, dim=1)` 扩展到与 Q 相同头数，再进入 `F.scaled_dot_product_attention`。

---

## KV Cache 原理与项目实现

### 什么是 KV Cache？

自回归生成时，每次预测新 token 都需要对全部历史 token 做 self-attention。不缓存时，第 t 步计算量为 O(t^2)。KV Cache 将每层已计算的 Key/Value 向量缓存下来，第 t+1 步只需计算新 token 的 Q/K/V，再做 O(t) 的 attention。

```
无缓存 (no-cache):              有缓存 (cache):
Step 1: [tok1] -> Q1K1V1          Step 1 (prefill): [tok1...tokN] -> cache K1...KN, V1...VN
Step 2: [tok1,tok2] -> Q12K12     Step 2 (decode):  [tokN+1] -> Q_ + cached K/V -> 只算1次attention
Step 3: [tok1,tok2,tok3] -> ...   Step 3 (decode):  [tokN+2] -> 同上
...                               每个 decode step: O(1) 新计算 + O(t) attention
```

### 本项目中的 KV Cache 实现

1. **`KVCacheManager`** (`kv_cache_utils.py`)：管理逐层 `(past_key, past_value)` 张量对，支持 `init_cache` / `update` / `reset_cache` 操作，内建滑动窗口 (`max_cache_len` 超限自动截断)

2. **`CausalSelfAttention.forward()`** (`modern_gpt.py`)：接收 `past_kv` 和 `use_cache` 参数
   - 若 `past_kv` 非空：将新 K/V concat 到缓存 K/V 上，RoPE 使用 `start_pos + past_len` 计算绝对位置角度
   - 若 `use_cache=True`：返回此行新产生的 K/V 供外部更新缓存

3. **`ModernGPT.generate()`**：cache 路径分 prefill + decode 两阶段
   - **Prefill**：一次性 forward 全部 prompt tokens，缓存全序列 K/V
   - **Decode**：逐 token 循环，只传 `idx[:, -1:]`，复用缓存

4. **RoPE 的绝对位置追踪**：`start_pos` 记录缓存中第一个 token 的绝对位置。当滑动窗口截断时 `start_pos += trim`，后续 RoPE 角度基于 `start_pos + past_len` 计算，保证截断后位置信息不丢失。

### Cache vs No-Cache 何时有收益？

- **短序列 (< 100 tokens)**：no-cache 可能更快 (Python 循环 + kernel launch overhead 超出节省的 FLOPs)
- **长序列 (> 400 tokens)**：cache 显著胜出，因为避免了重复计算全序列 attention
- 完整 54M 模型上的推理消融见 `FULL_EXPERIMENT_LOG.md`

---

## 已实现的完整功能清单

以下列出从 [IMPROVEMENT_CHECKLIST.md](IMPROVEMENT_CHECKLIST.md) 中已落地的主要改进：

### 模型架构层

- [x] **[P0] GQA 完整实现**：`n_kv_head=2` 已配置，`repeat_interleave` 正确执行，KV Cache 节省 75%
- [x] **[P1] BaselineGPT SDPA 后端**：`attention_backend="sdpa"|"manual"` 可切换
- [x] **[P1] Pre/Post-Norm 消融开关**：`norm_position="pre"|"post"`
- [x] **[P2] SwiGLU multiple-of 对齐**：`intermediate_size` 向上取整至 128 的倍数 (1408)
- [x] **[P3] MoE FFN 支持**：`num_experts` >= 1, top-1 gating

### 训练系统层

- [x] **[P0] 梯度累积**：`--gradient_accumulation_steps` 参数
- [x] **[P1] LR Scheduler 统一**：cosine / linear / wsd / constant 四种模式
- [x] **[P1] EMA**：`--use_ema` + `init_ema` / `update_ema` / `apply_ema_weights`
- [x] **[P1] SFT LR Scheduler**：已集成
- [x] **[P2] GradScaler**：float16 模式下自动启用
- [x] **[P2] Early Stopping**：`--early_stopping_patience`
- [x] **[P3] FSDP**：`--fsdp` + `--fsdp_sharding_strategy`
- [x] **[P0] 完整训练状态 checkpoint**：RNG + scaler + scheduler + EMA + resume_offset，支持断点精确续训

### 推理系统层

- [x] **[P1] CUDA Events 计时**：prefill/decode 阶段分离
- [x] **[P1] RoPE 缓存**：`_cos_cached` / `_sin_cached` 惰性计算 + 复用
- [x] **[P2] KVCacheManager 集成**：generate() cache 路径使用 KVCacheManager
- [x] **[P2] 生成策略扩展**：top_p (nucleus) + repetition_penalty
- [x] **[P3] Batch 推理**：eos_token_id + finished mask 实现逐序列提前终止，cache/no-cache 双路径支持

### 数据管道层

- [x] **[P2] 数据质量验证**：`data/validate.py`
- [x] **[P1] 算术数据多样化模板**：5种 hard 模板 (嵌套括号/优先级/多组/指数取模/两组乘法)
- [x] **[P1] 文档边界数据集可运行**：`DocBoundaryDataset` 支持 `resume_offset`

### 质量保障

- [x] **[P0] 回归测试**：`tests/test_bugfixes.py` 覆盖 KV Cache dtype、优化器去重、EMA、checkpoint  round-trip、DocBoundaryDataset

---

## 预训练架构改造详解

从老式 GPT-2 升级到现代架构的具体改动：

### 1. LayerNorm → RMSNorm (`model/modern_gpt.py`)

```python
# LayerNorm: 减均值 + 除标准差 + 仿射变换
y = (x - mean) / std * gamma + beta  # 2个可学习参数

# RMSNorm: 仅除 RMS + 缩放 (无 bias, 无减均值)
rms = sqrt(mean(x^2) + eps)
y = x / rms * weight  # 仅1个可学习参数
```

**收益**：计算量减少约15%, 参数减少 (去掉了 bias), LLaMA 系列的标准选择。

### 2. GELU FFN → SwiGLU FFN (`model/modern_gpt.py`)

```python
# GELU FFN: x -> Linear(4d) -> GELU -> Linear(d)
# 参数量: 2 * d * 4d = 8d^2

# SwiGLU: x -> gate(x)*up(x) -> down
# gate = Linear(d -> hidden), up = Linear(d -> hidden), down = Linear(hidden -> d)
# hidden = 8d/3 (向上取整至128倍数), 参数量: 3 * d * hidden
```

**收益**：SwiGLU 在相同计算量下提供更好的训练动态和收敛效果，LLaMA 等模型验证。

### 3. 绝对位置编码 → RoPE 旋转位置编码 (`model/modern_gpt.py`)

```python
# 绝对位置: lookup embedding[wpe] (固定长度, 无法外推)
pos_emb = wpe[positions]

# RoPE: 通过旋转矩阵注入位置信息 (相对位置天然编码, 可外推)
q_rot = q * cos(pos) + rotate_half(q) * sin(pos)
k_rot = k * cos(pos) + rotate_half(k) * sin(pos)
```

**收益**：

- 相对位置关系天然编码在 attention score 中
- 支持长序列外推 (训练1024长度可推理更长)
- 配合 KV Cache 时通过 `start_pos` 追踪绝对位置

配置参数：`rotary_base=10000`, `max_seq_len=block_size * 2`

---

## 实验指标目标

| 阶段    | 指标                                      | 目标                                    |
| ------- | ----------------------------------------- | --------------------------------------- |
| 预训练  | ModernGPT val loss vs Baseline @ 18k iter | **-2.29%** (3.9126 -> 3.8229)           |
| GQA     | KV Cache 显存 vs MHA                      | **-50%** (GQA-4KV) / **-75%** (GQA-2KV) |
| 推理    | KV Cache 吞吐 vs no-cache                 | 长序列 (>400 tokens) 正向提升           |
| SFT     | Format pass rate                          | easy 100%, medium >= 96%                |
| GRPO-G4 | Accuracy vs SFT-only                      | easy +19.1 pts, medium +2.8 pts         |
| GRPO-G4 | Format pass rate                          | **100%**, invalid rate **0%**           |

---

## 复现清单

- [ ] 数据准备: `python data/prepare.py --split train` + `data/validate.py`
- [ ] 预训练: `BaselineGPT` + `ModernGPT` (n_kv_head=8/4/2)
- [ ] 推理消融: cache vs no-cache, 多种长度, CUDA Events
- [ ] SFT: `sft-only` + `sft-continued`
- [ ] GRPO: G4 (1000 steps) + G8 (250 steps)
- [ ] 评估: `eval_alignment.py` 全维度
- [ ] 消融: GQA (n_kv_head=8/4/2), Pre/Post-Norm, LR schedule (cosine/wsd), KL (beta=0 vs 0.04)
- [ ] 回归测试: `python tests/test_bugfixes.py`

> 固定种子 `1337`

---

## 关键命令速查

```bash
# ====== 预训练 ======
# ModernGPT + GQA-2KV
python training/train_pretrain.py --model modern --n_kv_head 2 --use_ema --use_wandb

# BaselineGPT (对照)
python training/train_pretrain.py --model baseline

# Gradient accumulation (effective batch = 48)
python training/train_pretrain.py --model modern --gradient_accumulation_steps 4

# FSDP (多GPU)
torchrun --nproc_per_node=4 training/train_pretrain.py --model modern --fsdp

# 断点续训
python training/train_pretrain.py --model modern --resume out/pretrain/latest_ckpt.pt

# ====== SFT ======
python training/train_sft.py --init_from out/pretrain/best_ckpt.pt --out_dir out/sft

# ====== GRPO ======
python training/train_grpo.py --init_from out/sft/best.pt --ref_from out/sft/best.pt --group_size 4 --num_steps 1000 --beta 0.04

# ====== 推理消融 ======
python inference/generate.py --checkpoint out/pretrain/best_ckpt.pt --max_new_tokens 200 400 600 --num_samples 30 --output_json out/ablation.json

# ====== 评估 ======
python evaluation/eval_alignment.py --checkpoint out/grpo/best.pt --ref_checkpoint out/sft/best.pt

# ====== 系统消融 ======
python run_inference_ablation.py    # KV Cache 消融
python run_full_evaluation.py       # 全维度评估
python generate_experiment_log.py   # 生成实验日志

# ====== 质量保障 ======
python tests/test_bugfixes.py       # 关键 Bug 回归测试
```

---

## ModernGPTConfig 完整参数

```python
ModernGPTConfig(
    vocab_size=50257,         # 词表大小 (GPT-2 tokenizer)
    block_size=1024,          # 最大上下文长度
    n_layer=12,               # Transformer 层数
    n_head=8,                 # Query 注意力头数
    n_embd=512,               # 隐藏维度 (head_dim = 512/8 = 64)
    n_kv_head=None,           # KV 头数 (None=n_head 即 MHA; 设为 2/4 启用 GQA)
    intermediate_size=None,   # SwiGLU 隐层维度 (None=自动: 8d/3 向上取整到128倍数 -> 1408)
    dropout=0.0,              # Dropout rate
    norm_position="pre",      # Pre-Norm / Post-Norm
    num_experts=1,            # MoE experts (1=密集 SwiGLU, >1=top-1 gating)
)
```

### 参数量精确对齐

| 配置 (12L/512D)        | 总参数量 | 非嵌入参数 | KV Cache (B/tok) |
| ---------------------- | -------- | ---------- | ---------------- |
| BaselineGPT            | 54.6M    | 28.4M      | N/A              |
| ModernGPT MHA (n_kv=8) | 54.0M    | 28.3M      | 18,432           |
| ModernGPT GQA-4KV      | 51.7M    | 26.0M      | **9,216** (-50%) |
| ModernGPT GQA-2KV      | 50.5M    | 24.8M      | **4,608** (-75%) |

---

## 项目结构

```
nanogpt-modern/
├── config/                     # YAML 配置
│   ├── pretrain.yaml, sft.yaml, grpo.yaml, generate.yaml
├── data/
│   ├── prepare.py              # 下载 -> tokenize -> 二进制
│   ├── openwebtext.py          # MemmapDataset + DocBoundaryDataset
│   ├── arithmetic.py           # 三档算术生成 + 多样化模板 + safe_eval
│   └── validate.py             # 数据质量验证
├── model/
│   ├── baseline_gpt.py         # GPT-2 (LayerNorm/GELU/AbsPos + SDPA/manual + Pre/Post-Norm)
│   ├── modern_gpt.py           # ModernGPT (RMSNorm/SwiGLU/RoPE/GQA/MoE/EMA + Pre/Post-Norm)
│   └── kv_cache_utils.py       # KVCacheManager + 滑动窗口
├── training/
│   ├── train_pretrain.py       # AMP+GradScaler+梯度累积+LR Scheduler+EMA+EarlyStop+DDP/FSDP
│   ├── train_sft.py            # SFT + LR Scheduler
│   ├── train_grpo.py           # GRPO (PPO Clip + KL + dropout guard)
│   └── iterative_grpo.py       # 迭代 RLHF + Rejection Sampling SFT
├── inference/
│   └── generate.py             # CUDA Events benchmark + prefill/decode 分离
├── evaluation/
│   └── eval_alignment.py       # accuracy/reward/format/invalid/KL
├── rewards/
│   └── rule_reward.py          # 规则奖励 (格式+正确性)
├── utils/
│   ├── lr_scheduler.py         # 统一 LR Scheduler (cosine/linear/wsd/constant)
│   ├── logging.py              # wandb/TensorBoard/console
│   └── checkpoint.py           # 完整训练状态序列化
├── tests/
│   ├── __init__.py
│   └── test_bugfixes.py        # 关键 Bug 回归测试
├── out/                        # 训练产出 (checkpoints + logs)
├── run_inference_ablation.py    # 推理消融自动化脚本
├── run_full_evaluation.py       # 全维度评估自动化脚本
├── generate_experiment_log.py   # 实验日志生成
├── IMPROVEMENT_CHECKLIST.md     # 系统改进清单 (30+项)
├── IMPROVEMENT_REPORT.md        # 深度诊断与优化报告
├── REPRODUCTION_REPORT.md       # 完整复现报告
├── FULL_EXPERIMENT_LOG.md       # 详细实验日志
├── requirements.txt
└── README.md
```

---

## 设计决策 FAQ

**为什么用 GRPO 而非 PPO/DPO？**
GRPO 不需要 Value/Critic 网络，在轻量级模型上显著降低实现复杂度与显存开销。通过组内相对奖励计算优势，天然适合规则奖励函数场景。

**SwiGLU 的 intermediate_size 为什么取 128 的倍数？**
`8d/3` 向上取整到 128 的倍数（如 512->1408）提升 GPU 内存对齐。参数量仍与 Baseline GELU FFN 的 `8d^2` 在合理范围内对齐。

**KV Cache 如何保证与 no-cache 的数值一致？**
`start_pos` 追踪 KV cache 中第一个 token 的绝对位置。截断时 `start_pos += trim`，后续 RoPE 角度基于真实索引计算，而非缓存物理长度。

**为什么 old_logprobs 必须在 eval 模式下采样？**
需要 dropout=0 保证 old/new logprobs 的 ratio 无偏。`GRPOTrainer` 在 `dropout > 0` 时发出警告。

**为什么规则奖励不用神经网络 Critic？**
算术任务正确性是离散且可验证的。规则奖励函数零延迟、100% 确定、无 approximation error，是此类结构化任务的最优选择。

**GQA 的 KV head 数量如何选择？**
`n_kv_head` 是 `n_head` 的约数。推荐 `n_head / n_kv_head = 2 或 4`：

- 比例越大，KV Cache 节省越多，但表达能力下降
- 本项目中 `n_head=8`，推荐 `n_kv_head=2` (4x 压缩) 或 `n_kv_head=4` (2x 压缩)

**Checkpoint 为什么保存 RNG 和 EMA 状态？**
为了严格复现训练过程。断点续训时恢复 RNG 可保证数据顺序与 dropout mask 一致；恢复 EMA 可避免 shadow weights 从头累计，保证评估稳定性。

---

## 参考文献

- [nanoGPT](https://github.com/karpathy/nanoGPT) --- Andrej Karpathy
- [RMSNorm] Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
- [SwiGLU] GLU Variants Improve Transformer (Shazeer, 2020)
- [RoPE] RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
- [GQA] GQA: Training Generalized Multi-Query Transformer Models (Ainslie et al., 2023)
- [GRPO] Group Relative Policy Optimization (DeepSeekMath, Shao et al., 2024)
- [LLaMA] LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)

---

## 许可

基于 nanoGPT 思想构建，仅供研究与学习使用。
