# nanoGPT-Modern

一个**端到端的轻量级大语言模型训练-推理-对齐全栈框架**，基于 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT) 思想构建，在 **~50M 参数** 规模下完整验证现代 Transformer 组件的架构增益与效率 trade-off。

> :clipboard: 完整的系统改进清单见 [IMPROVEMENT_CHECKLIST.md](IMPROVEMENT_CHECKLIST.md)  
> :bar_chart: 深度诊断与优化报告见 [IMPROVEMENT_REPORT.md](IMPROVEMENT_REPORT.md)  
> :white_check_mark: 关键阻塞 Bug 已修复；RL 管线（GRPO / Iterative GRPO）已完成批量化、AMP、梯度累积、LR 调度改造；`pretrain/sft/grpo` 已统一继承 `BaseTrainer`；配置/日志/依赖/可安装化已补齐；SDPA attention 后端可显式选择；checkpoint 生命周期、种子管理、GRPO dropout  guard 已落地；新增 benchmark 评估与消融自动化脚本；DataLoader 缓冲乱序、文档打包 (`PackingDataset`) 与跨文档 mask、`generate()` `torch.compile` 推理加速已完成；**GQA grouped-broadcast 零拷贝**（M2 完成）与 **FlashAttention varlen/kvcache 显式集成**（M7 完成）已落地；回归测试覆盖 `tests/` 全目录（201 passed, 2 skipped），核心模块通过 `mypy` 类型检查。

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
2. **现代组件全集成**：RMSNorm、SwiGLU、RoPE、GQA、可选第三方 `flash-attn`、QK-Norm、Attention Temperature、NTK-aware 长度外推、Sliding Window Attention、带负载均衡的 MoE、Paged KV Cache、MTP（Multi-Token Prediction）、KV Cache、EMA 全部可配置。
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

# 2b. 启用数据质量管道：长度过滤 + 重复过滤 + MinHash 去重 + 多源混合
python data/prepare.py --split train \
  --min_doc_chars 50 --max_doc_chars 50_000 \
  --max_repetition_ratio 0.3 \
  --dedup_threshold 0.85 --dedup_num_hashes 128

# 混合多源数据示例（mixture.json 见文档下方）
# python data/prepare.py --split train --mixture_config mixture.json

# 3. 快速验证预训练 (1000步, ~5分钟)
python training/train_pretrain.py --config config/pretrain.yaml --max_iters 1000 --n_kv_head 2

# 4. 推理
python inference/generate.py --config config/generate.yaml --checkpoint out/pretrain/best_ckpt.pt --max_new_tokens 200

# 5. 完整三阶段流水线（YAML 配置 + 命令行覆盖）
python training/train_pretrain.py --config config/pretrain.yaml --use_ema --keep_last_n 3
python training/train_sft.py --config config/sft.yaml --init_from out/pretrain/best_ckpt.pt
python training/train_grpo.py --config config/grpo.yaml --init_from out/sft/best_sft-only.pt --ref_from out/sft/best_sft-only.pt
python evaluation/eval_alignment.py --checkpoint out/grpo/best_grpo_g4.pt --ref_checkpoint out/sft/best_sft-only.pt

# 6. 预训练 Benchmark 评估
python evaluation/eval_benchmark.py --checkpoint out/pretrain/best_ckpt.pt \
    --data_dir data/openwebtext --split val --tasks hellaswag,lambada_openai

# 6.1 Hydra / OmegaConf 入口（M23，与上面 argparse 入口等价）
python training/train_pretrain_hydra.py
python training/train_pretrain_hydra.py batch_size=16 n_layer=2
python training/train_sft_hydra.py init_from=out/pretrain/best_ckpt.pt
python training/train_grpo_hydra.py init_from=out/sft/best_sft-only.pt ref_from=out/sft/best_sft-only.pt
python inference/generate_hydra.py checkpoint=out/pretrain/best_ckpt.pt prompt="Hello"
python evaluation/eval_benchmark_hydra.py checkpoint=out/pretrain/best_ckpt.pt

# 7. 消融实验自动化
python run_ablations.py --mode inference --checkpoint out/pretrain/best_ckpt.pt

# 8. 运行回归测试
python -m pytest tests/ -q
```

---

## 近期关键优化

- **类型安全**：核心模块（`model/modern_gpt.py`、`utils/rl_utils.py`）已添加类型注解；`tests/test_mypy.py` 在 pytest 中调用 `mypy` 防止回退。
- **MoE 负载均衡**：`num_experts > 1` 时自动启用 load-balancing aux loss 与 per-expert 容量限制，可通过 `return_aux_loss=True` 获取并加入训练 loss。
- **FlashAttention 可选后端**：`model/flash_attention.py` 封装 `flash_attn_func` / `flash_attn_varlen_func` / `flash_attn_with_kvcache` 三接口；`use_flash_attn=True` 时优先调用 flash kernel，不可用时自动回退 SDPA/eager；GQA 原生支持无需 repeat_interleave。
- **GQA grouped-broadcast 零拷贝**：`CausalSelfAttention` 在 GQA 模式下将 Q reshape 为 `[B, n_kv, n_rep, T, hd]`、KV unsqueeze 为 `[B, n_kv, 1, S, hd]`，SDPA 自动广播 singleton 维度，消除 `repeat_interleave` 的 KV 临时膨胀，完全保留 GQA 显存收益。`probe_gqa_sdpa_support()` 懒探测设备支持，`gqa_broadcast` 配置项支持 auto/grouped/raw/repeat 四种策略。
- **QK-Norm / Attention Temperature / NTK 外推 / Sliding Window**：`qk_norm`、`attn_temperature`、`rope_scaling`、`sliding_window_size` 已接入 `ModernGPTConfig`。
- **Paged KV Cache**：新增 `model/paged_kv_cache.py`，`use_paged_kv_cache=True` 时 `generate()` 使用 block-table 布局，未来可对接 PagedAttention kernel。
- **Multi-Token Prediction (MTP)**：`n_future > 0` 时增加未来 token 预测头，训练 loss 自动叠加 MTP loss。
- **DPO / IPO / KTO**：新增 `utils/dpo_utils.py` 与 `training/train_dpo.py`，支持三种偏好对齐损失的统一训练入口。
- **类型注解扩展**：`model/modern_gpt.py`、`model/paged_kv_cache.py`、`utils/rl_utils.py`、`utils/dpo_utils.py`、`inference/generate_utils.py` 通过 `mypy` 检查。
- **`generate()` fullgraph 编译**：`ModernGPT.generate(..., compile="fullgraph")` 对单 token decode 步骤做 `torch.compile(..., fullgraph=True, dynamic=True)`；提前终止用 mask 操作替代，失败自动回退 eager。
- **Speculative Decoding**：`ModernGPT.generate(..., draft_model=...)` 支持 draft-then-verify 投机采样，batch size 1 场景下可用小模型加速目标模型推理。
- **HuggingFace 兼容层**：新增 `model/hf_model.py`、`export_to_hf.py`、`load_from_hf.py`；`NanoGPTModernConfig` / `NanoGPTModernForCausalLM` 包装原生 ``ModernGPT``，支持 HF 格式保存与加载；`RMSNorm` 改用 `F.rms_norm` 消除与 fused 层的共享权重，兼容 safetensors 序列化。
- **模型量化与 GGUF 导出**：新增 `model/quantization.py` 提供静态 per-channel INT8 量化与可选 `bitsandbytes` 8-bit/4-bit 后端；新增 `model/gguf_utils.py` 与 `export_gguf.py`，无需外部 `gguf` 包即可导出 `f32/f16/q8_0` 的 GGUF 文件。
- **数据质量管道**：新增 `data/filter.py`（长度/重复/正则/可选 fasttext 过滤）、`data/dedup.py`（MinHash + LSH 近重复检测）、`data/mixer.py`（多源数据按权重混合）；`data/prepare.py` 集成过滤/去重/混合参数，支持命令行一键启用。

---

## 系统全景：各组件职能与协作关系

本项目围绕 "预训练 → 监督微调 → RL 对齐" 三阶段流水线组织，每层由独立模块构成，通过配置对象串接。

### 1. 数据管道层 --- 生产训练数据

| 模块 | 职能 |
|------|------|
| `data/prepare.py` | 下载 OpenWebText → `datasets.map(batched=True, num_proc=...)` 多进程 tokenize → 固定大小 binary shards + 全局 `.bin` + `.idx` 元数据（含 dtype/vocab_size/`doc_boundary`/`eot_token`）；streaming 模式可选 |
| `data/openwebtext.py` | `MemmapDataset` 流式读取 + `shuffle_buffer` 缓冲乱序；`DocBoundaryDataset` 按 EOT 文档边界切分，支持断点续训 offset；`PackingDataset` 多文档打包并产出 `document_ids`；`.idx` 自动识别 uint16/uint32 |
| `data/arithmetic.py` | 合成算术数据集生成器：`easy` / `medium` / `hard`；`ArithmeticDataset` 在 `__init__` 中一次性预编码全部样本，避免 DataLoader worker 重复 tokenize |
| `data/validate.py` | 数据质检：token 范围检查、词表覆盖率、EOT 频率统计、随机 decode 采样 |
| `data/filter.py` | 文档级质量过滤：长度、字符 n-gram 重复、正则、可选 fasttext 语言/质量分类 |
| `data/dedup.py` | MinHash + LSH 近重复检测，支持批量 `MinHashDeduplicator` 与流式 `StreamingDuplicateDetector` |
| `data/mixer.py` | 多源数据集按权重混合，支持温度缩放与 `MixedIterableDataset` |

**数据质量管道**：`data/prepare.py` 在 tokenize 之前可依次执行过滤、去重、混合：
- 过滤：`--min_doc_chars`、`--max_doc_chars`、`--max_repetition_ratio`、`--require_regex`、`--reject_regex`
- 去重：`--dedup_threshold`（Jaccard 阈值）、`--dedup_ngram`、`--dedup_num_hashes`、`--dedup_num_bands`、`--dedup_rows_per_band`
- 混合：`--mixture_config` 指向 JSON，例如
  ```json
  {
    "datasets": {
      "openwebtext": {"name": "openwebtext", "split": "train"},
      "skylion": {"name": "Skylion007/openwebtext", "split": "train", "trust_remote_code": true}
    },
    "weights": {"openwebtext": 0.7, "skylion": 0.3},
    "temperature": 0.8,
    "seed": 42
  }
  ```

### 2. 模型层 --- 双轨制架构对比

| 模块 | 职能 |
|------|------|
| `model/baseline_gpt.py` | GPT-2 经典架构：LayerNorm + GELU FFN + 绝对位置编码。支持 SDPA/manual 双后端切换、Pre/Post-Norm |
| `model/modern_gpt.py` | ModernGPT：RMSNorm + SwiGLU + RoPE + GQA + 可选 flash-attn + QK-Norm + Attention Temperature + NTK 外推 + Sliding Window + MoE(带负载均衡) + MTP + EMA。支持 KV Cache / Paged KV Cache 原生推理 |
| `model/kv_cache_utils.py` | `KVCacheManager`：管理逐层 past KV 张量，支持滑动窗口截断、批量重置、GQA 适配 |
| `model/paged_kv_cache.py` | `PagedKVCacheManager`：block-table 布局 KV Cache，API 兼容 `KVCacheManager`，为 PagedAttention kernel 预留扩展点 |

**两模型的关系**：共享相同的训练超参、数据顺序、随机种子，确保对比实验中 **唯一变量是架构差异**。

### 3. 训练系统层 --- 三阶段流水线

| 模块 | 职能 | 核心特性 |
|------|------|----------|
| `training/trainer_base.py` | 统一训练抽象 | `BaseTrainer` + `CheckpointManager`：分布式初始化/销毁、种子管理、AMP/GradScaler、logger、checkpoint 保存/恢复（EMA/scaler/scheduler/RNG/resume_offset）、FSDP/DDP 包装 |
| `training/train_pretrain.py` | 语言建模预训练 (OpenWebText) | 继承 `BaseTrainer`；AMP + GradScaler + 梯度累积 + LR Scheduler (cosine/linear/wsd/constant) + EMA + EarlyStopping + DDP/FSDP + 完整训练状态 checkpoint |
| `training/train_sft.py` | 监督微调 (算术数据) | 继承 `BaseTrainer`；AMP + 梯度累积 + LR Scheduler + 数据混叠 (easy+medium+hard) + DDP |
| `training/train_grpo.py` | GRPO 对齐 (算术任务) | 继承 `BaseTrainer`；**Batched rollout + batched logprobs** + AMP (bf16/fp16) + 梯度累积 + LR Scheduler (cosine/linear/wsd/constant) + PPO Clip + KL Penalty + Old/New Policy 分离 + Dropout Guard |
| `training/iterative_grpo.py` | 迭代 RLHF | 继承优化后的 `GRPOTrainer`，支持周期性 ref-model EMA 更新 + 批量 Rejection Sampling SFT |

**三阶段关系**：预训练 → SFT → GRPO 是串行依赖链。预训练产出基座模型，SFT 在算术数据上注入任务格式，GRPO 通过规则奖励进一步优化正确率。

### 4. 推理系统层 --- 生成与性能测量

| 模块 | 职能 |
|------|------|
| `inference/generate.py` | CUDA Events 精确计时 + prefill/decode 阶段分离 + cache/no-cache 消融对比；命令行 `--temperature/--top_k/--top_p` 正确传入采样；新增 `--compile` 使用 `torch.compile` 降低 token-by-token decode 的 kernel launch overhead |

**推理管线**：`ModernGPT.generate()` 支持两种路径：
- **No-cache 路径**：每次生成一个 token 都完整 forward 全序列 (O(T^2) 复杂度)
- **Cache 路径**：prefill 阶段一次编码全 prompt → decode 阶段逐 token forward 仅新 token，复用 cached KV (O(T) 复杂度)
- **编译加速**：`ModernGPT.generate(..., compile=True)` 或 `python inference/generate.py --compile` 在 CUDA 上通过 `torch.compile` 编译 forward，失败自动回退 eager；`compile="fullgraph"` 进一步对单 token decode 步骤做 `fullgraph=True` 编译，提前终止用 mask 操作替代
- **投机采样**：`ModernGPT.generate(..., draft_model=...)` 支持 draft-then-verify 投机解码，可用小 draft 模型加速大目标模型生成

```python
# 使用与目标模型同架构的较小模型作为 draft 模型
draft_config = ModernGPTConfig(n_layer=2, n_head=4, n_embd=256, block_size=1024)
draft_model = ModernGPT(draft_config).cuda().eval()

out = target_model.generate(
    idx,
    max_new_tokens=200,
    use_cache=True,
    draft_model=draft_model,
    draft_tokens=4,
    draft_temperature=0.0,   # greedy draft 最稳定
)
```

**HuggingFace 格式导出/加载**：

```bash
# nanoGPT-Modern -> HuggingFace
python export_to_hf.py --checkpoint out/pretrain/best_ckpt.pt --out_dir hf/nanogpt-modern

# HuggingFace -> nanoGPT-Modern
python load_from_hf.py --hf_dir hf/nanogpt-modern --out out/pretrain/best_ckpt_roundtrip.pt
```

```python
from model.hf_model import NanoGPTModernForCausalLM

wrapper = NanoGPTModernForCausalLM.from_pretrained("hf/nanogpt-modern")
out = wrapper.generate(idx, max_new_tokens=100, use_cache=True, top_k=1)
```

**模型量化与 GGUF 导出**：

```bash
# 静态 per-channel INT8 量化（纯 PyTorch，CPU/GPU 通用）
python - <<'PY'
import torch
from model.modern_gpt import ModernGPT, ModernGPTConfig
from model.quantization import QuantConfig, quantize_model
model = ModernGPT(ModernGPTConfig()).eval()
quantize_model(model, QuantConfig(method="int8", compute_dtype=torch.float16))
PY

# 导出为 GGUF（支持 f32 / f16 / q8_0）
python export_gguf.py --checkpoint out/pretrain/best_ckpt.pt --out out/pretrain/best_ckpt.q8_0.gguf --quant q8_0
```


### 5. 评估与奖励系统层

| 模块 | 职能 |
|------|------|
| `evaluation/eval_alignment.py` | 全维度评估：按 prompt 长度分组批量生成，KL 仅对 response token 计算，训练与评估统一使用反向 KL `ref_logp - policy_logp` 并做数值裁剪；输出 accuracy / format_pass_rate / process_score / invalid_rate / reward / KL_divergence |
| `rewards/rule_reward.py` | 规则奖励函数：格式分 + 过程奖励（中间推导步骤）+ 连续正确性分（相对误差多级 partial credit）；拒绝 nan/inf/overflow，检查标签闭合与模板一致性 |

### 6. 工具与基础设施层

| 模块 | 职能 |
|------|------|
| `config/` | YAML 配置文件 (pretrain.yaml, sft.yaml, grpo.yaml, generate.yaml)，统一管理超参 |
| `utils/config.py` | 统一配置加载：YAML + argparse CLI 覆盖，支持嵌套配置（`--optimizer.lr`）、环境变量展开、`to_dict`/`validate_keys` |
| `utils/lr_scheduler.py` | 统一 LR Scheduler：cosine / linear / wsd / constant 四种模式 |
| `utils/logging.py` | 日志系统：wandb + TensorBoard + console 三后端，支持 auto-degrade fallback；新增文本样例、梯度 norm / 显存直方图、日志等级控制 |
| `utils/checkpoint.py` | Checkpoint 持久化：模型 + 优化器 + iter + config + RNG + scaler + scheduler + EMA + resume_offset，支持 FSDP Full State Dict |

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

| 维度 | BaselineGPT | ModernGPT |
|------|-------------|-----------|
| 归一化 | LayerNorm (含 bias) | **RMSNorm** (无 bias) |
| 前馈网络 | GELU (4x 扩展, 8d^2 参数) | **SwiGLU** (gate/up/down, 3d*hidden) |
| 位置编码 | 可学习绝对位置 Embedding | **RoPE** 旋转位置编码 |
| Attention | SDPA (默认) / manual causal mask | SDPA (FlashAttention 自动分发) |
| KV Cache | --- | 原生支持 + KVCacheManager + 滑动窗口 |
| GQA | --- | n_kv_head {2,4,8} |
| MoE FFN | --- | num_experts >= 1, top-1 gating |
| Pre/Post-Norm | 支持 | 支持 |
| Weight Tying | wte <-> lm_head | wte <-> lm_head |
| EMA | --- | 内置 shadow weights |

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

实现方式：`CausalSelfAttention` 中 K/V 投影到 `n_kv_head * head_dim`（窄投影），通过 **grouped-broadcast** 策略将 Q reshape 为 `[B, n_kv_head, n_rep, T, hd]`、KV unsqueeze 为 `[B, n_kv_head, 1, S, hd]`，SDPA 自动广播 singleton 维度，**零 KV 拷贝**，保留 GQA 显存节省。运行时通过 `probe_gqa_sdpa_support()` 懒探测当前设备/ dtype 的支持情况，可通过 `gqa_broadcast` 配置项（`"auto"`/`"grouped"`/`"raw"`/`"repeat"`）强制指定策略。

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

1. **`KVCacheManager`** (`kv_cache_utils.py`)：预分配静态环形缓存 `[B, n_kv_heads, max_cache_len, head_dim]`，用位置指针写入新 K/V，避免 decode 阶段反复 `torch.cat`。内建滑动窗口，超限时按先进先出淘汰旧 token，并同步更新 `start_pos` 保证 RoPE 绝对位置连续。支持多段缓存返回，便于 ring-buffer 拼接。

2. **`CausalSelfAttention.forward()`** (`modern_gpt.py`)：接收 `past_kv` 和 `use_cache` 参数
   - 若 `past_kv` 非空：将新 K/V 与缓存 K/V 拼接，RoPE 使用 `start_pos + past_len` 计算绝对位置角度
   - 兼容单段 tuple 与多段 list 两种缓存格式
   - 若 `use_cache=True`：返回当前输入对应的新 K/V 供外部更新缓存

3. **`ModernGPT.generate()`**：cache 路径分 prefill + decode 两阶段
   - **Prefill**：一次性 forward 全部 prompt tokens，缓存全序列 K/V
   - **Decode**：逐 token 循环，只传 `idx[:, -1:]`，复用缓存
   - **滑动窗口一致性**：cache 路径与 no-cache 路径均使用绝对 `start_pos`，并在每次 advance 后保留 `max_cache_len - 1` 个 token，确保与 no-cache 裁剪到最近 `max_cache_len` 个 token 的行为数值一致

4. **RoPE 的绝对位置追踪**：`start_pos` 记录缓存中第一个 token 的绝对位置。当滑动窗口淘汰旧 token 时 `start_pos += evicted`，后续 RoPE 角度基于 `start_pos + past_len` 计算，保证截断后位置信息不丢失。

### Cache vs No-Cache 何时有收益？

- **短序列 (< 100 tokens)**：no-cache 可能更快 (Python 循环 + kernel launch overhead 超出节省的 FLOPs)
- **长序列 (> 400 tokens)**：cache 显著胜出，因为避免了重复计算全序列 attention
- 完整 54M 模型上的推理消融见 `FULL_EXPERIMENT_LOG.md`

---

## 已实现的完整功能清单

以下列出从 [IMPROVEMENT_CHECKLIST.md](IMPROVEMENT_CHECKLIST.md) 中已落地的主要改进：

### 模型架构层
- [x] **[P0] GQA 完整实现**：`n_kv_head=2` 已配置，grouped-broadcast 零拷贝替代 `repeat_interleave`，KV Cache 节省 75%，`gqa_broadcast` 配置项支持 auto/grouped/raw/repeat
- [x] **[P1] BaselineGPT SDPA 后端**：`attention_backend="sdpa"|"manual"` 可切换
- [x] **[P1] Pre/Post-Norm 消融开关**：`norm_position="pre"|"post"`
- [x] **[P2] SwiGLU multiple-of 对齐**：`intermediate_size` 向上取整至 128 的倍数 (1408)
- [x] **[P3] MoE FFN 支持**：`num_experts` >= 1, top-1 gating

### 训练系统层
- [x] **[P2] DataLoader shuffle 与文档边界**：`MemmapDataset.shuffle_buffer` 缓冲乱序 + `DocBoundaryDataset` EOT 截断
- [x] **[P0] 梯度累积**：`--gradient_accumulation_steps` 参数，预训练/SFT/GRPO 均已支持
- [x] **[P1] LR Scheduler 统一**：cosine / linear / wsd / constant 四种模式 + warmup
- [x] **[P1] EMA**：`--use_ema` + `init_ema` / `update_ema` / `apply_ema_weights`
- [x] **[P1] SFT LR Scheduler**：已集成
- [x] **[P1] GRPO LR Scheduler**：已集成，支持 warmup + decay
- [x] **[P1] GRPO 批量化**：group 内 rollout 与 logprob 均 batch 化，避免 Python 逐条循环
- [x] **[P2] AMP (bf16/fp16 + GradScaler)**：预训练/SFT/GRPO/Iterative GRPO 均已启用
- [x] **[P2] Early Stopping**：`--early_stopping_patience`
- [x] **[P3] FSDP**：`--fsdp` + `--fsdp_sharding_strategy`
- [x] **[P0] 完整训练状态 checkpoint**：RNG + scaler + scheduler + EMA + resume_offset，支持断点精确续训
- [x] **[P2] Checkpoint 生命周期管理**：`--keep_last_n` 自动清理旧 checkpoint
- [x] **[P2] 种子管理健壮性**：DDP/FSDP 模型初始化一致 + DataLoader worker 确定性种子
- [x] **[P2] SDPA attention 后端显式选择**：`--attn_backend auto/flash/mem_efficient/math/default`

### 推理系统层
- [x] **[P1] CUDA Events 计时**：prefill/decode 阶段分离
- [x] **[P1] RoPE 缓存**：`_cos_cached` / `_sin_cached` 惰性计算 + 复用
- [x] **[P2] 静态 KV Cache**：`KVCacheManager` 预分配环形缓存，消除长序列每步 `torch.cat` 开销
- [x] **[P2] KVCacheManager 集成**：generate() cache 路径使用 KVCacheManager
- [x] **[P2] 生成策略扩展**：top_p (nucleus) + repetition_penalty
- [x] **[P3] Batch 推理**：eos_token_id + finished mask 实现逐序列提前终止，cache/no-cache 双路径支持
- [x] **[P1] 生成循环 torch.compile**：`ModernGPT.generate(compile=True)` + `inference/generate.py --compile`，CUDA 上编译 forward 降低 kernel launch overhead，失败自动回退 eager；`compile="fullgraph"` 对单 token decode 步骤做 `fullgraph=True` 编译，mask 替代 break
- [x] **[P2] Speculative Decoding**：`ModernGPT.generate(..., draft_model=...)` 支持 draft-then-verify 投机采样，自 draft 模型 greedy 输出与目标 greedy 完全一致
- [x] **[P3] HuggingFace 兼容层**：`model/hf_model.py` 提供 `NanoGPTModernConfig` / `NanoGPTModernForCausalLM`；`export_to_hf.py` / `load_from_hf.py` 实现 nanoGPT ↔ HF 格式互转；`RMSNorm` 改用 `F.rms_norm` 以兼容 safetensors
- [x] **[P3] 模型量化与 GGUF 导出**：`model/quantization.py` 提供 INT8 静态量化与可选 `bitsandbytes` 8-bit/4-bit；`model/gguf_utils.py` 内置 F32/F16/Q8_0 GGUF writer；`export_gguf.py` 一键导出；`tests/test_quantization.py`、`tests/test_gguf_export.py` 覆盖量化误差与 GGUF 往返

### 数据管道层
- [x] **[P3] 数据质量管道**：`data/filter.py` 长度/重复/正则/可选 fasttext 过滤；`data/dedup.py` MinHash+LSH 去重；`data/mixer.py` 多源混合；`data/prepare.py` 集成全部参数；`tests/test_data_quality.py` 覆盖
- [x] **[P2] 数据质量验证**：`data/validate.py`
- [x] **[P1] 算术数据多样化模板**：5种 hard 模板 (嵌套括号/优先级/多组/指数取模/两组乘法)
- [x] **[P1] 文档边界数据集可运行**：`DocBoundaryDataset` 支持 `resume_offset`
- [x] **[P1] 数据打包 (Packing)**：`PackingDataset` 多短文档打包 + `document_ids` 跨文档 mask；`prepare.py` `.idx` 写入 `doc_boundary`
- [x] **[P1] 数据预处理多进程化**：`data/prepare.py` 支持 `datasets.map(batched=True, num_proc=...)`，`.idx` 写入 dtype/vocab_size
- [x] **[P1] 算术数据集预编码**：`ArithmeticDataset` 默认 `pre_tokenize=True`，避免运行时重复 encode
- [x] **[P1] 评估批量化**：`evaluation/eval_alignment.py` 按长度分组 batch generate，KL 仅计算 response token
- [x] **[P2] 规则奖励细化**：`rewards/rule_reward.py` 增加过程奖励、多级 partial credit、异常输入拒绝、标签一致性检查
- [x] **[P2] 推理参数透传**：`inference/generate.py` benchmark 正确应用 `--temperature/--top_k/--top_p`
- [x] **[P1] 预训练标准化 Benchmark 评估**：`evaluation/eval_benchmark.py` 支持 perplexity + lm-eval
- [x] **[P1] 消融实验自动化脚本**：`run_ablations.py` train/inference 矩阵
- [x] **[P0] GRPO dropout guard**：默认拒绝 dropout > 0，提供 `--allow_dropout` 覆盖

### 质量保障
- [x] **[P0] 回归测试**：`tests/test_bugfixes.py` 覆盖 KV Cache dtype / 环形缓存顺序 / 滑动窗口淘汰 / 超长生成与 no-cache 一致性 / set-restore、优化器去重、EMA、checkpoint round-trip、DocBoundaryDataset、GRPO batched logprob 一致性、GRPO 梯度累积行为、IterativeGRPOTrainer 数据池构建、`data/prepare.py` shard/dtype/index、算术预编码、规则奖励各种 case；`tests/test_packing.py` 覆盖 packing 格式 / `document_ids` mask / shuffle buffer；`tests/test_generate_compile.py` 覆盖 `compile=True` 与 eager 输出一致；`tests/test_data_quality.py` 覆盖过滤/去重/混合
- [x] **[P0] 配置系统测试**：`tests/test_config.py` 覆盖 YAML 加载/环境变量/嵌套 CLI 覆盖、`NestedNamespace`、flatten/unflatten、`to_dict`、`validate_keys`
- [x] **[P1] 日志系统测试**：`tests/test_logger.py` 覆盖 wandb/TensorBoard 失败降级、标量/文本/直方图、梯度 norm / 显存日志、close 幂等
- [x] **[P1] 训练基础设施测试**：`tests/test_trainer_base.py` 覆盖分布式辅助单进程行为、种子可复现、`infer_device`、AMP 上下文、`CheckpointManager` 全状态往返与 `keep_last_n`、`BaseTrainer` 初始化、worker 种子
- [x] **[P2] Attention 后端测试**：`tests/test_attention_utils.py` 覆盖 backend 切换与默认值
- [x] **[P0] GRPO 测试**：`tests/test_grpo.py` 覆盖 dropout guard / allow override

---

## GRPO 与 Iterative GRPO 管线优化

针对 [IMPROVEMENT_REPORT.md](IMPROVEMENT_REPORT.md) §4.1 中识别的 RL 管线效率瓶颈，已完成以下改造：

### 1. 批量 Rollout 与批量 Logprob

```text
改造前:  for each prompt:
            for g in group_size:
                generate one response      <- batch_size=1
                forward policy/ref/new     <- 3-4次单条 forward

改造后:  encode prompts once
         for g in group_size:
             batch_generate(prompts)       <- 按长度分组 + KV Cache
         flatten all (prompt+response) sequences
         single forward for old_logprobs  <- [G*B, T]
         single forward for ref_logprobs  <- [G*B, T]
         single forward for new_logprobs  <- [G*B, T]
```

实现位置：
- `training/train_grpo.py::_generate_responses_from_tokens`
- `training/train_grpo.py::_batch_logprobs`
- `training/train_grpo.py::compute_grpo_loss`

### 2. 混合精度 (AMP)

通过 `BaseTrainer.setup_amp(use_bf16=True)` 启用 `torch.amp.autocast`：
- 支持 bf16（现代 GPU 原生）与 fp16 + GradScaler 自动 fallback；
- 覆盖 GRPO 的 rollout 采样、logprob forward、PPO/KL 损失、拒绝采样 SFT；
- CPU 训练时自动退化为 fp32。

### 3. 梯度累积

`--gradient_accumulation_steps` 现在真正生效：
- 每个 rollout 计算 loss 并 `backward()`；
- 损失按 `1 / gradient_accumulation_steps` 缩放；
- 每 N 个 rollout 执行一次 `optimizer.step()` + `zero_grad()`；
- 有效 batch size 从 `batch_size * group_size` 扩展到 `batch_size * group_size * gradient_accumulation_steps`。

### 4. 学习率调度

接入 `utils.lr_scheduler.LRScheduler`：
- 支持 `cosine` / `linear` / `wsd` / `constant`；
- 默认 cosine warmup：warmup 步数为总优化器步数的 5%，之后 decay 到 `min_lr`；
- 替代固定 `1e-5`，缓解 RL 初期不稳定与后期震荡。

### 5. Iterative GRPO 继承优化基座

`training/iterative_grpo.py` 重构为继承 `GRPOTrainer`：
- 自动复用批量采样、AMP、梯度累积、LR 调度、分布式支持；
- 保留迭代 RLHF（EMA 更新 ref）与 Rejection Sampling SFT；
- 拒绝采样阶段对同一 prompt 重复 N 次后做 batched generate，SFT 阶段也走 AMP。

### 6. Attention Mask 正确性修复

`model/modern_gpt.py` 中 `CausalSelfAttention` 在传入 `attention_mask` 时也会正确叠加 causal mask，保证 batch forward 与单条 forward 结果一致，这是 batched logprob 正确性的前提。

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

| 阶段 | 指标 | 目标 |
|------|------|------|
| 预训练 | ModernGPT val loss vs Baseline @ 18k iter | **-2.29%** (3.9126 -> 3.8229) |
| GQA | KV Cache 显存 vs MHA | **-50%** (GQA-4KV) / **-75%** (GQA-2KV) |
| 推理 | KV Cache 吞吐 vs no-cache | 长序列 (>400 tokens) 正向提升 |
| SFT | Format pass rate | easy 100%, medium >= 96% |
| GRPO-G4 | Accuracy vs SFT-only | easy +19.1 pts, medium +2.8 pts |
| GRPO-G4 | Format pass rate | **100%**, invalid rate **0%** |

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
python training/train_grpo.py --config config/grpo.yaml --init_from out/sft/best_sft-only.pt --ref_from out/sft/best_sft-only.pt

# GRPO + 梯度累积 (effective group_batch = 4 * 4 * 4 = 64)
python training/train_grpo.py --config config/grpo.yaml --gradient_accumulation_steps 4 --group_size 4 --batch_size 4

# ====== Iterative GRPO ======
python training/iterative_grpo.py --config config/grpo.yaml --init_from out/sft/best_sft-only.pt --ref_from out/sft/best_sft-only.pt \
    --ref_update_interval 250 --rejection_interval 200 --rejection_samples 64 --rejection_top_k 8

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
    vocab_size=50257,            # 词表大小 (GPT-2 tokenizer)
    block_size=1024,             # 最大上下文长度
    n_layer=12,                  # Transformer 层数
    n_head=8,                    # Query 注意力头数
    n_embd=512,                  # 隐藏维度 (head_dim = 512/8 = 64)
    n_kv_head=None,              # KV 头数 (None=n_head 即 MHA; 设为 2/4 启用 GQA)
    intermediate_size=None,      # SwiGLU 隐层维度 (None=自动: 8d/3 向上取整到128倍数 -> 1408)
    dropout=0.0,                 # Dropout rate
    norm_position="pre",         # Pre-Norm / Post-Norm
    num_experts=1,               # MoE experts (1=密集 SwiGLU, >1=top-1 gating)
    gradient_checkpointing=False,# 是否启用梯度检查点
    qk_norm=False,               # 是否在 RoPE 前对 Q/K 做 per-head RMSNorm
    attn_temperature=1.0,        # Attention 温度系数 (1.0=标准 1/sqrt(head_dim))
    rmsnorm_eps=1e-6,            # RMSNorm epsilon
    rope_theta=10000.0,          # RoPE 基频 theta
    rope_scaling=None,           # RoPE 长度外推, e.g. {"type": "ntk", "factor": 2.0}
    moe_aux_loss_factor=0.01,    # MoE load-balancing aux loss 系数
    moe_capacity_factor=1.25,    # MoE per-expert 容量因子
    use_flash_attn=False,        # 是否尝试第三方 flash-attn 后端
    sliding_window_size=None,    # Sliding Window Attention 窗口大小 (None=全长)
    use_paged_kv_cache=False,    # 是否使用 block-table Paged KV Cache
    kv_cache_block_size=16,      # Paged KV Cache 每块 token 数
    n_future=0,                  # Multi-Token Prediction 未来头数量 (0=关闭)
    mtp_weight=1.0,              # MTP loss 权重
    gqa_broadcast="auto",        # GQA 广播策略: auto(探测)/grouped(零拷贝)/raw(直传)/repeat(兼容)
)
```

### 参数量精确对齐

| 配置 (12L/512D)        | 总参数量  | 非嵌入参数 | KV Cache (B/tok) |
|------------------------|----------|-----------|-------------------|
| BaselineGPT            | 54.6M    | 28.4M     | N/A               |
| ModernGPT MHA (n_kv=8) | 54.0M    | 28.3M     | 18,432            |
| ModernGPT GQA-4KV      | 51.7M    | 26.0M     | **9,216** (-50%)  |
| ModernGPT GQA-2KV      | 50.5M    | 24.8M     | **4,608** (-75%)  |

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
│   ├── flash_attention.py      # FlashAttention 封装 (flash_attn_func/varlen/kvcache + 条件导入 + 自动回退)
│   ├── attention_utils.py      # Attention 工具 (后端切换 + GQA SDPA 探测 probe_gqa_sdpa_support)
│   └── kv_cache_utils.py       # KVCacheManager + 滑动窗口
├── training/
│   ├── trainer_base.py         # BaseTrainer: 统一分布式/AMP/scheduler/checkpoint/logger
│   ├── train_pretrain.py       # AMP+GradScaler+梯度累积+LR Scheduler+EMA+EarlyStop+DDP/FSDP
│   ├── train_sft.py            # SFT + AMP + 梯度累积 + LR Scheduler
│   ├── train_grpo.py           # GRPO + 批量采样/logprob + AMP + 梯度累积 + LR Scheduler
│   └── iterative_grpo.py       # 迭代 RLHF + Rejection Sampling SFT (继承优化后 GRPOTrainer)
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
cache 路径与 no-cache 路径均采用绝对 `start_pos`：no-cache 在裁剪到最近 `block_size` 个 token 时传入对应的 `start_pos`，cache 路径在滑动窗口淘汰旧 token 时同步推进 `start_pos`。同时 `KVCacheManager.advance()` 保留 `max_cache_len - 1` 个 token，为下一个 decode token 预留位置，使得两种路径看到的上下文长度与位置编码完全一致。`tests/test_bugfixes.py::test_kv_cache_long_generate_consistency` 覆盖超出缓存长度后的 bit-wise 一致性。

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

**GRPO 中梯度累积如何扩大有效 batch size？**
每个 rollout 先采样 `batch_size` 个 prompt，每个 prompt 生成 `group_size` 条 response。设置 `--gradient_accumulation_steps N` 后，每 N 个 rollout 才更新一次参数，因此有效 batch size 变为 `batch_size * group_size * N`，可用显存换取更稳定的 RL 梯度。

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
