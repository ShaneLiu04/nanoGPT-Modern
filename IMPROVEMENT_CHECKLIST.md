# nanoGPT-Modern 系统改进清单

> 基于对全项目源码（模型、训练、推理、数据、评估、工具）的逐模块审查，整理出以下归类改进建议。
> 每项按 **优先级（P0-P3）** 和 **影响面** 标注，P0 为最紧急/高收益，P3 为锦上添花。

---

## 实施进度追踪 (2026-06-02 更新)

以下统计基于对当前代码库 (`nanogpt-modern/`) 的完整审查。

### 已完成 (38 / 38 项)

| 编号 | 优先级 | 项目 | 实施说明 |
|------|--------|------|----------|
| 1.1 | P0 | GQA 完整实现 | `pretrain.yaml` 设 `n_kv_head: 2`，`repeat_interleave` 正确执行，KV Cache 节省 75% |
| 1.2 | P1 | BaselineGPT SDPA 后端 | `attention_backend="sdpa"|"manual"` 通过 Config 切换 |
| 1.3 | P1 | Pre/Post-Norm 消融 | `norm_position="pre"|"post"` 两个模型均支持 |
| 1.4 | P2 | SwiGLU multiple-of 对齐 | `intermediate_size` 向上取整至 128 的倍数 (512→1408) |
| 1.6 | P3 | MoE FFN 探索 | `num_experts` >= 1, top-1 gating, `SwiGLU` 类完整实现 |
| 2.1 | P0 | 梯度累积 | `--gradient_accumulation_steps` 参数, 正确除以 accum_steps |
| 2.2 | P1 | LR Scheduler 多模式 | `utils/lr_scheduler.py` 实现 cosine/linear/wsd/constant |
| 2.3 | P1 | EMA | `ModernGPT.init_ema/update_ema/apply_ema_weights/restore_ema_weights` |
| 2.4 | P1 | SFT LR Scheduler | `train_sft.py` 已集成 LR Scheduler |
| 2.5 | P2 | GradScaler | `train_pretrain.py` 中 float16 时自动启用 |
| 2.6 | P2 | Early Stopping | `--early_stopping_patience` 参数, 基于 val loss |
| 2.8 | P3 | FSDP | `--fsdp` + `--fsdp_sharding_strategy` + `save_checkpoint_raw` |
| 3.1 | P0 | CUDA Events 计时 | `inference/generate.py` 使用 CUDA Events + prefill/decode 分离 |
| 3.1b | P2 | 推理采样参数透传 | `inference/generate.py` benchmark 正确应用 `--temperature/--top_k/--top_p` |
| 3.3 | P1 | RoPE 缓存 | `RotaryEmbedding._cos_cached/_sin_cached` 惰性计算 + 复用 |
| 3.4 | P2 | 静态 KV Cache / KVCacheManager 集成 | `generate()` cache 路径使用预分配环形 `KVCacheManager`，避免 `torch.cat`，cache/no-cache 超长生成一致 |
| 3.5 | P2 | 生成策略扩展 | top_p (nucleus) + repetition_penalty |
| 4.1 | P0 | dtype 自动选择 + map/流式 shard 化 | `prepare.py` 默认 `datasets.map(batched=True, num_proc=...)` 多进程 tokenize；`.idx` 写入 dtype/vocab_size/边界；保留 `--streaming` 路径 |
| 4.3 | P1 | 算术数据多样化 | 5 种 hard 模板 (嵌套/优先级/多组/指数取模/两组乘法) |
| 4.4 | P2 | 算术数据集预编码 | `ArithmeticDataset` 默认 `pre_tokenize=True`，避免 DataLoader worker 重复 tokenize |
| 4.5 | P2 | 数据质量验证 | `data/validate.py` 完整实现 |
| 5.4 | P2 | 规则奖励细化 | 格式分 + 过程奖励 + 连续正确性分 + 异常输入拒绝 |
| 6.2 | P1 | 评估批量化与 KL 一致性 | `eval_alignment.py` 按长度分组 batch generate，KL 仅计算 response token |
| 7.1 | P0 | 统一训练抽象 BaseTrainer | `training/trainer_base.py` 提供 `BaseTrainer`/`CheckpointManager`；pretrain/sft/grpo 均已继承 |
| 7.1b | P0 | 统一 Config 系统 | `utils/config.py` 支持嵌套 YAML + argparse CLI 覆盖（含 `--optimizer.lr` 点号语法）、环境变量展开、`to_dict`/`validate_keys` |
| 7.2 | P1 | 日志系统失败降级与扩展 | `utils/logging.py` 支持 wandb/TB 失败降级、文本样例、梯度 norm / 显存直方图、日志等级控制 |
| 7.3 | P1 | 补齐依赖 | `requirements.txt` 新增 `datasets`、`huggingface_hub`、`safetensors`、`omegaconf`、`hydra-core`、`pytest` |
| 7.4 | P0 | 单元测试覆盖 | `tests/` 新增/扩展 `test_bugfixes.py`、`test_config.py`、`test_logger.py`、`test_trainer_base.py`、`test_attention_utils.py`、`test_grpo.py`；运行 `pytest tests/ -q` |
| 1.5 | P2 | Flash Attention / SDPA 显式后端选择 | `model/attention_utils.py` 提供 `set_attention_backend`（auto/flash/mem_efficient/math/default）；训练与推理脚本新增 `--attn_backend` 参数 |
| 4.4 | P2 | Streaming DataLoader 可恢复状态 | `MemmapDataset` / `DocBoundaryDataset` 新增 `state_dict` / `load_state_dict`；`DocBoundaryDataset` 正确应用 `resume_offset` |
| 5.1 | P0 | GRPO old_logprobs dropout 一致性强化 | `train_grpo.py` 默认拒绝 dropout > 0 的 SFT checkpoint，提供 `--allow_dropout` 覆盖；保留 `maybe_warn_dropout` 警告 |
| 6.1 | P0 | 预训练标准化 Benchmark 评估 | 新增 `evaluation/eval_benchmark.py`：本地 val set  perplexity + 可选 `lm-eval` 下游任务（HellaSwag/LAMBADA 等） |
| 6.2 | P1 | 消融实验自动化脚本 | 新增 `run_ablations.py`：支持 `train` / `inference` 两种消融矩阵，输出 JSON 与 Markdown 汇总表 |
| 7.5 | P2 | Checkpoint 生命周期管理 | `CheckpointManager` 新增 `--keep_last_n`：自动保留最近 N 个非 best/final/ema checkpoint |
| 7.6 | P2 | 种子管理健壮性 | `BaseTrainer` 所有 rank 使用相同全局种子保证 DDP/FSDP 模型初始化一致；新增 `make_worker_init_fn` 用于 DataLoader worker；各训练脚本接入 |
| 7.7 | P3 | pyproject.toml 可安装化 | 新增 `pyproject.toml`，支持 `pip install -e .` 与 `[project.scripts]` 入口 |
| 2.7 | P2 | DataLoader shuffle 与 token 跨越优化 | `MemmapDataset` 新增 `shuffle_buffer` 缓冲式乱序；`DocBoundaryDataset` 按 EOT 截断，避免跨文档注意力；`PackingDataset` 支持多文档打包 |
| 3.2 | P1 | `generate()` token-by-token 循环的 CUDA Graph 优化 | `ModernGPT.generate()` 新增 `compile=True`：在 CUDA 上用 `torch.compile(mode="reduce-overhead", fullgraph=False, dynamic=True)` 编译 forward，CPU 无 C++ 编译器时自动回退 eager；`inference/generate.py` 新增 `--compile` 参数 |
| 4.2 | P1 | 数据打包 (Packing) 策略与跨文档 mask | `data/prepare.py` 记录 `doc_boundary`；`data/openwebtext.py` 新增 `PackingDataset`，产出 `(x, y, document_ids)`；`ModernGPT.forward/CausalSelfAttention` 支持 `document_ids` 跨文档 mask；`train_pretrain.py` 解包 document_ids 传入模型 |

### 待实施 (0 项)

全部改进项已完成。

### 文档完善 (RECOMMENDED)

| 编号 | 优先级 | 项目 |
|------|--------|------|
| 8.1 | P1 | Quick Start (已加入 README) |
| 8.2 | P1 | API 文档 / docstring 标准化 |
| 8.3 | P2 | Architecture Decision Records |


---

## 一、模型架构层

### 1.1 [P0] 补全 GQA (Grouped Query Attention) 的完整实现

**现状**：`ModernGPTConfig` 已预留 `n_kv_head` 参数，`CausalSelfAttention` 中也计算了 `self.n_rep = self.n_head // self.n_kv_head`，且 `k/v` 使用了 `repeat_interleave`。但在配置初始化和训练入口中，`n_kv_head` 始终等于 `n_head`（未暴露配置入口），这意味着 GQA 实际并未生效。

**建议**：
- 在 `pretrain.yaml` 和 `ModernGPTConfig` 中暴露 `n_kv_head` 参数（如设为 `n_head / 2`）
- 对照实验：BaselineGPT（无 GQA）vs ModernGPT（GQA=2/4），测量推理时 KV Cache 显存节省与生成质量
- 注意 `BaselineGPT` 的 `c_attn` 用的是合并 QKV 的单矩阵，若要支持 GQA 需拆分投影

### 1.2 [P1] 统一 BaselineGPT 的 Attention 实现

**现状**：`BaselineGPT.CausalSelfAttention` 使用手写矩阵乘法和三角 mask（`self.bias`），而 `ModernGPT` 使用 `F.scaled_dot_product_attention`（PyTorch 内置 Flash Attention 后端）。两者实现方式不一致，对照实验的可信度受影响——Baseline 若替换为 SDPA 可能也有提速。

**建议**：
- 为 `BaselineGPT` 也提供 `F.scaled_dot_product_attention` 路径，通过配置开关控制
- 或明确记录：Baseline 的手动实现是故意的（保留 nanoGPT 原始代码风格），规避优化后端差异对 loss 对比的干扰

### 1.3 [P1] 添加 Pre-Norm vs Post-Norm 的消融选项

**现状**：两个模型都是 Pre-Norm（`x = x + Attn(Norm(x))`），这是 LLaMA 风格。但 GPT-2 原始架构用的是 Post-Norm（`x = Norm(x + Attn(x))`）。

**建议**：
- 在 Block 中通过配置 `norm_position: "pre" | "post"` 支持切换
- 这是一个经典消融实验，对小模型尤其明显——Post-Norm 在浅层网络中训练更稳定但最终 loss 略高于 Pre-Norm

### 1.4 [P2] SwiGLU 的 multiple-of 对齐

**现状**：`intermediate_size = int(8/3 * n_embd) = 1365`，这个数字不被 2 的幂整除，在 GPU 上可能因内存对齐问题产生微小性能损失。

**建议**：
- 参考 LLaMA 的 `multiple_of=256` 约束，将隐层维度向上取整至 128 的倍数（如 1408）
- 附带影响：参数量会从 54.0M 略微增加到 ~54.5M。可在报告中记录 "参数对齐区间内 ±1% 视为等价"

### 1.5 [P2] 添加 Flash Attention / SDPA 显式后端选择 ✅ **已完成**

**现状**：依赖 `F.scaled_dot_product_attention` 自动选择后端（在 PyTorch 2.0+ 上，满足条件时自动使用 Flash Attention）。但没有显式的 fallback 检测日志，也无法强制指定后端。

**实现要点**：
- 新增 `model/attention_utils.py`：
  - `set_attention_backend(backend)`：强制启用 `flash` / `mem_efficient` / `math`，或 `auto`/`default` 让 PyTorch 自动选择；
  - `get_attention_backend_info()` / `print_attention_backend()`：查询/打印当前启用的后端；
- `train_pretrain.py`、`train_sft.py`、`train_grpo.py`、`inference/generate.py` 均新增 `--attn_backend` 参数，在模型创建前调用 `set_attention_backend()`；
- 启动日志现在会输出当前启用的 SDPA 后端列表，便于排查实际使用的 attention kernel。



### 1.6 [P3] MoE (Mixture of Experts) FFN 探索

**现状**：SwiGLU FFN 的参数量为 `3d × hidden`。在 50M 参数总量下，一个 Sparsely-Gated MoE（如 4 experts, top-1 gating）能保持单次 forward 计算量不变，但增加总容量。

**建议**：
- 作为实验性 Extension：在 FFN 层加 `num_experts` 和 `top_k` 参数
- 评估指标：相同 forward FLOPs 下的 val loss vs 参数量 scaling 曲线







---

## 二、训练系统层

### 2.1 [P0] 预训练缺少梯度累积 (Gradient Accumulation)

**现状**：`train_pretrain.py` 中每步 `batch_size=12, block_size=1024` 只做单次 `backward()` + `optimizer.step()`。虽然支持 DDP 扩展全局 batch，但**单卡上的 effective batch size 仅为 12**，这对于 1.13B token 的数据显然偏小。而且代码注释中写了 `if gradient accumulation needed, divide by accum_steps here` 但未实现。

**建议**：
```python
# 添加
accum_steps = args.gradient_accumulation_steps  # 如 4
loss = loss / accum_steps
loss.backward()
if (iter_num + 1) % accum_steps == 0:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```
- 这能在不增加显存的情况下将 effective batch size 提升至 48/96

### 2.2 [P1] 学习率调度器与 Warmup 可配置性不足

**现状**：LR 调度内嵌在 `get_lr()` 函数中，固定为 linear warmup + cosine decay。不支持 step decay、constant with cooldown、WSD（Warmup-Stable-Decay）等流行方案。

**建议**：
- 抽象一个 `LRScheduler` 类，支持 `cosine` / `linear` / `wsd` / `constant` 四种模式
- 在 `pretrain.yaml` 中添加 `lr_schedule: cosine` 配置项

### 2.3 [P1] 缺少 EMA (Exponential Moving Average) 模型

**现状**：未使用 EMA。这在小型模型上通常不是问题，但对于 50M 参数训练 18000 步，EMA 能稳定 eval loss，减少因最后几步权重抖动造成的 checkpoint 选择偏差（`best_val_loss` 更平滑）。

**建议**：
- 使用 `torch.optim.swa_utils.AveragedModel` 或手动维护 shadow weights
- 保存 EMA checkpoint 用于最终评估

### 2.4 [P1] SFT 训练未用 LR Scheduler

**现状**：`train_sft.py` 中训练 3 个 epoch，但学习率始终为 `3e-4`，没有任何衰减。

**建议**：
- 至少加入 epoch-level 的 cosine decay 或 step decay
- 或用 `torch.optim.lr_scheduler.CosineAnnealingLR`

### 2.5 [P2] 缺少 Mixed Precision 的梯度缩放 (GradScaler)

**现状**：`train_pretrain.py` 使用了 `torch.amp.autocast` 但没有配套 `GradScaler`。在 `float16` 模式下，梯度下溢可能导致某些参数不更新。

**建议**：
- 当 dtype 为 `float16`（非 bfloat16）时，包装 `GradScaler`:
```python
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
with ctx:
    loss = ...
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2.6 [P2] ~~预训练 DataLoader 的 Shuffle 与 Token 跨越问题~~ ✅ **已完成**

**现状**：`openwebtext.py` 中 `MemmapDataset` 按 `block_size+1` 连续切分，每次 yield `x=chunk[:-1], y=chunk[1:]`。但 DataLoader 设置 `shuffle=False`，且 `IterableDataset` 的 shuffle 在流式场景下无法跨 epoch 重排。

**实现要点**：
- `MemmapDataset` 新增 `shuffle_buffer` 参数：在 worker 范围内按 buffer 大小（默认 10000）对 chunk 索引做局部随机打乱，无需加载全量数据即可破坏顺序相关性；
- `DocBoundaryDataset` 扫描 EOT 位置，遇到文档边界时截断 chunk，避免模型在无关文档之间建立跨文档注意力；
- `PackingDataset`（见 4.2）进一步将多个短文档打包到同一序列，并通过 `document_ids` mask 阻断跨文档注意力，兼顾显存效率与语义边界。

**验证**：`tests/test_packing.py::test_memmap_dataset_shuffle_buffer`。

### 2.7 [P2] 缺少 Early Stopping 机制

**现状**：预训练固定跑完 `max_iters=18000`。若 val loss 在 12k iter 就已收敛，后面的 6k iter 只是在浪费算力。

**建议**：
- 添加 `early_stopping_patience` 参数，当 `best_val_loss` 在连续 N 次 eval 中未改善时提前终止



### 2.8 [P3] 未支持 FSDP (Fully Sharded Data Parallel)

**现状**：仅支持 DDP（单卡或多卡数据并行）。对 54M 模型来说 DDP 已够用，但作为框架，FSDP 支持可扩展至更大模型。

**建议**：
- 添加 `--fsdp` 参数，使用 PyTorch FSDP 包装模型，支持跨 GPU 分片参数、梯度和优化器状态







---

## 三、推理系统层

### 3.1 [P0] 推理吞吐实验的测量方法有误导性

**现状**：实验日志中 cache 模式吞吐反而**低于** no-cache（400 tokens: 158 vs 205 tok/s）。原因是测试模型只有 3M 参数（fast 验证模式），KV Cache 的 Python 循环管理开销超出了节省的 FLOPs。

**建议**：
- 在完整 54M 模型上重做推理消融（这是 README 承诺的实验，但目前未做）
- 在 benchmark 函数中添加 GPU warmup 的同步 + CUDA event timing（当前使用 `time.time()` 不够精确）
- 分离测量 "prefill 阶段" 和 "decode 阶段" 的延迟，这是业界的标准分析格式

### 3.1b [P2] ~~`inference/generate.py` 采样参数未生效~~ ✅ **已修复**

**现状**：`benchmark()` 中 cache 路径硬编码 `temperature=1.0` 且未使用 `top_k/top_p`。

**修复**：`benchmark()` 现在接收 `temperature`/`top_k`/`top_p`/`use_cache` 参数，cache 路径调用 `inference.generate_utils._sample_logits` 应用这些超参数，no-cache 路径通过 `model.generate(...)` 透传。`main()` 循环将命令行参数正确传入。

### 3.2 [P1] ~~`generate()` 中 token-by-token 循环未向量化~~ ✅ **已完成**

**现状**：`ModernGPT.generate(use_cache=True)` 每次 forward 只传 `idx[:, -1:]`，循环 `max_new_tokens` 次。每次循环都要从 Python 发起一次完整的 PyTorch forward（kernel launch overhead）。

**实现要点**：
- `ModernGPT.generate()` 新增 `compile=False` 参数；在 CUDA 设备上启用时，用 `torch.compile(self, mode="reduce-overhead", fullgraph=False, dynamic=True)` 编译 forward 调用，减少 Python 侧 kernel launch overhead；
- 捕获/编译失败（如缺少 C++ 工具链、CUDA Graph 不支持的结构）自动降级到 eager 模式并打印 warning，不中断生成；
- `inference/generate.py` 新增 `--compile` 参数，benchmark 的 cache/no-cache 路径均会透传编译开关；
- 当前 KV Cache 采用动态环形 buffer（单段 tuple / 多段 list 混合），真正的静态 CUDA Graph 需要把 cache 形状完全固定；因此本次采用 `torch.compile` 作为工程上可落地的 graph-mode 优化，后续如需极致低延迟可进一步把 KV Cache 改为固定长度 padded tensor。

**验证**：`tests/test_generate_compile.py` 覆盖 CUDA/CPU 两种场景，确保 `compile=True` 与 eager 输出一致。

### 3.3 [P1] RoPE 旋转矩阵每次生成都重新计算

**现状**：`RotaryEmbedding.forward` 每次调用都重新生成 `cos`、`sin`。在 token-by-token 推理中，`cos_full` 和 `sin_full` 对相同 `seq_len` 多次重复计算。

**建议**：
- 缓存最近使用的 `cos`/`sin`，只在 `seq_len` 增大时扩展
- 预计算到 `max_seq_len`，按索引切片

### 3.4 [P2] ~~`KVCacheManager` 未在 `generate()` 中使用~~ ✅ **已实现**

**现状**：`ModernGPT.generate()` 已使用 `KVCacheManager` 管理缓存，并进一步将其重构为预分配静态环形缓存 `[B, n_kv_heads, max_cache_len, head_dim]`，通过 `write_pos` 指针写入，避免 decode 阶段反复 `torch.cat`。

**实现要点**：
- `update()` 切片写入，支持物理缓冲区环绕；
- `advance()` 维护逻辑长度、`start_pos` 与写指针，淘汰旧 token 时无数据搬移；
- 保留 `max_cache_len - 1` 个 token，为下一个 decode token 预留位置，保证 cache/no-cache 超长生成时数值一致；
- `CausalSelfAttention.forward()` 兼容单段 tuple 与多段 list 两种 past_kv 格式。

**验证**：`tests/test_bugfixes.py` 新增 `test_kv_cache_ring_buffer_order`、`test_kv_cache_sliding_window_eviction`、`test_kv_cache_long_generate_consistency`、`test_kv_cache_set_restore`。

### 3.5 [P2] 生成策略单一（仅 top-k sampling）

**现状**：`generate()` 只支持 `temperature + top_k`。缺少 `top_p (nucleus)`、`min_p`、`beam_search`、`repetition_penalty`、典型采样等。

**建议**：
- 添加 `top_p`、`repetition_penalty` 作为最低限度的扩展
- 这些策略对 GRPO 中的采样多样性有直接影响（当前用 temperature=1.0 + top_k=50 较为粗糙）



### 3.6 [P3] ~~未支持 batch 推理~~  ✅ **已实现**

**现状**：`generate()` 的 KV Cache 模式虽通过 `start_pos` 支持 batch 维度（见代码 `B, T, C = x.size()`），但同 batch 内所有序列必须同时完成生成（没法逐条提前终止）。

**建议**：
- 添加 `eos_token_id` 参数 + `finished` mask，允许 batch 内提前结束的序列停止生成
- 这对评估阶段批量生成提升显著







---

## 四、数据管道层

### 4.1 [P0] ~~OpenWebText 数据预处理的 Token 截断风险~~ ✅ **已实现**

**现状**：`prepare.py` 根据 tokenizer 词表大小自动选择 `np.uint16`（vocab <= 65535）或 `np.uint32`（vocab > 65535），并在 `.idx` 中记录 `dtype=`。`data/openwebtext.py` 的 `_detect_dtype_from_index()` 读取该字段，避免未来大词表静默溢出。

**实现要点**：
- `_select_dtype(vocab_size)` 自动选择 dtype；
- `_ShardWriter.write_index()` 写入 `dtype=`、`vocab_size=` 等元数据；
- 默认使用 `datasets.map(batched=True, num_proc=...)` 多进程 tokenize，流式写入 shards；
- `--streaming` 模式保留为低内存备选。

### 4.2 [P1] ~~缺少数据打包 (Packing) 策略~~ ✅ **已完成**

**现状**：`MemmapDataset` 把 token 序列按 `block_size` 硬切分，不考虑 document 边界。这会导致一个文档的结尾和另一个文档的开头拼接在一起，产生语义噪声。

**实现要点**：
- `data/prepare.py` 在 shard 写入阶段记录 `doc_boundary=`（抽样写入，最多 10k 条），并在 `.idx` 元数据中写入 `eot_token=`；
- `data/openwebtext.py` 新增 `PackingDataset`：
  - 扫描 EOT 将 token 流切分为文档；
  - 贪心将多个短文档打包进 `block_size` 序列，长文档则切成连续 block；
  - 每个 sample 返回 `(x, y, document_ids)`，其中 `document_ids` 标记每个 token 所属文档，padding 位置为 `-1`；
  - 新文档第一个 token 及 padding 对应的 `y` 被设为 `-1`，避免在这些位置计算 loss；
- `model/modern_gpt.py` 的 `CausalSelfAttention` 与 `ModernGPT.forward` 支持 `document_ids`：在 causal mask 基础上增加 "同文档才允许 attend" 的 additive mask，阻止跨文档注意力；
- `training/train_pretrain.py` 的 `_forward()` 解包 `(input_ids, targets, document_ids)` 并在 `model == "modern"` 时传入模型；
- `get_openwebtext_dataset(..., use_packing=True)` 与 `get_dataloader(..., use_packing=True)` 暴露打包开关。

**验证**：`tests/test_packing.py` 覆盖 packing 输出格式、`document_ids` mask 对 logits 的影响、shuffle buffer。

### 4.3 [P1] 算术数据生成缺乏难度校准

**现状**：`generate_hard()` 固定生成 `(a + b) * c / d` 格式，变化有限。且 hard 数据仅 1 种模式，缺乏多样性。

**建议**：
- 新增更多模板：嵌套括号、模运算、指数运算（`a**b`）、多步表达式（A + B * C - D / E）
- 添加 `generate_expert` 级别：包含文字推理（"Alice has 3 apples and buys 5 more..."）
- 对每个生成样本，用 `eval()` 计算答案时捕获异常更细致地区分格式错误 vs 数值错误

### 4.4 [P2] Streaming DataLoader 不支持 Resumable State ✅ **已修复**

**现状**：`IterableDataset` 的 `__iter__` 依赖 `worker_info` 划分区间，重启后从头开始遍历，断点续训时数据状态无法保存。

**实现要点**：
- `MemmapDataset` 与 `DocBoundaryDataset` 新增 `state_dict()` / `load_state_dict()`，返回/恢复 `{"resume_offset": N}`；
- `DocBoundaryDataset.__iter__()` 现在正确按全局计数跳过前 `resume_offset` 个 segment；
- `get_dataloader()` 保持 `resume_offset` 参数，训练脚本可在 checkpoint 恢复后重建 DataLoader 并从断点继续。

**验证**：`tests/test_bugfixes.py` 已覆盖 `DocBoundaryDataset.resume_offset` 行为。

### 4.5 [P2] 缺少数据质量验证脚本

**现状**：数据准备后无单独的验证步骤（如检查 BPE 编码是否正常、token 分布是否符合预期、特殊 token 出现频率）。

**建议**：
- 添加 `data/validate.py`：检查 token 范围是否在词表内、统计 token 直方图、输出 random samples 的 decode 文本







---

## 五、RL 对齐系统层

### 5.1 [P0] GRPO 实现中 Old/New Policy 分离有 Bug 隐患 ✅ **已修复**

**现状**：`GRPOTrainer.sample_group()` 在 `self.policy.eval()` + `torch.no_grad()` 下生成样本并记录 `old_logprobs`，随后 `compute_grpo_loss()` 在 `policy.train()` 下重新 forward 计算 `new_logprobs`。对于 `dropout=0.0` 的模型，这在数学上是正确的；但如果 `dropout > 0`，eval/train 模式下的 dropout mask 不同，会导致 PPO ratio 有偏。

**实现要点**：
- `train_grpo.py` 在 `_build_models()` 中检查加载模型的 `config.dropout`；
- 若 `dropout > 0` 且未传入 `--allow_dropout`，直接抛出 `ValueError`，防止用户无意中使用有偏设置；
- 新增 `--allow_dropout` 参数供明确知晓风险的用户覆盖；
- 保留 `maybe_warn_dropout()` 的显式 `UserWarning`。

**验证**：`tests/test_grpo.py` 新增 `test_grpo_rejects_dropout_by_default`、`test_grpo_allows_dropout_with_flag`。

### 5.2 [P1] GRPO 缺少 Value Clipping 和 Advantage Normalization

**现状**：`compute_loss()` 中的 advantage 计算方式为：
```python
rewards_t = torch.tensor(rewards, device=device).float()
adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
```
这是 Group-level normalization。但 GRPO 原论文还建议 advantage clipping 和 whitening。

**建议**：
- 添加 `--adv_norm` 参数，支持 `group` / `batch` / `none` 三种 advantage 归一化策略
- 可选 advantage clipping：`adv = torch.clamp(adv, -3.0, 3.0)`

### 5.3 [P1] KL 散度计算方法可能不稳定

**现状**：KL 计算为 `kl = exp(new_logp) * (new_logp - ref_logp)`，这是 KL(policy || ref)。但 GRPO 论文使用的是反向 KL `KL(ref || policy) = ref_logp - new_logp`（不乘概率权重）。两者不等价。

**问题**：当前实现的 KL 形式 `ref_logp - new_logp`（逐 token 差值的平均值）与 README 中描述的不完全一致。且 `exp(new_logp)` 在分布尖锐时可能溢出。

**建议**：
- 统一确认 KL 方向：KL(policy || ref) vs KL(ref || policy)
- 用 `torch.clamp(logp, min=-20, max=0)` 限制 log probability 范围
- 若使用反向 KL（`ref_logp - new_logp`），它天然不需要 `exp` 加权，数值更稳定

### 5.4 [P2] ~~Reward 函数过于二元~~ ✅ **已实现**

**现状**：`rule_reward.py` 已重构为连续、多组件奖励函数：

- **格式分 (0.0~0.5)**：正确且唯一的 `<answer>...</answer>` 块得 0.5；存在标签但 malformed（多块、未闭合、非数字内容）得 0.3；无标签 0.0。
- **过程分 (0.0~0.3)**：检测 `<answer>` 之前的推导标记（`=`, `step`, `then`, `=>` 等），鼓励展示中间步骤。
- **正确性分 (0.0~1.2)**：基于相对误差的多级 partial credit：`<1e-6` 1.2 分，`<1e-4` 0.9 分，`<1e-2` 0.6 分，`<1e-1` 0.3 分；`ref==0` 时退化为绝对误差。
- **数值安全**：`_parse_number` 拒绝 `nan`/`inf`/`overflow`/`1e1000`，并清理千分位、空格、货币符号。

**验证**：`tests/test_bugfixes.py` 新增 `test_rule_reward_perfect_partial_and_malformed`、`test_rule_reward_process_score`。

### 5.5 [P2] GRPO 的 Prompt 构造过于简单

**现状**：所有 prompt 格式为 `"Solve: {expr}\nWrap your final answer in <answer>...</answer>."`。模型学会的是 "在特定模板后输出 `<answer>`"——这不是真正的指令遵从。

**建议**：
- 构造多样化的 prompt 模板（如 10-20 种变体）："计算下列表达式" / "请解答" / "What is" / "帮我算一下"
- 在 GRPO 训练时随机混合模板，迫使模型学习通用的指令服从而非模板记忆

### 5.6 [P3] 缺少 Iterative RLHF / Rejection Sampling

**现状**：GRPO 仅做一轮，从 SFT 模型开始训练 1000 steps。

**建议**：
- 实现 iterative RLHF：每 N steps 用当前 best policy 替换 reference model（类似 LLaMA 2 的 iterative fine-tuning）
- 或加入 rejection sampling：从当前 policy 采样大批量，只保留 top-K 高奖励样本用于 SFT 微调







---

## 六、评估与实验层

### 6.1 [P0] 缺少预训练的标准化 Benchmark 评估 ✅ **已完成**

**现状**：预训练仅评估 val loss（perplexity），没有在下游任务上评估。README 中所有的指标都是 alignment 阶段的。

**实现要点**：
- 新增 `evaluation/eval_benchmark.py`：
  - 本地 tokenized val set  perplexity（始终可用，不依赖外部库）；
  - 可选 `lm-eval` 下游任务（HellaSwag、LAMBADA 等），若未安装则跳过并提示；
  - 支持 `--output_json` 保存结果。
- 命令示例：
  ```bash
  python evaluation/eval_benchmark.py --checkpoint out/pretrain/best_ckpt.pt \
      --data_dir data/openwebtext --split val --tasks hellaswag,lambada_openai
  ```

### 6.2 [P1] 评估脚本与训练的指标体系不统一

**现状**：
- 预训练：仅 log `train/loss, val/loss, tokens_per_sec`
- SFT：仅 log `train/loss, val/loss`
- GRPO：log `loss, policy_loss, kl_loss, mean_reward`
- 评估脚本：log `accuracy, reward, format_pass_rate, invalid_rate, kl_divergence`

**建议**：
- 统一所有训练阶段的指标命名规范（如 `train/loss`, `train/perplexity`, `train/throughput_tok_per_sec`）
- 在 Logger 中支持 `log_metrics()` 批量写入 + auto flatten（当前 `log_scalars` 需手动展开）

### 6.2b [P1] ~~`eval_alignment.py` 串行推理与 KL 计算不一致~~ ✅ **已修复**

**现状**：早期版本对每条 prompt 单独调用 `model.generate(...)`，batch_size=1；KL 对 `prompt + response` 整体计算，未 mask prompt token。

**修复**：
- 使用 `inference.generate_utils.generate_by_length` 按 prompt 长度分组 batch generate；
- KL 仅对 response target positions 求和，与 GRPO 训练时的 KL 项一致；
- 输出新增 `process_score` 指标。

### 6.3 [P1] 缺少 Ablation Study 自动化脚本 ✅ **已完成**

**现状**：每个消融实验需要手动改配置或命令行参数。

**实现要点**：
- 新增 `run_ablations.py`：
  - `--mode inference`：运行 baseline/modern × cache/no-cache 推理吞吐矩阵；
  - `--mode train`：运行 baseline/modern 小模型预训练对比（可配置 `--n_layer/--n_head/--n_embd/--max_iters`）；
  - 自动收集结果到 `--out_json`，并打印 Markdown 汇总表。
- 命令示例：
  ```bash
  python run_ablations.py --mode inference --checkpoint out/pretrain/best_ckpt.pt
  python run_ablations.py --mode train --data_dir data/openwebtext_test --max_iters 100
  ```

### 6.4 [P2] README 中的目标指标与实际快速验证实验结果脱节

**现状**：README 宣称的目标（val loss 3.8229, easy accuracy 89.1%…）来自 50M 模型 + 18k iter + 完整 OpenWebText 的预期。但 `FULL_EXPERIMENT_LOG.md` 中的快速验证仅 3M 参数 + 500 iter + 合成随机数据，accuracy 全部为 0。

**建议**：
- 在 README 中醒目标注 "目标指标" 和 "当前已验证指标"
- 维护一个 `STATUS.md`，用 ✅ / 🔄 / ⏳ 标注每项目标的实验完成状态







---

## 七、工程与可维护性

### 7.1 [P0] 统一训练抽象 BaseTrainer ✅ **已完成**

**现状**：四个训练脚本（`train_pretrain.py`、`train_sft.py`、`train_grpo.py`、`iterative_grpo.py`）原本各自独立实现 seed / device / distributed、模型加载、优化器、日志、checkpoint、AMP、梯度累积等逻辑。

**实现要点**：
- 新增 `training/trainer_base.py`：
  - `setup_distributed` / `cleanup_distributed`：基于 `RANK`/`WORLD_SIZE` 环境变量初始化 NCCL process group；
  - `set_seed`：统一 torch/numpy/random 种子，支持 rank offset；
  - `infer_device`：根据 local_rank 解析 `cuda:N`；
  - `build_amp_context`：构造 `torch.amp.autocast` + `GradScaler`（fp16 启用，bf16/cpu 关闭）；
  - `load_model_from_checkpoint`：按保存的 `model_type` 自动选择 BaselineGPT / ModernGPT 加载；
  - `CheckpointManager`：封装 `save_checkpoint` / `load_checkpoint`，自动 unwrap DDP、保存/恢复 EMA / scaler / scheduler / `resume_offset` / RNG 状态；
  - `BaseTrainer`：抽象类，子类仅需实现 `train()`，复用分布式、AMP、logger、checkpointing、配置持久化。
- `train_pretrain.py`、`train_sft.py`、`train_grpo.py` 已全部继承 `BaseTrainer`。

### 7.1b [P0] 统一 Config 系统 ✅ **已完成**

**现状**：存在 `config/*.yaml` 但脚本主要依赖 argparse，YAML 文件曾是 "文档参考"。

**实现要点**：
- 扩展 `utils/config.py`：
  - `load_yaml_config` 去除 BOM，递归展开 `${VAR}` 环境变量；
  - `parse_args_with_config` 支持嵌套 YAML + argparse CLI 覆盖，CLI 可用点号语法覆盖嵌套值（如 `--optimizer.lr 1e-4`）；
  - 返回 `NestedNamespace`，支持 `args.optimizer.lr` 属性访问与 `args.get('optimizer.lr')`；
  - 新增 `flatten` / `unflatten` / `to_dict` / `validate_keys` 等工具函数。
- `training/trainer_base.py` 的 `save_run_config` 与 `BaseTrainer.build_logger` 统一使用 `to_dict(args)` 持久化合并配置。

### 7.2 [P1] 日志系统失败降级与扩展 ✅ **已完成**

**现状**：`Logger.__init__` 仅 catch TensorBoard 初始化失败，wandb 失败会中断训练，且缺少文本样例、梯度/显存直方图、日志等级控制。

**实现要点**：
- `utils/logging.Logger`：TensorBoard 与 wandb 任一初始化失败均打印 warning 并降级，不中断训练；
- 新增 `log_text(tag, text, step)`：记录生成文本样例；
- 新增 `log_histogram(tag, values, step)`：记录梯度/参数分布；
- 新增 `log_grad_norms(model, step)`：总梯度 norm + 逐层 norm 直方图；
- 新增 `log_memory_stats(step)`：CUDA allocated / reserved / max allocated；
- 新增 `log_model_weights(model, step)`：逐层参数直方图；
- 新增 `log_level` 参数控制控制台输出级别；
- `BaseTrainer` 提供 `log_scalar` / `log_scalars` / `log_text` / `log_memory_stats` / `log_grad_norms` 便捷入口。

### 7.3 [P1] 补齐依赖 ✅ **已完成**

**现状**：`requirements.txt` 仅 6 行，无版本 pin，且缺少关键依赖（`datasets`, `huggingface_hub` 在 `prepare.py` 中使用但未列入）。

**实现要点**：
- `requirements.txt` 已扩展，新增：
  - `datasets`、`huggingface_hub`、`safetensors`（数据下载与缓存）；
  - `omegaconf`、`hydra-core`（配置管理）；
  - `pytest`（回归测试）。
- 仍建议后续添加 `environment.yml` / `pyproject.toml` / Dockerfile 以进一步标准化环境。

### 7.4 [P0] 单元测试覆盖 ✅ **已完成**

**现状**：整个项目零测试，重构风险高。

**实现要点**：
- 已扩展 `tests/test_bugfixes.py`（17 项），覆盖 KV Cache、优化器权重共享、EMA、DocBoundaryDataset、GRPO 批量化/梯度累积/IterativeGRPO、数据准备、算术数据集、规则奖励。
- 新增 `tests/test_config.py`：YAML 加载、环境变量展开、嵌套 YAML + CLI 覆盖、`NestedNamespace`、`flatten/unflatten`、`to_dict`、`validate_keys`。
- 新增 `tests/test_logger.py`：wandb/TensorBoard 失败降级、标量/文本/直方图/梯度 norm / 显存日志、close 幂等。
- 新增 `tests/test_trainer_base.py`：分布式辅助单进程行为、种子可复现、`infer_device`、AMP 上下文、`CheckpointManager` 全状态往返、`BaseTrainer` 初始化。
- 运行方式：`python -m pytest tests/ -q`（当前 57 passed，1 skipped）。

### 7.5 [P2] Checkpoint 管理缺乏生命周期策略 ✅ **已完成**

**现状**：每次 eval 都保存 `latest_ckpt.pt` 和 `best_ckpt.pt`（如果改善）。长时间训练下磁盘占用持续增长。

**实现要点**：
- `CheckpointManager` 新增 `keep_last_n` 参数；
- `best_ckpt.pt`、`final_*.pt`、`ema_ckpt.pt`、`latest_ckpt.pt` 受保护，永不删除；
- 其他 checkpoint 按 FIFO 保留最近 N 个，超出自动删除；
- `train_pretrain.py`、`train_sft.py`、`train_grpo.py` 均新增 `--keep_last_n` 参数。

**验证**：`tests/test_trainer_base.py::test_checkpoint_manager_keep_last_n`。

### 7.6 [P2] 种子管理健壮性 ✅ **已完成**

**现状**：`BaseTrainer` 原本按 rank 偏移全局种子，导致 DDP/FSDP 下各 rank 模型初始化不同，违反 DDP 参数一致性前提。

**实现要点**：
- `BaseTrainer.__init__` 现在所有 rank 使用相同 `args.seed` 调用 `set_seed()`，保证模型初始化一致；
- 新增 `make_worker_init_fn(base_seed, rank)` 工厂，为 DataLoader worker 分配确定性但互不冲突的种子；
- `train_pretrain.py` 的 `get_dataloader()`、`train_sft.py` 的 `DataLoader` 已接入 `worker_init_fn`；
- `train_grpo.py` 的合成数据生成按 `rank` 偏移，保证分布式下各 rank 数据池多样化。

**验证**：`tests/test_trainer_base.py::test_make_worker_init_fn_deterministic`、`tests/test_trainer_base.py::test_set_seed_reproducibility`。

### 7.7 [P3] 缺少 `pyproject.toml` 与可安装化 ✅ **已完成**

**现状**：项目是一个脚本集合，没有包结构。

**实现要点**：
- 新增 `pyproject.toml`：
  - 定义 `name="nanogpt-modern"`、`version="0.2.0"`、依赖、optional-dependencies（`eval`、`dev`）；
  - `[project.scripts]` 入口：`nanogpt-prepare`、`nanogpt-train-pretrain`、`nanogpt-train-sft`、`nanogpt-train-grpo`、`nanogpt-generate`、`nanogpt-eval-alignment`；
- 支持 `pip install -e .` 开发安装；
- 建议后续逐步将 `sys.path.insert` 替换为相对/绝对包内导入。





---

## 八、文档层

### 8.1 [P1] README 缺少 Quick Start 的可运行命令

**现状**：README 描述了丰富的架构和技术细节，但没有 "5 分钟跑起来" 的命令序列。

**建议**：
- 添加 "Quick Start" 章节：
```bash
pip install -r requirements.txt
python data/prepare.py --split train
python training/train_pretrain.py --model modern --max_iters 1000
python inference/generate.py --checkpoint out/pretrain/best_ckpt.pt --use_cache
```

### 8.2 [P1] API 文档缺失

**现状**：无 docstring 标准（`modern_gpt.py` 的 `RotaryEmbedding` 有部分注释，但不是 Google/NumPy 风格），无自动文档生成。

**建议**：
- 统一 docstring 风格（Google style），补全缺失的函数和类文档
- 可选：用 Sphinx + `sphinx.ext.autodoc` 生成 HTML 文档

### 8.3 [P2] 缺少 Architecture Decision Record (ADR)

**现状**：README 中的 FAQ 回答了部分设计决策，但没有独立的决策记录。

**建议**：
- 添加 `docs/adr/` 目录，记录每个关键决策：
  - `001-why-grpo-not-ppo.md`
  - `002-why-swiglu-1365-hidden.md`
  - `003-why-rope-not-alibi.md`





---

## 九、实验扩展方向

这些是需要改代码或大幅重构的较大改动，不属于 Bug 修复，但代表了系统演进的方向：

| 方向 | 说明 | 难度 | 预期收益 |
|------|------|------|----------|
| **多模态预训练** | 添加简单的 vision encoder（ViT-tiny）+ cross-attention 融合，在图文数据上做多模态训练 | 高 | 大幅拓展框架的展示价值 |
| **DPO (Direct Preference Optimization)** | 实现 DPO 对齐方案，与 GRPO 做系统对比（DPO 更简单，适合小规模场景） | 低 | 对齐方法矩阵完整 |
| **RLVR (RL with Verifiable Rewards)** | 扩展奖励函数到代码生成（pass@k）和数学证明（formal verification） | 中 | 展示 GRPO 在更复杂推理任务上的能力 |
| **LoRA 微调** | 添加 LoRA adapter 层，展示参数高效微调 | 低 | 降低对齐实验的显存门槛 |
| **蒸馏 (Knowledge Distillation)** | 用更大模型（如 GPT-2 Small）蒸馏到 50M ModernGPT，展示压缩效果 | 中 | 实用的模型压缩示例 |
| **Quantization 推理** | 添加 GPTQ / AWQ / bitsandbytes 量化推理，与 KV Cache 结合测试 | 中 | 推理优化选型完整 |

---

## 优先级汇总

| 优先级 | 数量 | 涵盖领域 |
|--------|------|----------|
| **P0** | 5 | GQA 实现补全、梯度累积、预训练 benchmark 评估、GRPO old/new policy 健壮性、数据 uint16 风险、单元测试 |
| **P1** | 12 | Attention 统一、Pre/Post Norm 消融、LR 调度器、EMA、SFT LR、Config 系统、指标统一、文档补全、Docker 环境、消融自动化 |
| **P2** | 11 | SwiGLU 对齐、Flash Attn 检测、GradScaler、DataLoader 改进、Early Stop、生成策略、数据验证、Reward shaping、KL 实现审查、Checkerpoint 管理、ADR |
| **P3** | 5 | MoE 探索、FSDP、batch 推理、pyproject.toml、Iterative RLHF |

**建议的改进路线图**：
1. **第一轮（P0）**：补齐安全性、正确性和可复现性的缺口
2. **第二轮（P1）**：提升工程健壮性和实验严谨性
3. **第三轮（P2）**：优化细节和扩展实验维度
4. **第四轮（P3）**：探索性实验和框架扩展
