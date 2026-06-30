# nanoGPT-Modern 变更日志

> 汇总从 v0.1.0 到 v0.3.0 的系统改进、bug 修复与质量保障。

---

## 版本概览

| 版本 | 时间 | 关键里程碑 |
|------|------|-----------|
| v0.1.0 | 2026-05 | 经典 GPT-2 基线（BaselineGPT） |
| v0.1.5 | 2026-05 | 引入 ModernGPT（RMSNorm + SwiGLU + RoPE） |
| v0.2.0 | 2026-06 | 完整三阶段流水线 + 23 项优化落地 |
| **v0.3.0** | **2026-06** | **全维度 42 项优化落地 + 244 回归测试全绿** |

当前回归测试：**244 passed, 1 skipped**。

---

## v0.3.0 重大变更（从 2026630 优化方向全景报告落地）

### 模型架构优化

| 功能 | 状态 | 说明 |
|------|------|------|
| MoE grouped GEMM | ✅ | 专家权重堆叠 + `bmm` 向量化，消除 Python for-loop 路由 |
| KV Cache 量化（INT8/FP8） | ✅ | `QuantizedKVCacheManager` 量化存储 + 反量化计算，显存节省 50% |
| FlashAttention 自动调度 | ✅ | SDPA 后端自动探测 + 自动回退 |

### 训练系统优化

| 功能 | 状态 | 说明 |
|------|------|------|
| 全局数据 Shuffle | ✅ | 预计算 `.shuffle.idx` 随机索引，训练时全局随机化 |
| 数据质量 2.0 | ✅ | `QualityScoreFilter` 规则/FastText 评分 + 分层采样 |
| 增量去重 | ✅ | `MinHashDeduplicator` 增量签名持久化与追加比对 |
| 多语言数据管道 | ✅ | `MultilingualTokenizer`（tiktoken + SentencePiece）+ 按语言权重采样 |
| 代码数据管道 | ✅ | `CodeDataset` 本地目录/The Stack + AST 语法过滤 |
| 数学 CoT 增强 | ✅ | `ChainOfThoughtDataset` GSM8K 风格逐步推理（`<reasoning>` + `<answer>`） |
| 对话模板系统 | ✅ | `ChatTemplate` 抽象（chatml / llama-2 / gemma）+ system prompt 注入 |
| 监控告警体系 | ✅ | `LossSpikeDetector`（>3σ 触发 LR 降低）、`GradientNormMonitor`、`ThroughputMonitor` |
| 配置 Pydantic 校验 | ✅ | `ConfigValidator` 运行时结构化校验（`n_head % n_kv_head == 0` 等） |

### 推理与服务优化

| 功能 | 状态 | 说明 |
|------|------|------|
| 连续批处理 | ✅ | `RequestQueue` + `ContinuousBatchScheduler` + 动态 batch 拼接 |
| Prefix Cache | ✅ | 共享 prompt 的 KV Cache 跨请求复用 |
| 投机解码 batch > 1 | ✅ | `_generate_speculative_batched` 树形解码，支持 batch 扩展 |
| 注意力可视化 | ✅ | `AttentionVisualizer` heatmap + `LogitLens` 各层 top-k 预测词 |
| 性能 Profiler | ✅ | PyTorch Profiler Chrome trace 导出 + 显存分层报告 |
| ONNX 导出 | ✅ | `export_to_onnx.py` 动态轴导出 + ONNX Runtime 验证 |
| API 服务 | ✅ | `api_server.py` FastAPI 兼容 OpenAI API（`/v1/completions` + `/v1/chat/completions`） |

### 对齐与 RL 优化

| 功能 | 状态 | 说明 |
|------|------|------|
| DPO 完整训练 | ✅ | 从 GRPO 自动生成偏好对 + 完整训练循环 + 胜率评估 |
| 胜率评估 | ✅ | `eval_win_rate.py` policy vs reference 胜率对比 |

### 评估与可观测性

| 功能 | 状态 | 说明 |
|------|------|------|
| 标准 Benchmark 集成 | ✅ | `benchmark_suite.py` MMLU / ARC / HellaSwag / Winogrande / HumanEval / GSM8K |
| 长上下文测试 | ✅ | `needle_in_haystack.py` Needle-in-Haystack 验证 |

### 工程与 DevOps

| 功能 | 状态 | 说明 |
|------|------|------|
| CI/CD | ✅ | GitHub Actions（pytest + mypy + black + ruff + 50-step smoke test） |
| 容器化 | ✅ | Dockerfile 多阶段 + docker-compose（训练 + TensorBoard + API） |
| 预提交钩子 | ✅ | `.pre-commit-config.yaml`（black + ruff + mypy + pytest） |

---

## v0.2.0 重大变更

### 模型架构

| 功能 | 状态 | 说明 |
|------|------|------|
| GQA grouped-broadcast 零拷贝 | ✅ | Q reshape `[B,n_kv,rep,T,D]` + KV unsqueeze `[B,n_kv,1,S,D]`，SDPA 自动广播，零 KV 拷贝 |
| FlashAttention 显式集成 | ✅ | 封装 `flash_attn_func` / `varlen_func` / `with_kvcache`，自动回退 SDPA/eager |
| QK-Norm + Attention Temperature | ✅ | 可选 per-head RMSNorm 与温度缩放 |
| RoPE NTK-aware 外推 | ✅ | 支持 `rope_scaling` 长度外推 |
| Sliding Window Attention | ✅ | 训练与推理路径均支持窗口注意力 |
| MoE 负载均衡 | ✅ | `num_experts > 1` 时启用 aux loss 与容量限制 |
| Multi-Token Prediction (MTP) | ✅ | `n_future` 个未来 token 预测头 |
| Ring Attention | ✅ | 纯 PyTorch blockwise 序列并行 |

### 训练系统

| 功能 | 状态 | 说明 |
|------|------|------|
| `BaseTrainer` 统一抽象 | ✅ | pretrain/sft/grpo/dpo 统一继承，消除重复代码 |
| AMP (bf16/fp16 + GradScaler) | ✅ | 所有训练阶段启用 |
| 梯度累积 | ✅ | 预训练/SFT/GRPO 均支持 |
| LR Scheduler | ✅ | cosine / linear / wsd / constant + warmup |
| EMA | ✅ | shadow weights 保存/恢复 |
| Early Stopping | ✅ | 基于 val loss 的耐心机制 |
| DDP / FSDP | ✅ | 统一分布式包装 |
| 完整 checkpoint | ✅ | 模型 + 优化器 + scaler + scheduler + EMA + RNG + resume_offset |
| Checkpoint 生命周期 | ✅ | `--keep_last_n` 自动清理 |
| 种子管理 | ✅ | 全局统一 + DataLoader worker 确定性种子 |
| 梯度检查点 | ✅ | `--gradient_checkpointing` 降低激活显存 |

### 推理系统

| 功能 | 状态 | 说明 |
|------|------|------|
| 静态 KV Cache | ✅ | `KVCacheManager` 预分配 ring buffer，避免 `torch.cat` |
| Paged KV Cache | ✅ | `PagedKVCacheManager` block-table API，兼容 ring buffer |
| `torch.compile` 生成 | ✅ | `compile=True/"fullgraph"`，失败自动回退 eager |
| Speculative Decoding | ✅ | `draft_model` 参数支持 draft-then-verify（batch size 1） |
| 生成策略扩展 | ✅ | top_p, repetition_penalty, eos mask |
| Batch 推理 | ✅ | finished mask 逐序列提前终止 |
| CUDA Events 计时 | ✅ | prefill/decode 阶段分离 |

### 数据管道

| 功能 | 状态 | 说明 |
|------|------|------|
| 多进程 shard 化预处理 | ✅ | `datasets.map(batched=True, num_proc=...)`，不持全量 token |
| 数据质量管道 | ✅ | 过滤(filter) / 去重(dedup) / 混合(mixer) |
| Packing + 跨文档 mask | ✅ | `PackingDataset` 产出 `document_ids` |
| MemmapDataset shuffle buffer | ✅ | chunk-level 缓冲乱序 |
| DocBoundaryDataset | ✅ | EOT 文档边界截断 + `resume_offset` |
| 算术数据预编码 | ✅ | `ArithmeticDataset(pre_tokenize=True)` |

### RL 对齐

| 功能 | 状态 | 说明 |
|------|------|------|
| GRPO 批量化 | ✅ | 按长度分组 batch generate + 单次 forward 计算 logprobs |
| GRPO 梯度累积 | ✅ | 有效 batch size 线性扩展 |
| GRPO LR 调度 | ✅ | cosine warmup + decay |
| Dropout Guard | ✅ | 默认拒绝 `dropout > 0` 的 SFT checkpoint |
| Iterative GRPO | ✅ | 继承 GRPOTrainer，支持 EMA ref 更新与拒绝采样 SFT |
| DPO / IPO / KTO | ✅ | 偏好对齐损失工具 + `DPOTrainer` 骨架 |
| 规则奖励细化 | ✅ | 格式分 + 过程分 + 连续正确性分 |

### 评估与导出

| 功能 | 状态 | 说明 |
|------|------|------|
| 标准化 Benchmark | ✅ | 本地 PPL + 可选 lm-eval 下游任务 |
| 对齐评估批量化 | ✅ | `generate_by_length` 按长度分组，KL 仅计算 response token |
| HuggingFace 兼容 | ✅ | `model/hf_model.py` + `export_to_hf.py` / `load_from_hf.py` |
| 量化 | ✅ | INT8 / bitsandbytes 8-bit/4-bit |
| GGUF 导出 | ✅ | 内置 F32/F16/Q8_0 writer，`export_gguf.py` 一键导出 |
| 消融自动化 | ✅ | `run_ablations.py` 训练/推理矩阵 |

### 工程与配置

| 功能 | 状态 | 说明 |
|------|------|------|
| 统一配置系统 | ✅ | YAML + argparse 嵌套覆盖、环境变量展开、`NestedNamespace` |
| Hydra / OmegaConf | ✅ | `*_hydra.py` 入口 + `config/hydra/` 配置组合 |
| 日志失败降级 | ✅ | wandb/TensorBoard 失败不中断训练 |
| 依赖补齐 | ✅ | `requirements.txt` + `pyproject.toml` |
| 可安装化 | ✅ | `pip install -e .` + `[project.scripts]` |
| 类型注解 | ✅ | 核心模块通过 `mypy` 检查 |

---

## 关键 Bug 修复

| 问题 | 修复 |
|------|------|
| KV Cache dtype 不匹配 | `cache.init_cache` 使用模型参数 dtype |
| KV Cache 超长生成不一致 | 静态 ring buffer + `start_pos` 绝对位置追踪 |
| 优化器权重共享重复更新 | `configure_optimizers` 去重 |
| EMA 未更新 | 每次 `optimizer.step()` 后调用 `update_ema` |
| GRPO old/new policy dropout 偏差 | 默认拒绝 dropout > 0，提供 `--allow_dropout` |
| `inference/generate.py` 采样参数未透传 | `benchmark()` 接收 temperature/top_k/top_p |
| `eval_alignment.py` KL 含 prompt token | 仅对 response target positions 计算 KL(ref‖policy) |

---

## 测试覆盖

```bash
python -m pytest tests/ -q
```

覆盖模块：config、logger、trainer_base、bugfixes、attention_utils、GQA broadcast、FlashAttention、GRPO、generate compile、speculative decode、HF compat、quantization、GGUF export、data quality、packing、DPO、LR scheduler、MoE load balance、MTP、Paged KV Cache、RoPE NTK、sliding window、determinism、mypy。

---

## 已知限制与未来方向（v0.3.0 后）

| 方向 | 优先级 | 说明 |
|------|--------|------|
| 50M 参数完整训练验证 | P0 | 当前实验停留在 3.3M 快速验证模式，需验证 50M 完整收敛 |
| 3D 并行验证（TP+PP+DP） | P1 | 在 8×A100 上验证张量/流水线/数据并行组合 |
| 序列并行（RingAttention）生产级 | P1 | 当前为纯 PyTorch 实现，需验证 >100K 上下文 |
| 权重量化感知训练（QAT） | P1 | 预训练/SFT 阶段引入伪量化节点 |
| 结构化剪枝 | P2 | SwiGLU FFN 隐藏维度重要性评分剪枝 |
| 注意力头剪枝 | P2 | 基于 GQA 的 n_rep 比例移除冗余 Query head |
| 渐进式训练 | P2 | 从短序列（256）到长序列（4096）逐步扩展 |
| Mamba / SSM 混合 | P3 | 在局部注意力层引入 selective SSM 层 |
| 线性注意力 | P3 | 基于 kernel feature map 的近似注意力 |
| Web UI（Gradio/Streamlit） | P3 | 浏览器端交互式 demo |
| 边缘/移动端部署 | P3 | llama.cpp 适配 + Android/iOS 推理 |
| MLX 导出（Apple Silicon） | P3 | macOS 统一内存加速推理 |

> 原始优化方向全景报告（2026630）已归档到本文件（v0.3.0 部分）及技术白皮书 Roadmap。

---

## 参考

- 完整技术白皮书：`docs/TECH_REPORT.md`
- 实验记录与复现命令：`docs/EXPERIMENTS.md`
- 架构决策记录：`docs/adr/`
