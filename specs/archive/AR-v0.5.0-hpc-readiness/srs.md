# [AR-v0.5.0-hpc-readiness] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.5.0-hpc-readiness |
| AR 主题 | 高算力迁移就绪：推理正确性修复 + 训练性能强化 + 可移植性夯实 + 工程整理 |
| 关联 SR | 无（源自我检分析：四路深度分析报告，2026-10-08） |
| 关联 AR | AR-v0.4.0-experiments（本 AR 的 FR4/FR5/FR7 是 v0.4.0 推理实验 FR8/FR9 数据可信度的硬前提；v0.4.0 T007-T013 待迁移后执行） |
| 日期 | 2026-10-08 |
| 状态 | Draft |

## 1. 背景与目标

本机（Quadro RTX 5000，sm_75/16GB）实测无法高效完成 AR-v0.4.0 的 7 组 × 18k iters 消融训练（~71h+，已中止于 baseline step 4610）。决定迁移至**云租用 GPU 实例**（Linux，A100/H100 级，可能抢占式计费）执行实验矩阵。

四路深度分析（训练性能 / 推理性能 / 可移植性 / 工程质量）发现：

1. **推理子系统存在会直接产出错误实验数据的缺陷**：Paged KV 对 n_layer≥2 输出错误、generate 全路径 KV cache 上限锁死 block_size 使 NTK/needle 实验退化为 1024 滑窗、投机解码接受分布有偏（接受率虚高）、temperature=0 不做 greedy（评估不可复现）。
2. **「换机即跑」存在 P0 阻塞**：torch 未 pin（lock 中 cu12/cu13 静默分裂）、uv.lock 与 pyproject 失同步、api_server/pydantic 幽灵依赖、数据复现命令与实际数据不一致。
3. **断点续跑不完整**：编排器无组内 resume（崩溃组从 0 重训）、训练 --resume 的 RNG 状态存而不取、resume_offset 恒 0（数据重放）——云抢占场景下这些是成本与正确性的双重风险。
4. **高算力就绪度不足**：GQA 未走 SDPA enable_gqa（flash 后端降级）、MoE 每步 stack 全专家权重 + Python 逐专家循环、decode 每步全量 cat、多卡 DDP 数据不分片（数据并行名存实亡）、FSDP checkpoint 是死代码。
5. **工程债**：626 行 argparse 重复、load_model 7 份拷贝、black 行长两套真相、根目录 12 脚本散乱、BaselineGPT/ModernGPT 接口契约分裂。

**目标**：完成本 AR 后，在云租用 A100/H100 实例上「一条 runbook 从零到可训练 ≤1 小时」，且 AR-v0.4.0 剩余实验（T007-T013）可产出正确、可信、可复现的数据。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- 推理正确性修复（Paged KV / NTK cache 上限 / 投机解码分布 / greedy / 假计时）
- 依赖与环境固化（pyproject 声明补全、torch pin、uv.lock 同步、Docker/compose 现代化、CI 补检查）
- 数据迁移 runbook + 复现陷阱修复 + 本机迁移演练
- 断点续跑强化（编排器组内 resume、训练 resume 完整性、ckpt_interval、运行管理）
- 训练性能（配置级 + kernel 级：MoE 重写、KV decode 去 cat、int8 惰性反量化、GQA enable_gqa、数据路径、eval 瘦身、异步 checkpoint）
- 多卡（DDP 数据分片 + no_sync/static_graph；FSDP auto_wrap + checkpoint 修复）
- 连续批处理 iteration-level 重构 + api_server 接线
- 工程整理（根目录收编、死代码删除、args/load_model 提取、conftest、接口统一允许 breaking）
- lint/测试基建修复 + 理解力设计文档

**Out of Scope（本 AR 不做的）：**

- AR-v0.4.0 剩余执行型任务（T007-T013 的实际训练/实验/图表，待迁移后在云上执行）
- 7B 级以上大模型适配（FSDP CPU offload、quantized optimizer 等）
- 训练框架迁移（维持 DDP/FSDP，不引入 Megatron/DeepSpeed）
- flash-attn 外部包的强依赖（wrapper 已有，作为可选加速路径，不进核心依赖）
- Windows 多进程 spawn 竞态的根因修复（已用 num_workers=0 + 环境分层规避）

## 3. 功能需求

### 3.1 FR1 依赖与环境固化

**描述：** 消除「换机即跑」的依赖层阻塞：pyproject 补全声明、torch 版本 pin、uv.lock 重新同步、Docker/compose 现代化、CI 补一致性检查。

**触发条件：** 新机器执行 `uv sync` / `pip install -e .` / `docker compose up`。

**期望行为：**
- pyproject 新增 `server = ["fastapi", "uvicorn", "pydantic"]` 与 `validate = ["pydantic"]` extras（api_server.py:33-35 与 utils/config_models.py:14 的幽灵依赖）；requirements.txt 与 pyproject 双轨收敛（标注 requirements.txt 由 pyproject 派生或删除）
- torch 约束 pin 到 `~=2.5`（下限锁 2.5 以支持 SDPA enable_gqa），文档注明 cu12x 安装变体命令；消除 lock 中 cu12/cu13 按 Python 版本静默分裂
- `uv lock` 重新生成，provides-extras 覆盖全部 extras（当前缺 hf/plot）
- Dockerfile 基础镜像与 torch CUDA 变体对齐；docker-compose 废弃的 `runtime: nvidia` 改现代 `gpus` 写法
- CI 新增 `uv lock --check` 一致性 job 与 Docker build job

**异常处理：** pydantic 缺失时 utils/config.py 已有降级（保持）；测试用 `importorskip` 对齐降级行为。

**验收标准：**
- Given 干净 Linux 环境，When `uv sync --extra eval --extra quant --extra server`，Then 一键成功且 `python -c "import api_server"` 可导入
- Given CI，When push，Then `uv lock --check` 与 Docker build job 绿色
- Given pyproject 全部 extras，When 检查 uv.lock provides-extras，Then 完整一致

### 3.2 FR2 数据迁移 runbook 与复现陷阱修复

**描述：** 新环境数据就位的最短路径文档化 + 修复文档命令与实际数据的偏差。

**触发条件：** 迁移到云实例需要数据；或新机器按 README/EXPERIMENTS 命令重建数据。

**期望行为：**
- 修复复现陷阱：`data/prepare.py --split train` 默认 `max_docs=1_000_000` 与磁盘实际 7,913,597 docs 不一致 → 增加 `--full` 显式模式或在文档/`--help` 中明示默认值语义，README/EXPERIMENTS 复现命令更新为与实际数据一致的写法
- 新增 `docs/MIGRATION.md`：必带文件清单（train/val.bin + .idx + shuffle.idx ≈ 18.5GB，明确**不带** train_shards/owt_parquet 的 ~42GB 冗余）、环境重建命令、30 秒数据验证、50 步 GPU 冒烟、编排器启动命令
- 磁盘冗余处置：train_shards/（与 train.bin 重复）删除指引写入 runbook

**异常处理：** 数据校验失败时给出明确的 vocab/EOT/doc 数检查输出（复用 data/validate.py）。

**验收标准：**
- Given 本机，When 按 MIGRATION.md 在干净 venv/Docker 中演练环境搭建 + 数据验证 + 50 步冒烟，Then 全流程通过且耗时 ≤1 小时
- Given README/EXPERIMENTS.md，When 检查数据准备命令，Then 与磁盘实际数据规模语义一致（无「照文档跑出 1M docs 子集」陷阱）

### 3.3 FR3 断点续跑强化（云抢占刚需）

**描述：** 编排器组内自动 resume + 训练 resume 完整性补齐 + 抢占友好存档。

**触发条件：** 云实例抢占/进程崩溃后重启编排器或训练。

**期望行为：**
- `run_component_ablations.py`：组 manifest 非 completed 但 `latest_ckpt.pt` 存在且可读时，子进程命令自动注入 `--resume`；续跑前校验 ckpt 内 config 与组 flags 一致（n_kv_head/num_experts 等不匹配则拒绝续跑）；resume 加载失败自动降级为无 resume 重跑一次再记 error；summary 记录 `resumed_from_iter`
- 训练 resume 完整性：`CheckpointManager.load` 调用已实现但零调用的 `set_rng_state`（RNG 恢复）；训练循环真实推进 `resume_offset`（每 eval/checkpoint 时点累计已消费 chunk 数）；补存/恢复 `early_stop_counter`
- 新增 `--ckpt_interval`（独立于 eval_interval 的存档节奏，抢占场景可调小）
- 运行管理：日志轮转（RotatingFileHandler 到 out/<run>/logs/）；基于 metrics.jsonl 的心跳 watchdog（超过 3×log_interval×平均步时无新事件即告警）；编排器失败汇总支持 webhook 通知（环境变量注入 URL，缺失静默跳过）

**异常处理：** 坏 checkpoint 不阻塞矩阵（降级重跑）；通知渠道缺失不阻塞（静默）；与现有「监控失败降级为警告」哲学一致。

**验收标准：**
- Given 某组训练至 step N 被杀（模拟抢占），When 重跑编排器，Then 该组从 latest_ckpt 恢复而非从 0 重训，且 summary 含 resumed_from_iter
- Given 中断后 --resume 恢复的运行，When 对比不中断运行的 loss 轨迹，Then 恢复点后首个 eval 的 val loss 与不中断运行同点值差 ≤ 2× 该点前历史 eval 波动幅度（且抽样断言数据序列不重放）
- Given ckpt 内架构与组 flags 不匹配，When 编排器尝试续跑，Then 拒绝并按无 resume 处理且记录告警
- Given `--ckpt_interval 200`，When 训练 500 步，Then latest_ckpt 至少更新 2 次

### 3.4 FR4 推理正确性修复（v0.4.0 实验数据可信硬前提）

**描述：** 修复四个会产出错误/不可复现实验数据的推理缺陷 + 假计时。

**触发条件：** 运行 KV cache/投机解码/needle/NTK 相关路径。

**期望行为：**
- Paged KV 双 bug：`get_cache()` 逐层返回各自 block（修复所有层共享 layer 0）；`update()` 的 block table 追加逻辑修复逐层重复 append；补 n_layer=2 的 ring/paged 等价回归测试（当前测试 n_layer=1 恰好掩盖）
- generate 全路径（modern_gpt.py 四处）`max_cache_len` 改用 `effective_block_size`（NTK factor 生效于生成路径），或统一走 `KVCacheManager.from_config`
- 投机解码分布修正：greedy draft 的 p_d 取 point mass（=1）；拒绝残差 `norm(max(0, p_t − q))` 作用于完整 draft 分布；验证与额外 token 采样温度统一；新增输出分布等价性统计测试（KL 散度 / 逐位置对比，draft==target 时必须无偏）
- `temperature=0` → argmax greedy（modern_gpt.py、generate_utils.py、continuous_batching.py 三处）
- `profile_inference.py` 修复假 per-token 计时（step() 挂入真实 decode 回调）

**异常处理：** NTK cache 扩容受显存约束失败时给出明确错误（而非静默滑窗截断）。

**验收标准：**
- Given n_layer=2 的 tiny 模型，When 对比 paged 与 ring cache 生成输出，Then 逐 token 一致
- Given NTK factor=4 的模型，When 生成 2048 token，Then KV cache 容量按 effective_block_size 分配（不再截断到 1024）
- Given draft==target 的投机解码，When 采样 ≥10k token 统计输出分布，Then 与直接采样分布 KL < 1e-3（无偏）；greedy draft==target 时接受率 = 1.0（不再系统性虚高）
- Given temperature=0，When 重复生成两次，Then 输出完全一致（greedy 确定性）

### 3.5 FR5 KV cache decode 路径 kernel 级改造

**描述：** 消除 decode 每步全量 `torch.cat`（ring cache 的 O(1) 语义被 attention 层 cat 抵消）+ int8 全量反量化物化。

**触发条件：** 任意带 KV cache 的 decode 步。

**期望行为：**
- decode 步将 ring buffer 的 K/V 直接喂 SDPA（有效前缀 + 新 token 分两次调用拼接输出，或等价零拷贝方案）；条件可用时接线已有 `flash_attention_with_kvcache` wrapper（可选路径）
- int8 cache 按块惰性反量化（不再每步全量物化），显存阶梯图与实测峰值一致
- `build_sliding_window_mask` 的 Python O(T²) 双循环向量化

**异常处理：** flash wrapper 不可用（未装包/版本不符）时回退纯 SDPA 路径。

**验收标准：**
- Given 固定种子 tiny + 50M 模型，When 对比改造前后 cache/no-cache 生成输出，Then 逐 token 一致（正确性不变）
- Given 4096 上下文 decode 基准，When 对比改造前后，Then 单步 decode 时间中位数改善 ≥1.5x（本机可测）且对 cache 长度 {512,1024,2048,4096} 各段的 max_memory_allocated 线性回归斜率不显著（p>0.05，不再随 cache 长度线性增长）
- 性能对比数据写入实验记录（out/experiments/ 或 docs 基准表）

### 3.6 FR6 MoE 向量化重写

**描述：** 消除 MoE 每步 `torch.stack` 全专家权重（~260MB/step 纯带宽浪费）与 Python 逐专家循环（compile 图断裂 + 低利用率）。

**触发条件：** num_experts>1 的训练/推理 forward。

**期望行为：**
- 专家权重预注册（不再每 forward stack）
- token 路由向量化：`index_add`/排序分桶按专家批量 GEMM，消灭逐专家 Python 循环与 `numel()==0` 数据依赖分支
- load balance loss 语义保持不变
- torch.compile 兼容（无图断裂、无逐 shape 重编译）

**异常处理：** 单专家（num_experts=1）路径保持原 SwiGLU 快速路径不变。

**验收标准：**
- Given 固定种子与相同输入，When 对比重写前后 MoE forward 的 loss 与 load balance loss，Then atol=1e-5 一致
- Given num_experts=4 的训练基准，When 对比前后 step 时间，Then MoE 层前向+反向 ≥2x（本机可测），对比表写入实验记录
- Given torch.compile 开启，When MoE 训练 100 步，Then 无持续 recompile（`torch._dynamo` 计数不随步数增长）

### 3.7 FR7 连续批处理 iteration-level 重构

**描述：** 当前是伪 continuous（static batching + 完成序列继续喂 EOS + Prefix Cache/compile 死代码 + 右 padding cache 污染）。重构为真 iteration-level 调度。

**触发条件：** 并发推理请求（bench_continuous_batching / api_server）。

**期望行为：**
- 每 decode step：收割 finished 槽位 → 新请求 prefill 插入（iteration-level 调度）
- 修复右 padding + cache 污染（pad 位置 KV 不得进 cache 或 decode 带 mask）；修复短 prompt 序列的 RoPE 位置偏移（按各自真实长度）
- Prefix Cache：实现注入逻辑或删除占位（二选一，不留死代码）
- 批量采样张量化（消灭 per-request Python 循环与逐序列 `.item()` 同步）；RequestQueue 加锁或移除 thread-safe 声明
- `_init_decode_compile` 死代码：接线或删除
- api_server：接入新调度器（asyncio.to_thread/线程安全队列），补齐文档声称的 `--continuous_batching` flag 或删除声称

**异常处理：** 单请求退化路径与 batch 路径输出一致。

**验收标准：**
- Given 并发 16 请求（长度不齐），When 运行重构后调度器，Then 已完成请求立即释放槽位、新请求在下一个 step 边界插入（日志/计数可验证）
- Given 不等长 prompt batch，When 对比逐条单独生成，Then 输出一致（padding 修复后无质量劣化）
- Given 并发 1-16 基准，When 对比 static batching，Then 并发 16 下平均延迟降低 ≥30%，且吞吐/延迟曲线可测出 continuous 语义差异

### 3.8 FR8 训练配置级性能优化

**描述：** 高算力上低成本高回报的训练性能项。

**触发条件：** A100/H100 上执行训练。

**期望行为：**
- GQA 走 SDPA 原生 `enable_gqa=True`（torch≥2.5 探测，回退现有 broadcast 路径）；GQA 惰性探测移出首次 forward（compile 图断裂修复）
- 数据路径：`non_blocking=True`（pin_memory 生效）、`persistent_workers`、num_workers 环境分层（Linux 默认 >0，Windows 默认 0，配置显式覆盖）
- eval 瘦身：eval batch 放大、多卡下 allreduce 平均（非 master 不再空转）、去逐 iter `.item()` sync
- 训练循环去每 micro-step all_reduce/.item()（累积 tensor 定期同步）
- EMA 改 `torch._foreach_*` 批量更新（200+ kernel → 2-3 个）
- TF32 设置推广到全部 5 个训练器（当前仅 pretrain）
- 异步 checkpoint：staged CPU 副本 + 后台线程落盘（保持 tmp+rename 原子性）
- DDP accum 加 `no_sync()`、`static_graph=True`、`gradient_as_bucket_view=True`

**异常处理：** enable_gqa 探测失败自动回退（复用现有 probe 基建）；异步保存失败回退同步路径并告警。

**验收标准：**
- Given GQA 配置训练，When 检查 SDPA 后端命中，Then flash 后端不再因 5-D broadcast 降级（可用 torch.backends 日志/探针验证）
- Given 本机 100 步基准，When 对比优化前后 tok/s，Then tok/s ≥ 基线 × 0.98（无退化容差）且 eval 阶段耗时占 wall time 比例下降 ≥5 个百分点
- Given 全部 5 个训练器，When grep TF32 设置，Then 全覆盖
- Given 任一训练器 --resume，When 恢复后继续训练，Then 行为与优化前语义一致（回归测试）

### 3.9 FR9 多卡 DDP 修复 + FSDP 补齐

**描述：** DDP 修复数据正确性（rank 不分片）与调优；FSDP 从「生产不可用」补齐到可用。

**触发条件：** torchrun 多卡启动训练。

**期望行为：**
- `MemmapDataset` 增加 rank-aware 分片（每 rank 消费互不重叠的全局 shuffle 切片；resume_offset 语义随分片更新并在 ckpt 校验一致）
- DDP：accum 时 `no_sync()`、`static_graph=True`（结构固定）、`gradient_as_bucket_view=True`
- FSDP：`auto_wrap_policy`（per-layer 分片，通信/计算重叠）、`use_orig_params=True`（compile 兼容）、compile→FSDP 包裹顺序按官方推荐、修复 FSDP checkpoint（`_get_model_state_dict` 死代码接线或改用 `torch.distributed.checkpoint`）
- 分布式单元测试：rank 分片互斥性、DDP/FSDP 保存-恢复 round-trip（可用单进程多组模拟或 torchrun --nproc_per_node=2 冒烟）

**异常处理：** 单卡运行零行为变化（分布式路径仅 torchrun 激活）。

**验收标准：**
- Given 2 rank 模拟，When 各 rank 迭代数据集，Then 消费的 chunk 集合互斥且并集为全局（抽样验证）
- Given FSDP 保存的 checkpoint，When 加载恢复，Then 权重与保存前一致（无 `_fsdp_wrapped_module.` 前缀泄漏）
- Given 单卡运行，When 全量测试，Then 零回归

### 3.10 FR10 根目录收编与死代码清理

**描述：** 根目录 12 个 .py 脚本归类；删除确认无引用的死文件。

**触发条件：** 仓库结构维护。

**期望行为：**
- 导出/互操作 4 个（export_gguf/export_to_hf/export_to_onnx/load_from_hf）→ `tools/` 或 `export/` 包
- 编排/一次性脚本处置：`run_component_ablations.py` 保持根目录或迁 scripts/（**pyproject [project.scripts] 与测试引用同步**，test_component_ablations 的路径契约更新）；`run_ablations.py`（旧版）评估归并或标注 deprecated；`generate_experiment_log.py`、`run_inference_ablation.py`、`logs_prepare*.txt` 确认零引用后删除
- profile_*.py 2 个 → tools/（顺带修 sys.path hack）
- 迁移后 CI/pre-commit/文档路径全部同步；根目录只留 README/pyproject 等配置文件

**异常处理：** 任何删除前全库引用检查（grep）；保留 git 历史可恢复。

**验收标准：**
- Given 收编后仓库，When `pytest`/CI/pre-commit 全量运行，Then 绿色（无路径断裂）
- Given pyproject console scripts，When `nanogpt-ablations --help`，Then 仍可用
- Given 根目录，When 列出 .py，Then 仅剩入口必要性文件（或为零）

### 3.11 FR11 重复代码提取

**描述：** 消除三大重复税：args 复制、load_model 拷贝、测试样板。

**触发条件：** 代码维护/新训练脚本编写。

**期望行为：**
- 新增 `training/args.py::add_common_trainer_args()`：5 个训练脚本 626 行 argparse 中 23 个重复 flag 收敛（~350 行净删除），各脚本仅保留 stage 特有参数
- `load_model` 逻辑 7 份拷贝统一到 `load_model_from_checkpoint`（trainer_base.py 既有权威实现），导出场景加 eval/strict 变体
- 新增 `tests/conftest.py`：`tiny_modern_config`/`tiny_model` fixture + `slow`/`perf` marker 注册；7 处 `_build_tiny` 类 helper 与 85 处 `ModernGPTConfig(` 手工构造逐步收敛（重点文件优先）
- `tests/test_bugfixes.py`（663 行大杂烩）按主题拆分，删除文件尾手动 runner

**异常处理：** 无（纯重构，行为不变）。

**验收标准：**
- Given 重构后全量测试，Then 与基线等量通过（338 collected 基线）且运行时长 ≤ 1.1× 基线
- Given 任一训练脚本 --help，Then 公共 flag 语义与重构前一致（快照对比）
- Given grep `ModernGPTConfig(`，Then 测试文件中的手工 tiny 构造计数 < 40（conftest fixture 覆盖主要用例）

### 3.12 FR12 模型接口统一（允许 breaking）

**描述：** v0.5.0 大版本号下统一 BaselineGPT/ModernGPT 契约，消除静默 no-op。

**触发条件：** 任何调用两个模型 forward/generate 的代码。

**期望行为：**
- BaselineGPT.forward 返回 `(logits, loss, None)` 3-tuple（与 ModernGPT 对齐），全部调用点（trainers/evaluation/export）同步更新，`len(batch)==3` 式分支简化
- BaselineGPT 的 `attention_mask`（当前静默丢弃）与 generate 的 `top_p/use_cache/eos_token_id` no-op 参数：实现或显式 `raise NotImplementedError`/warning + docstring 声明降级行为
- 下游反射 hack（`run_full_evaluation.py` 用 `__code__.co_varnames` 探测能力）改为显式能力查询
- CHANGELOG 记录 breaking change 与迁移指引

**异常处理：** no-op 参数显式 warning（一次性，不刷屏）。

**验收标准：**
- Given 统一后全部调用点，When grep `logits, loss =`（2-tuple 解包）与 `model(...)[0]`，Then 零残留
- Given packing 数据（document_ids）喂 BaselineGPT，When forward，Then 显式 warning/报错（不再静默丢弃 mask）
- Given 全量测试 + CI，Then 绿色

### 3.13 FR13 lint/测试基建修复

**描述：** 消除「配置的配置」矛盾与脆弱测试。

**触发条件：** CI/pre-commit/本地 lint。

**期望行为：**
- `[tool.black] line-length = 100`（消除 pre-commit 100 vs CI 88 两套真相）+ 全库一次性格式化对齐
- 补 `[tool.ruff]` 配置（line-length=100，select 至少 E,F,I,B,UP）
- mypy `python_version` 与 `requires-python >= 3.9` 对齐
- `test_kv_cache_benchmark` 绝对时间断言（<60s，本机 65s 红灯）改相对断言（speedup 比值）+ `@pytest.mark.perf` 可 deselect
- 文档数字刷新：README 徽章 230+/CI 注释 234 vs 实际 338；TECH_REPORT 行号锚点过期处修正；README 目录树补漏列脚本

**验收标准：**
- Given CI，When black --check + ruff check，Then 绿色且行长规则唯一（100）
- Given 本机，When `pytest -m "not perf"`，Then 全绿（红灯测试修复）；`pytest -m perf` 可单独运行且断言机器无关
- Given README/CI/TECH_REPORT，When 检查测试数与行号引用，Then 与实际一致

### 3.14 FR14 理解力交付物：设计文档与代码导航

**描述：** 以文档形式沉淀框架理解（「夯实技术实力和框架理解力」的交付物）。

**触发条件：** 新人/未来的自己理解框架。

**期望行为：**
- 新增 `docs/ARCHITECTURE.md`（训练基座设计文档）：BaseTrainer 模板方法链与生命周期、DDP/FSDP/AMP/EMA/Checkpoint 机制、监控降级哲学、5 个训练器的差异矩阵
- `model/modern_gpt.py`（2627 行）：补 13 个缺失 docstring 的公共方法 + 分区导航注释（架构组件/KV cache/投机解码/生成四大区）
- GRPO 家族对比笔记（train_grpo vs iterative_grpo vs train_dpo 的算法差异心智模型），入 docs/ 或 TECH_REPORT 章节
- （AR-v0.4.0 衔接）MIGRATION.md 与 ARCHITECTURE.md 交叉引用

**验收标准：**
- Given ARCHITECTURE.md，When 按文档描述对照 trainer_base.py 代码，Then 钩子链/生命周期/降级点描述准确（抽查无漂移）
- Given modern_gpt.py，When 检查公共方法，Then docstring 覆盖 100%
- Given GRPO 笔记，When 对照三份训练代码，Then 算法差异描述准确（advantage 计算/参考更新/KL 处理三维度）

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 回归 | 测试套件 | 全量通过（基线 338 collected，336 passed/2 skipped + 本 AR 新增）；无既有测试被删或弱化 |
| 等价性 | kernel 级改造 | 固定种子下 MoE loss / KV decode 输出与原实现 atol=1e-5 一致 |
| 性能 | MoE 前向+反向 | ≥2x（本机可测代理指标） |
| 性能 | decode 步（4096 ctx） | ≥1.5x 且显存分配不随 cache 长度线性增长 |
| 可移植 | Linux 干净环境 | uv sync + 数据就位 + 冒烟 ≤1 小时（MIGRATION.md 演练验证） |
| 覆盖率 | 新增模块 | 行覆盖 ≥85%（沿用 v0.4.0 口径） |
| 兼容性 | Python | ≥3.9 保持（CI 3.9/3.11 双版本） |

## 5. 约束与假设

**约束：**
- 云租用实例可能抢占 → FR3 的存档/恢复设计按「最多丢一个 ckpt_interval」标准
- v0.5.0 语义化大版本 → FR12 允许 breaking，但必须 CHANGELOG + 迁移指引
- 不引入重型强依赖（flash-attn/triton kernel 手写等仅可选路径）
- 全程在本机（RTX 5000/Windows）开发验证，性能指标用本机代理基准 + 云端复核两段式
- Phase A（FR1-FR4）优先于 Phase B（FR5-FR9）优先于 Phase C（FR10-FR14）；正确性修复不可被性能目标妥协

**假设：**
- 目标环境：Linux + NVIDIA GPU（sm_80+），torch 2.5-2.8 cu12x，单机 ≤8 卡
- 数据集 bin（~18.5GB）随仓库迁移，不依赖 HF Hub 在线重建
- AR-v0.4.0 的 srs/design/tasks 保持有效，其实验任务在迁移后按本 AR 修复后的代码执行

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| iteration-level 调度 | 每 decode step 边界动态收割/插入请求的批处理调度（vs static batching 整批跑完） |
| effective_block_size | NTK 缩放后的等效上下文长度（block_size × factor） |
| rank 分片 | 多卡数据并行下各 rank 消费互不重叠数据切片的正确性要求 |
| 抢占 | 云spot实例被回收，进程被杀，需从最近 checkpoint 恢复 |
| 幽灵依赖 | 代码 import 但 pyproject/requirements 未声明的依赖 |
| point mass | 贪心 draft 提议分布的数学语义：被选 token 概率为 1 |
