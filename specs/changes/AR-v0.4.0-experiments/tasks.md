# [AR-v0.4.0-experiments] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.4.0-experiments |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（待生成） |
| 创建日期 | 2026-10-07 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | FR1 TrainerMetricsRecorder：metrics.jsonl 逐事件写入 + experiment manifest（环境/超参/种子/产物指针），接入 BaseTrainer 与五个训练脚本 | - | passing | TDD：先写测试 |
| T002 | FR2 监控接入：LossSpikeDetector/GradientNormMonitor 接入训练循环，事件入 metrics 流，CLI 开关 | T001 | passing | 不破坏既有测试 |
| T003 | FR3a ModernGPT `rope=off` 开关（可学习绝对 PE，KV cache/生成路径正确）+ CLI/YAML/Hydra/Pydantic | - | passing | 与 T004 可并行 |
| T004 | FR3b ModernGPT `ffn_variant=gelu4x` 开关 + 非法组合校验 | - | passing | 与 T003 可并行 |
| T005 | FR4 run_component_ablations.py：7 组配置、断点续跑、失败隔离、汇总表（JSON+Markdown） | T001,T003,T004 | passing | 用小模型做集成测试 |
| T006 | FR5 全量 OpenWebText 数据准备：下载、bin 化、validate、全局 shuffle 索引 | - | passing | 长时执行任务，2-5h |
| T007 | FR6+E7 执行 7 组 50M × 18k iters 消融训练（baseline/modern-full/no-rope/no-swiglu/mha/moe/qk-norm） | T005,T006 | in_progress | 跨多会话，断点续跑；已启动，见 2026-10-08 会话记录 |
| T008 | FR8 推理实验：KV cache 矩阵、投机解码、连续批处理、torch.compile，结果 JSON | T007 | pending | 依赖 modern-full checkpoint |
| T009 | FR9 长上下文实验：NTK ppl-vs-长度曲线、Needle heatmap 数据 | T007 | pending | 依赖 modern-full checkpoint |
| T010 | FR10 对齐实验：SFT → GRPO(G4/G8) → DPO 胜率，全部在 50M checkpoint 上 | T007 | pending | 依赖 modern-full checkpoint |
| T011 | FR11 图表生成器：统一样式模块 + ≥10 类图（loss/消融/KV/投机/批处理/NTK/needle/reward/胜率），SVG+PNG 双输出 | T007,T008,T009,T010 | pending | 可在 T008-T010 完成前先建样式模块与 loss 图 |
| T012 | FR12 文档：docs/FIGURES.md 图库 + EXPERIMENTS.md 全量重写 + README 嵌入真实图表 | T011 | pending | |
| T013 | FR13 质量收口：全部新旧测试通过、mypy/black/ruff/CI 绿色 | T012 | pending | |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-07 T001 完成（FR1）

- **Red**：`tests/test_experiment.py` 22 个用例先行（collect_env 字段/git 缺失降级、record_step/eval/event/通用事件、ISO 时间戳、append/覆盖语义、manifest 幂等+extras 跨调用合并、finalize 三态、resume 追加、load_metrics 截断容忍/空文件、summarize_experiment manifest 优先、OSError 降级为告警、BaseTrainer 集成 4 例）
- **Green**：新增 `utils/experiment.py`（ExperimentRecorder + collect_env + load_metrics + summarize_experiment，含 finalized 属性与 wall_time()）；`training/trainer_base.py` 接入 `_setup_recorder`（master-only、`<out_dir>/experiments/<basename(out_dir)>`、no_metrics 关闭、异常降级警告）、`log_scalar`/`log_scalars` val 标签双写（val/loss、val/mean_reward、val/accuracy 映射）、`finalize_recorder` 助手（wall_time_s 自动填充）、cleanup 兜底 finalize("interrupted")
- **CLI**：train_pretrain/train_sft/train_grpo/train_dpo/iterative_grpo 新增 `--exp_id`/`--no_metrics`；train() 正常结束显式 finalize("completed")（pretrain 早停分支同样 finalize，iterative_grpo 额外回写 final_checkpoint artifact）
- **验证**：`tests/test_experiment.py` 24 passed；全量回归 257 passed, 3 skipped（原 233+3 基线 + 24 新增，无回归）；SFT CPU 端到端冒烟（tiny ModernGPT ckpt + 合成算术数据）确认 metrics.jsonl 2 条 eval 事件 + manifest status=completed/best_val_loss/steps/wall_time_s
- 备注：训练步级 train 事件（loss/lr/grad_norm/tokens_per_sec/mem）由 T002 的 on_step_end 接入；epoch_repetition 当前占位 0.0

### 2026-10-07 T002 完成（FR2）

- **Red**：子代理写 `tests/test_monitoring_integration.py` 14 用例（_setup_monitors 默认构建/禁用/构造异常降级、on_step_end 100 步 grad_norm 验收、字段透传、非 master no-op、spike 事件+LR×0.5、监控异常降级 ×2、5 脚本 CLI 冒烟）→ Red 确认 14 failed（AttributeError）
- **Green**：
  - `utils/monitoring.py`：LossSpikeDetector.update 返回值 additive 增加 window_mean/window_std（spike 事件字段，design §4.2 流程 B）
  - `training/trainer_base.py`：模板链新增 `_setup_monitors`（master-only、`--no_monitoring` 时不构建、构造异常降级警告）；新增 `on_step_end(step, loss, lr, grad_norm, tokens_per_sec)` 公共钩子：grad_norm 缺省时由 GradientNormMonitor.compute 计算 → LossSpikeDetector.update → spike 时 WARN+optimizer lr×factor+record_event("loss_spike") → record_step（None 字段省略、CUDA mem_gb）——全程异常降级不中断训练
  - **命名冲突解决**：GRPOTrainer 既有扩展钩子 `on_step_end(step, metrics, rollout)` 与新基类钩子撞名 → 重命名为 `on_rollout_end`（train_grpo.py 定义+调用、iterative_grpo.py 覆写、docs/TECH_REPORT.md 引用同步更新；无测试/其他引用）
  - 5 个训练脚本新增 `--no_monitoring`；pretrain/sft/grpo/dpo 日志块各接入一次 `on_step_end`（grad_norm 经 clip_grad_norm_ 返回值捕获，避免 zero_grad(set_to_none) 后监视器取到 0）
- **验证**：test_monitoring_integration.py 14 passed；全量回归 271 passed, 3 skipped（257+14，无回归）
- 备注：spike 降 LR 作用于当前 lr，下一次 scheduler.step() 覆盖为调度值（设计已注明，事件全量记录）

### 2026-10-07 T003+T004 完成（FR3，合并执行：共享测试文件与配置贯通面）

- **Red**：子代理写 `tests/test_ablation_switches.py` 22 用例 → Red 确认 22 failed；**发现预存 bug**：`utils/config_models.py` 在 pydantic v2 下无法导入（`assert n_kv_head is not None` 在类默认构造时失败；pydantic 原本未安装、该模块为死代码）。子代理已安装 pydantic 2.13.5，Green 一并修复
- **Green**：
  - `model/modern_gpt.py`：ModernGPTConfig 新增 `position_encoding`（rope|learned，非法值 ValueError；learned+rope_scaling → ValueError）与 `ffn_variant`（swiglu|gelu；gelu+num_experts>1 → ValueError）；intermediate_size 缺省规则随变体（gelu→4×n_embd，swiglu→8d/3 对齐 128 不变）；to_dict/from_dict 新字段（旧 checkpoint 兼容，getattr 默认值）；GELUFFN 类（fc→gelu→proj 无 bias，接口同 SwiGLU 返回 (out, zero_aux)，residual 输出投影沿用 0.02/sqrt(2L) 深度缩放初始化）；Block.mlp 按变体实例化；CausalSelfAttention learned 分支跳过 apply_rope（rope 模块不创建）；ModernGPT learned 分支 wpe Embedding + 绝对位置相加（start_pos+past_len 起算，与 RoPE 切片逻辑对齐）+ 超出 block_size → ValueError（B3 边界）
  - `utils/config_models.py`：修复预存导入 bug（_check_gqa 对 None 回退 n_head）；新增 position_encoding/ffn_variant Literal 字段 + _check_ablation_combos 验证器（中文提示）+ intermediate_size 联动
  - `training/train_pretrain.py`：新增 `--position_encoding`/`--ffn_variant`/`--qk_norm`/`--num_experts` 四个 CLI flags（后两个为设计消融矩阵 ABLATION_GROUPS 所需、原缺失）；_build_model 传入全部四个 kwargs
  - Hydra：config/hydra/defaults/model/modern.yaml 增加 position_encoding/ffn_variant（qk_norm/num_experts 已有）
- **验证**：test_ablation_switches.py 22 passed（含 cache/no-cache torch.equal 一致性、参数量公式、Pydantic 中文报错、旧 checkpoint 兼容、CLI/YAML/Hydra 贯通）；全量回归 293 passed, 3 skipped（271+22，无回归）；hydra-core 安装后 compose 验证新字段默认值正确
- 备注：hydra/pydantic 已装入当前环境（pydantic 2.13.5、hydra-core）；learned PE 与 NTK 外推为实验对照点（前者硬上限 block_size）

### 2026-10-07 T005 完成（FR4）

- **Red**：子代理写 `tests/test_component_ablations.py` 29 用例（7 组矩阵固定顺序/flags、目录布局、get_args 契约、is_group_completed 五态、build_train_command 精确命令序列、run_train_stage 断点续跑/子集按 GROUP_ORDER 排序/失败隔离/fail_fast、count_group_params total/active（moe active==dense 等价、no-rope +wpe）、summarize_groups 全矩阵+缺目录降级、write_markdown_summary 纯函数+errors 段、main 冒烟 ×4）→ Red 确认（ModuleNotFoundError）
- **测试文件 bug 修复**：7 个用例使用 `monkeypatch` 但未声明 fixture 参数（NameError）→ 补签名，非契约变更
- **Green**：新增 `run_component_ablations.py`（repo 根，run_ablations.py 同级）：ABLATION_GROUPS/GROUP_ORDER 固定 7 组矩阵；get_args（--stage/--groups/--config/--max_iters/--fail_fast/--out_root）；group_out_dir/group_exp_dir（`{out_root}/ablation-<name>/experiments/ablation-<name>`，正斜杠拼接）；is_group_completed（manifest status=="completed"，缺失/损坏/非 completed → False）；build_train_command（`python -m training.train_pretrain` + config/out_dir/exp_id + 组 flags + 尾部可选 --max_iters）；run_train_stage（完成组跳过、失败记录并继续、fail_fast 截停、模块级 subprocess.run 可替换）；count_group_params（YAML 基础配置 + 组 flags 覆盖 → 真实实例化 BaselineGPT/ModernGPT，get_num_params(non_embedding=False)，moe active = num_experts=1 等价实例）；summarize_groups（summarize_experiment 聚合 + delta_vs_baseline_pct 基准对比 + params_m/active_params_m，写 `{out_root}/experiments/ablation-summary.json|.md`）；write_markdown_summary（纯函数，列头含 best val loss/final val loss/tok/s/wall time，无 errors 时不含 "error" 字样）；main（train/summarize/all 编排，0/1/2 退出码）
- **pyproject.toml**：新增 `plot` extras（matplotlib>=3.7.0）与 `[project.scripts] nanogpt-ablations = "run_component_ablations:main"`
- **验证**：test_component_ablations.py 29 passed；全量回归 332 passed, 2 skipped（303+29，无回归）；入口点导入验证通过
- 备注：目录拼接刻意用 `/` 字符串拼接（跨平台与测试契约一致，Windows 下混合分隔符文件操作兼容）；delta_vs_baseline_pct 不取整（测试 approx 精度要求）

### 2026-10-07 T006 完成（FR5，执行型任务 + 1 个环境适配代码变更）

- **环境障碍与解决**：本机代理封锁 python HTTP 库对 huggingface.co 的 CONNECT 隧道（requests/httpx 均 403/500），但 curl.exe 可通；datasets 5.0.1 已移除 trust_remote_code/script 数据集支持 → 改用 HF 官方 parquet 转换分支（`api/datasets/Skylion007/openwebtext/parquet`，80 shards × ~289MB ≈ 23GB）curl 全量下载（断点续传脚本 + 逐 shard 校验）
- **代码变更（only-add）**：`data/prepare.py` 新增 `--data_files`（逗号分隔 glob，本地 parquet 直读，`load_dataset("parquet", data_files=...)`，优先于 hub 路径；无匹配 → FileNotFoundError）；`get_args(argv=None)` 可选 argv。新增 `tests/test_prepare_data_files.py` 4 用例（flag 解析/无匹配异常/多文件加载排序/优先级）→ 4 passed；全量回归 336 passed, 2 skipped（332+4）
- **数据产出**（`data/openwebtext/`）：val.bin = shard 79 前 10K docs（11,147,787 tokens，21.3MB）；train.bin = shards 0-78 全量 7,913,597 docs / **8,926,903,359 tokens**（17.0GB，uint16，90 shards）+ train.idx/val.idx 元数据
- **踩坑记录**：shard 69 首次下载损坏（thrift page header 反序列化失败，文件大小正常但中段页损坏）→ pyarrow 逐 row-group 全量扫描定位 + 删缓存（HF_DATASETS_CACHE 重定向至 D:/hf_cache/datasets）+ 重新下载修复（100,172 rows）
- **validate**：train.bin/val.bin 均 vocab check OK（0-50256）、EOT 计数与 doc 数一致（7,913,597 / 10,000）、平均 doc 长度 1128/1115 tokens、解码样本为正常英文网页文本
- **全局 shuffle 索引**：`generate_shuffle_index(train.bin, seed=1337)` → 8,709,174 chunks（block_size=1024+1 窗口），66.4MB，抽样验证为有效置换
- **端到端冒烟**：`get_dataloader(train, batch 4×1024)` 取批成功（token 范围 0-50256）
- 备注：磁盘占用 D: 约 40GB（parquet 23GB + bin 17GB + arrow cache ~24GB）；训练加载器（OpenWebTextDataset 包装）会自动使用 .shuffle.idx

### 2026-10-07 设计阶段完成

- design.md 已生成并获用户批准（用户指示自动化推进，免除逐步确认）
- 设计门控：用户选择跳过（自动化指示）
- 组件详设：`specs/component-detail-design/` 不存在，Step 7 按规则跳过
- 任务分解经与 design.md §3 功能点对照确认对齐（T001-T013 ↔ FR1-FR13 ↔ 功能点 1-18），无需调整

### 2026-10-08 T007 启动（FR6+E7，执行型任务）

- **环境问题根因与修复**：Windows 下训练在首次 eval（iter 10）必崩——`estimate_loss` 重新迭代 DataLoader 时 spawn 新 worker 报 `OSError [Errno 22]`；最小复现确认触发条件为「旧 iterator 的 4 个 worker 存活 + CUDA 已初始化时再次并发 spawn」（Windows multiprocessing spawn 竞态）。修复：`config/pretrain.yaml` 与 `config/hydra/defaults/data/openwebtext.yaml` 的 `num_workers: 4 → 0`。依据：① GPU 步进 ~0.5 it/s 为瓶颈，memmap 主进程加载零影响（smoke 实测 tok/s 持平 ~6100）；② `MemmapDataset`（IterableDataset）多 worker 时各 worker 取全局 shuffle 置换的连续切片交错读取，`num_workers=0` 才严格按全局 shuffle 索引顺序遍历，更符合设计 §4.2.2「数据顺序由全局 shuffle 索引锁定」
- **验证**：smoke 20 iters 完整跑通（exit=0，manifest status=completed，final_metrics 含 best/final val loss/steps/wall_time_s）；orchestrator 端到端 2 组 × 15 iters 冒烟通过（summary.md 全列正确：baseline 64.086M/modern-full 59.561M 参数、Δ%、tok/s、峰值显存）；全量回归 336 passed, 2 skipped（无回归）；test_hydra_config 两配置一致性保持
- **正式训练已启动**：`python run_component_ablations.py --stage all --out_root out` 以 detached 进程运行（编排器 PID 21308），日志 `out/ablation-runner.out.log`/`.err.log`（块缓冲，周期落盘）；GPU 99% 利用率确认 baseline 组训练中；Windows 睡眠/休眠已确认禁用（AC timeout=0）
- **吞吐估算**：~6100 tok/s → 单组 18k iters ≈ 10.5h + eval 开销，7 组总计 ≈ 75h（跨多会话）
- **已知风险**：modern-full 峰值显存 15.49GB/16GB（WDDM 共享显存溢出可跑通）；**moe 组 +~78M 参数（×16B 优化器状态 ≈ +1.25GB）有 OOM/降速风险**——编排器失败隔离保证其余 6 组完成，若 moe 失败需后续单独处理（候选：--gradient_checkpointing）
- **断点续跑语义**：manifest status=="completed" 的组自动跳过；中断组内可用 latest_ckpt.pt（每 1000 iters 保存）手动 `--resume` 恢复
- **提交**：41ac0cf（T001-T006 累积工作 + .gitignore data/owt_parquet/）+ 本次 num_workers 配置修复

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-07 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（G1/G4/G5/G6/G7/G8/G9 = YES；G2/G3 = Minor WARN）
- 修复的问题：G2（FR8/FR9/FR10/FR12/FR13 验收标准补写 Given/When/Then 结构）、G3（§3.5 重复率量化为 <5% + manifest 字段；§3.10 明确指标口径：三档各 ≥100 样本、format_rate/accuracy 定义、胜率 ≥100 对比样本）——已在门控后修复
- 审查代理：sdd-gate-reviewer
