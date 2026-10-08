# [AR-v0.5.0-hpc-readiness] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.5.0-hpc-readiness |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（已批准，B+B 方案） |
| 创建日期 | 2026-10-08 |

## 任务列表

> 阶段顺序：A 正确性与可移植性（T001-T007，v0.4.0 实验硬前提）→ B 性能强化（T008-T015）→ C 工程整理与理解力（T016-T021）。
> T017（args 提取）可与 A 阶段并行提前执行，减少后续任务在 5 个脚本间的重复改动。

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | FR1 依赖与环境固化：pyproject 补 server/validate extras、torch pin ~=2.5、uv.lock 重新同步、Dockerfile/compose 现代化、CI 补 lock 检查 + Docker build job | - | passing | 2026-10-08 完成；torch 收紧为 `~=2.5,<2.9` 消除 cu13 分裂 |
| T002 | FR2 数据迁移：修复 prepare.py --max_docs 复现陷阱、新增 docs/MIGRATION.md（必带清单/验证/冒烟/冗余清理）、本机干净环境演练 ≤1h | T001 | passing | 2026-10-08 完成；演练核心链路 2m17s |
| T003 | FR3a 断点续跑：编排器组内自动 --resume（ckpt 探测/架构校验/降级重跑/resumed_from_iter）、训练 RNG 恢复 + resume_offset 真实推进 + early_stop_counter、新增 --ckpt_interval | - | pending | 云抢占刚需，TDD |
| T004 | FR3b 运行管理：日志轮转（RotatingFileHandler）、metrics.jsonl 心跳 watchdog、编排器失败 webhook 通知（缺 URL 静默） | T003 | pending | 降级哲学与现有一致 |
| T005 | FR4a Paged KV 双 bug 修复（get_cache 逐层 / block table 逐层 append）+ n_layer=2 回归测试；generate 四处 max_cache_len → effective_block_size（NTK 解锁） | - | pending | v0.4.0 推理实验硬前提 |
| T006 | FR4b 投机解码分布修正（greedy p_d=point mass/完整 q 残差/温度统一）+ 分布等价性测试（draft==target KL→0）；temperature=0 → greedy 三处 | - | pending | 接受率不再虚高 |
| T007 | FR4c profile_inference 假 per-token 计时修复（挂入真实 decode 回调） | - | pending | 微型任务 |
| T008 | FR8a GQA enable_gqa=True（torch≥2.5 探测回退）+ GQA 探测移出首 forward（compile 图断裂修复）+ TF32 推广全部 5 训练器 | - | pending | |
| T009 | FR8b 数据路径（non_blocking/persistent_workers/num_workers 环境分层）+ eval 瘦身（batch 放大/多卡 allreduce/去 .item() sync） | - | pending | |
| T022 | FR8c EMA foreach 批量更新 + micro-step 去 all_reduce/.item() 同步 + 异步 checkpoint（staged+线程，保持原子性，失败回退同步） | T009 | pending | 异步 ckpt 需独立验证原子性回退 |
| T010 | FR9a DDP 数据 rank 分片（互斥切片 + resume_offset 分片语义）+ no_sync/static_graph/gradient_as_bucket_view | T009 | pending | 正确性级修复 |
| T011 | FR9b FSDP 补齐：auto_wrap_policy/use_orig_params/compile 顺序/FSDP checkpoint 修复（死代码接线或 DCP）+ 分布式 round-trip 测试 | T008,T010 | pending | |
| T012 | FR6 MoE 向量化重写（方案 B）：参数改单张量 `Parameter[E,H,C]`（gate/up/down 各一）+ load/save 翻译 hook（磁盘格式不变）、定容 padded bmm `[E,cap,C]` 零 Python 循环（cap 语义与现行 int 截断对齐，容量丢弃 token 输出 0）、compile 兼容；验收 loss atol=1e-5 + ≥2x 基准对比 | - | pending | kernel 级，TDD；旧 ckpt 双向兼容为设计锁定项（B1） |
| T013 | FR5 KV decode 去 cat（方案 B）：`KVCacheManager` 新增 `layout="append"` 追加式预分配缓冲 + `get_cache()` 返回 `[:pos]` 视图直喂 SDPA（零拷贝）、int8 惰性反量化切片、滑窗 mask 向量化；`pos ≥ max_cache_len` 切 ring 驱逐语义回退路径（`evicted` 标记，B5）；flash wrapper 仅可选分支（包存在才启用）；验收输出逐位一致 + ≥1.5x | T005 | pending | kernel 级；append vs 旧 cat 路径逐 token 等价测试先行 |
| T014 | FR7a 连续批处理 iteration-level 重构：槽位回收/prefill 插入/**左对齐 padding**（右侧 pad 不参与 KV 写入）/RoPE 位置按各序列真实长度/**PrefixCache 与 _init_decode_compile 删除**（设计决策：不留死代码）/批量采样张量化（`[B]` 温度张量，消灭逐序列 Python 循环） | T006,T013 | pending | ~300 行调度器重写；必须保留/继承 T006 的 temperature=0 greedy 语义，验收补 greedy 确定性断言 |
| T015 | FR7b api_server 接线新调度器（asyncio.to_thread/线程安全队列）+ 补齐或删除 --continuous_batching 声称 | T014 | pending | |
| T016 | FR10 根目录收编：导出 4 脚本 → tools/、profile 2 脚本 → tools/、编排脚本处置（[project.scripts] 与测试路径同步）、删除死文件（run_inference_ablation.py/generate_experiment_log.py/logs_prepare*.txt） | - | pending | 删除前全库引用检查 |
| T017 | FR11a training/args.py 公共 flag 提取（23 个重复 flag，~350 行净删） | - | pending | 可与 A 阶段并行提前 |
| T018 | FR11b load_model 统一到 load_model_from_checkpoint（7 份拷贝收敛）+ tests/conftest.py tiny fixture + test_bugfixes.py 拆分 | T017 | pending | |
| T019 | FR12 接口统一（breaking）：BaselineGPT 3-tuple 返回、no-op 参数显式化、调用点全量同步、反射 hack 清理、CHANGELOG | T013,T014 | pending | v0.5.0 大版本 |
| T020 | FR13 lint/测试基建：black line-length=100 统一 + ruff 配置 + mypy 版本对齐 + test_kv_cache_benchmark 相对断言/perf marker + 文档数字刷新 | T017,T018 | pending | 一次性格式化对齐 |
| T021 | FR14 理解力交付物：docs/ARCHITECTURE.md（训练基座）、modern_gpt.py 分区导航 + docstring 100%、GRPO 家族对比笔记、与 MIGRATION.md 交叉引用 | T011,T019 | pending | 收尾任务 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-08 立项（需求阶段）

- 背景：AR-v0.4.0 T007 在本机实测 ~71h+ 不可行（用户重启中止，baseline step 4610 处中断）；决定迁移云租用高算力环境，先完成框架强化
- 四路深度分析（训练性能/推理性能/可移植性/工程质量）完成，关键发现：
  - 推理正确性：Paged KV n_layer≥2 输出错误、generate 路径 NTK 被 block_size 架空、投机解码接受分布有偏、temperature=0 非 greedy
  - 可移植性 P0：torch 未 pin（lock cu12/cu13 静默分裂）、uv.lock 失同步、fastapi/pydantic 幽灵依赖、数据复现命令与实际不符
  - 断点续跑：编排器无组内 resume、RNG 存而不取、resume_offset 恒 0
  - 训练性能：GQA 未用 enable_gqa、MoE 每步 stack + Python 循环、DDP 数据不分片、FSDP checkpoint 死代码
  - 工程债：626 行 argparse 重复、load_model ×7、black 两套真相、根目录 12 脚本、1 个红灯计时测试
- 需求澄清结论：云租用实例假设、含 kernel 级改造、连续批处理纳入重构、接口统一允许 breaking、kernel 验收 = 等价（atol=1e-5）+ 基准对比、工程整理全面 + 设计文档
- srs.md 已生成（14 FR，三阶段），待用户确认

### 2026-10-08 设计阶段完成

- 方案提案：MoE 路线（A 保结构循环重写 / B 堆叠参数+定容 padded bmm）与 KV decode 路线（A 两段 SDPA / B 追加式预分配缓冲+视图切片）；用户选定 **B+B 组合**，flash wrapper 为可选分支
- design.md 已生成并经用户自动化授权批准：22 功能点 ↔ 14 FR 全追溯；3 时序图 + 4 流程图（PlantUML）；12 项边界条件矩阵（B1-B12）+ 10 项异常场景（E1-E10）；新增 19 个测试文件规划（含门控修复新增 5 个）
- Step 6 一致性核对：T012/T013/T014 任务描述已按选定方案同步（堆叠参数+翻译 hook / append 布局+ring 回退 / 左对齐 padding+死代码删除）；api_server.py 定为保留根目录（uvicorn 入口契约）
- Step 7：`specs/component-detail-design/` 不存在，组件详设刷新按规则跳过

### 2026-10-08 会话记录（T001）

- 完成任务：T001 依赖与环境固化（标准任务，Red/Green 分离）
- 修改文件：pyproject.toml（torch `~=2.5,<2.9` + server/validate extras）、requirements.txt（派生标注）、docker-compose.yml（`runtime: nvidia` → `gpus: all`，删过时 version 键）、Dockerfile（基础镜像 cuda 12.1→12.8 对齐 torch 2.8 cu128 默认变体 + cu12x 变体命令注释 + prod stage 补 `.[server]`）、.github/workflows/ci.yml（新增 lock-and-docker job：`uv lock --check` + `docker build`）、uv.lock（重生成：provides-extras 全 7 extras、torch 统一 2.8.0 消除 cu13 分裂）、tests/test_static_contracts.py（新增，7 用例）
- 测试：test_static_contracts 7/7（1 importorskip 后本机装 fastapi 验证 import api_server OK）；全量回归 342 passed / 2 skipped / 1 已知红灯 deselect（基线 335+2 无回归）
- 备注：uv 需内部镜像源（`--index-url http://mirrors.tools.huawei.com/pypi/simple/`，直连 pypi 隧道失败）；api_server 的 fastapi `on_event` 弃用 warning 为既有代码，T015 接线时现代化；覆盖率口径不适用（配置文件任务，无新增运行时模块）

### 2026-10-08 会话记录（T002）

- 完成任务：T002 数据迁移 runbook 与复现陷阱修复（标准任务，Red/Green 分离）
- 修改文件：data/prepare.py（新增 `resolve_max_docs()` + `--full` flag + help 明示默认 1M 陷阱语义；map/streaming 两模式接线）、docs/MIGRATION.md（新增：必带清单 17.1GB/环境重建/30s 数据验证/冒烟判据/编排器启动/冗余清理 39.1GB/数据重建/排障表/演练记录）、README.md + docs/EXPERIMENTS.md + docs/TECH_REPORT.md（6 处复现命令加 `--full`）、tests/test_prepare_full_mode.py（4 用例）+ tests/test_migration_runbook.py（3 用例）
- 演练验收（FR2-AC1）：干净 venv（UV_PROJECT_ENVIRONMENT 隔离）核心链路 **2m17s**（uv sync 1m20s + validate 27s + 50 步冒烟 30s）≤1h 达标；val loss 10.835→10.741；产物判据全过（latest_ckpt/best_ckpt/metrics.jsonl 5 train + 2 eval 事件）
- 演练发现并固化两个平台差异（写入排障表）：① Windows PyPI torch wheel 为 CPU-only（Linux manylinux 自带 cu128，云不受影响）；② `train_pretrain.py:110` argparse 默认 `--num_workers 4` 在 Windows 触发 spawn 竞态崩溃——runbook 冒烟命令显式 `--num_workers 0` 双平台通用，分层默认留 T009
- 测试：新增 7 用例全过；全量回归 349 passed / 2 skipped（零回归）；测试侧修正 1 处误报（usage 行含 --full 触发 help 语义断言，过滤为选项描述行）
- 备注：`--split train --full` 全量重建一致性未实际重跑（8.9B token 重建 2-4h，由 resolve_max_docs 单测 + idx 判据表覆盖）；冗余目录清理命令已文档化但本机未删（数据仍被 v0.4.0 潜在重建使用）

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-08 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（G1/G2/G4/G5/G8/G9 = YES；G3/G6/G7 = Minor WARN）
- 修复的问题：G3 六处模糊词量化（resume loss 跳变 ≤2× 历史波动、KL <1e-3 + greedy 接受率=1.0、显存线性回归斜率 p>0.05、CB 延迟 ≥30%、tok/s ≥0.98× + eval 占比降 ≥5pp、测试时长 ≤1.1× + ModernGPTConfig 计数 <40）；G6 T009 拆分为 T009+T022（异步 ckpt 独立验证原子性回退）；G7 T014 依赖补 T006 + 备注 greedy 语义继承 + 验收补确定性断言——已在门控后修复
- 审查代理：sdd-gate-reviewer

### 2026-10-08 design 门控记录（第 1 轮）

- 门控结果：FAIL
- 失败项（Important ×4）：G12（功能点 12/13/14 部分/19 部分/20/22 在 §6 零对应用例）、G13（边界 B2/B3/B6/B9/B11/B12 无用例；投机解码拒绝分支因用例全为 draft==target 永不触发）、G14（FR5/FR6/FR8 性能基准类验收及 FR1-AC3/FR10-AC1/AC3/FR11-AC1/AC3/FR12-AC1/FR13-AC1~AC3 共 12 条不可追溯）、G15（旧 ckpt 缺键默认值/num_experts=1 快速路径/ckpt_interval=0/get_dataloader 参数/api_server flag/benchmark 加固等接口边界缺用例）
- Minor WARN ×5：G2（FR8 实现要点缺失）、G3（build_train_command 等签名不完整）、G7（方案对比未入 design.md）、G9（§4.4 缺 run_full_evaluation.py/run_ablations.py）、G10（design.md 未声明组件详设跳过）
- 审查代理：sdd-gate-reviewer
- 修复方向：全部属 design.md §6 补充用例范畴（含 §4.2.2 补 FR8 实现要点、新增 §6.5 性能基准与静态检查节、§1 方案决策记录），无需返工 req

### 2026-10-08 design 门控记录（第 2 轮）

- 门控结果：FAIL
- 第 1 轮失败项修复验证：G12/G14/G15 Important 与 G2/G3/G7/G9/G10 Minor 全部确认修复；G13 部分修复
- 残留失败项（Important ×1）：G13-B10（异步 save 并发排队串行化边界无用例，交叉索引声称落点 test_training_perf_flags 但该文件仅含正常路径 round-trip——「声称覆盖但无实际用例」失配）
- 新增 Minor ×5：交叉索引漏 B1；FR1-AC2 Docker build job 未在 §6.2 明示；FR14-AC2 docstring 100% 未落静态检查；流程图 A completed→skip 分支无回归用例声明；FR2-AC1 冒烟步数 50 vs design 100 不一致；§4.4 未说明 5 个 *_hydra.py 变体处置
- 审查代理：sdd-gate-reviewer
- 修复方向：§6.1 补 B10 并发排队串行 + E5 显式用例等单点修补，第 3 轮增量复核

### 2026-10-08 design 门控记录（第 3 轮·增量复核）

- 门控结果：PASS
- 第 2 轮修复验证：8/8 落位（B10/E5 显式用例、交叉索引 B1、Docker build job、docstring 静态检查、completed→skip 回归、冒烟步数 50 对齐、hydra 变体处置、from_dict 旧 config 用例），行级证据确认
- 回退抽查：G12/G14/G7/G3/边界矩阵均未回退
- 顺手修复：srs.md FR2 期望行为「100 步冒烟」→「50 步」（与 FR2-AC1 统一，Minor 表述瑕疵）
- 累计轮次：3（FAIL/FAIL/PASS）
- 审查代理：sdd-gate-reviewer
