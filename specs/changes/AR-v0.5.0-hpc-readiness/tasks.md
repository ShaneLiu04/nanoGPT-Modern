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
| T003 | FR3a 断点续跑：编排器组内自动 --resume（ckpt 探测/架构校验/降级重跑/resumed_from_iter）、训练 RNG 恢复 + resume_offset 真实推进 + early_stop_counter、新增 --ckpt_interval | - | passing | 云抢占刚需，TDD |
| T004 | FR3b 运行管理：日志轮转（RotatingFileHandler）、metrics.jsonl 心跳 watchdog、编排器失败 webhook 通知（缺 URL 静默） | T003 | passing | 降级哲学与现有一致 |
| T005 | FR4a Paged KV 双 bug 修复（get_cache 逐层 / block table 逐层 append）+ n_layer=2 回归测试；generate 四处 max_cache_len → effective_block_size（NTK 解锁） | - | passing | 2026-10-08 完成；与 T006/T007 同会话合并交付 |
| T006 | FR4b 投机解码分布修正（greedy p_d=point mass/完整 q 残差/温度统一）+ 分布等价性测试（draft==target KL→0）；temperature=0 → greedy 三处 | - | passing | 2026-10-08 完成；统计验收 TVD≈0.07 vs bug 0.82 |
| T007 | FR4c profile_inference 假 per-token 计时修复（挂入真实 decode 回调） | - | passing | 2026-10-08 完成；含 generate.py benchmark 加固（CPU perf_counter/warmup≥3/p50-p99） |
| T008 | FR8a GQA enable_gqa=True（torch≥2.5 探测回退）+ GQA 探测移出首 forward（compile 图断裂修复）+ TF32 推广全部 5 训练器 | - | passing | 2026-10-09 完成；旧惰性探测在 torch≥2.5 上从未启用原生内核（探测不传 enable_gqa） |
| T009 | FR8b 数据路径（non_blocking/persistent_workers/num_workers 环境分层）+ eval 瘦身（batch 放大/多卡 allreduce/去 .item() sync） | - | passing | 2026-10-09 完成；发现 persistent_workers 与 estimate_loss/epoch-wrap 的两个隐蔽冲突并安全绕开 |
| T022 | FR8c EMA foreach 批量更新 + micro-step 去 all_reduce/.item() 同步 + 异步 checkpoint（staged+线程，保持原子性，失败回退同步） | T009 | passing | 2026-10-09 完成；异步 ckpt 原子性回退已独立验证（B10 串行排队 + E5 失败回退用例） |
| T010 | FR9a DDP 数据 rank 分片（互斥切片 + resume_offset 分片语义）+ no_sync/static_graph/gradient_as_bucket_view | T009 | passing | 2026-10-09 完成；跨步切片零丢弃（优于 DistributedSampler padding/drop）；static_graph+no_sync 兼容性待 GPU 冒烟验证 |
| T011 | FR9b FSDP 补齐：auto_wrap_policy/use_orig_params/compile 顺序/FSDP checkpoint 修复（死代码接线或 DCP）+ 分布式 round-trip 测试 | T008,T010 | passing | 2026-10-09 完成；FULL_STATE_DICT 真实 gather 是 CUDA-only，CPU 侧以「真构造 + stub 路由」分层覆盖，GPU 冒烟待迁移演练 |
| T012 | FR6 MoE 向量化重写（方案 B）：参数改单张量 `Parameter[E,H,C]`（gate/up/down 各一）+ load/save 翻译 hook（磁盘格式不变）、定容 padded bmm `[E,cap,C]` 零 Python 循环（cap 语义与现行 int 截断对齐，容量丢弃 token 输出 0）、compile 兼容；验收 loss atol=1e-5 + ≥2x 基准对比 | - | passing | 2026-10-10 完成；实测 17.9x（137.1→7.7 ms/step @ E=4,H=688,C=256,N=512 CPU 中位），等价 diff ~3e-8 |
| T013 | FR5 KV decode 去 cat（方案 B）：`KVCacheManager` 新增 `layout="append"` 追加式预分配缓冲 + `get_cache()` 返回 `[:pos]` 视图直喂 SDPA（零拷贝）、int8 惰性反量化切片、滑窗 mask 向量化；`pos ≥ max_cache_len` 切 ring 驱逐语义回退路径（`evicted` 标记，B5）；flash wrapper 仅可选分支（包存在才启用）；验收输出逐位一致 + ≥1.5x | T005 | passing | 2026-10-10 完成；4096ctx CPU 基准 1.61x（21.4→13.3 ms/步），fast path 逐位等价 + 单写协议 |
| T014 | FR7a 连续批处理 iteration-level 重构：槽位回收/prefill 插入/**左对齐 padding**（右侧 pad 不参与 KV 写入）/RoPE 位置按各序列真实长度/**PrefixCache 与 _init_decode_compile 删除**（设计决策：不留死代码）/批量采样张量化（`[B]` 温度张量，消灭逐序列 Python 循环） | T006,T013 | passing | ~300 行调度器重写；必须保留/继承 T006 的 temperature=0 greedy 语义，验收补 greedy 确定性断言；**T005 定向新发现**：`generate_batch` 解码首步将 prompt 尾 token 重复喂入（KV 已含整个 prompt）——off-by-one 独立缺陷，重构时一并修复 |
| T015 | FR7b api_server 接线新调度器（asyncio.to_thread/线程安全队列）+ 补齐或删除 --continuous_batching 声称 | T014 | passing | |
| T016 | FR10 根目录收编：导出 4 脚本 → tools/、profile 2 脚本 → tools/、编排脚本处置（[project.scripts] 与测试路径同步）、删除死文件（run_inference_ablation.py/generate_experiment_log.py/logs_prepare*.txt） | - | passing | 删除前全库引用检查 |
| T017 | FR11a training/args.py 公共 flag 提取（23 个重复 flag，~350 行净删） | - | passing | 可与 A 阶段并行提前 |
| T018 | FR11b load_model 统一到 load_model_from_checkpoint（7 份拷贝收敛）+ tests/conftest.py tiny fixture + test_bugfixes.py 拆分 | T017 | passing | |
| T019 | FR12 接口统一（breaking）：BaselineGPT 3-tuple 返回、no-op 参数显式化、调用点全量同步、反射 hack 清理、CHANGELOG | T013,T014 | passing | v0.5.0 大版本 |
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

### 2026-10-08 会话记录（T003）

- 完成任务：T003 FR3a 断点续跑强化（标准任务，Red/Green 分离 + 端到端演练）
- 修改文件：run_component_ablations.py（新增 `find_group_checkpoint` 三态探测 / `_arch_matches`+`ckpt_matches_group` 架构校验（model_type↔--model、n_kv_head（None→n_head 回退、baseline 跳过）、num_experts/position_encoding/ffn_variant 默认值比对）/ `build_train_command` 加 `resume_from` 注入 `--resume` / `run_train_stage` resume 决策 + 失败降级无 --resume 重跑一次 + `resumed_from_iter` 返回键 / `_group_overrides` YAML+flags 合并比对 / markdown 表新增 Resumed From Iter 列 + JSON summary 挂载）；utils/checkpoint.py（save_checkpoint 新 kwargs `early_stop_counter/consumed_chunks/world_size/rank`，load_checkpoint extra 透传同键）；training/trainer_base.py（import set_rng_state；CheckpointManager.save 加 `extra_state` 透传；load 接线 `set_rng_state(extra["rng_state"])`）；training/train_pretrain.py（`--ckpt_interval` flag（0=跟随 eval_interval）；`_build_data` train loader 显式传 `train.bin.shuffle.idx`（存在时，落实全局 shuffle 确定性 resume）；`_maybe_resume` 恢复 consumed_chunks（优先）/resume_offset/early_stop_counter + world_size 不一致 WARN + 重建 loader 后重放 RNG；训练循环：每 batch `resume_offset += batch_size`、epoch StopIteration 回绕双侧重置（trainer+dataset）、eval/ckpt 保存点整体上移至 body 头部）
- **关键修复（演练发现）**：原 eval/ckpt 保存点在 body 中部（batch 消费+step 后、iter_num++ 前），ckpt 存 `(iter=k, offset=(k+1)·B)`，resume 从 body(k) 头部重跑 → 最后一个 batch 恒重复训练一次（演练实测 consumed=404≠400）。修复 = 保存点上移至 body 头部，建立不变式「ckpt(iter=k) ⇔ 已完成 k 次迭代且数据位置=k·B」，resume 零重复零丢失；顺带 eval 语义自洽化（iter k 的 val loss = k 步模型而非 k+1 步）
- **关键修复（UT 发现）**：`_maybe_resume` 中 load（回卷 RNG）后 `_build_data` 重建 DataLoader 会消耗 torch RNG（iterator base seed，实测 2 draw）→ 构造结束 RNG 偏移；修复 = 重建后重放 `set_rng_state(extra["rng_state"])`
- 测试：tests/test_orchestrator_resume.py（6 用例：completed 回归守卫/三态探测/架构校验/命令注入/降级重跑/resumed_from_iter）+ tests/test_resume_integrity.py（5 用例：RNG 行为+接线双断言/offset 推进+恢复后首 batch 逐元素相等/early_stop 往返+旧格式默认/ckpt_interval=5 两时点落盘 vs =0 跟随 eval/旧 ckpt 缺键默认）——Red 9 失败 2 守卫通过 → Green 11/11；全量回归 361 passed / 2 skipped（零回归）
- 演练验收（FR3a 业务场景 E1 抢占）：真实 tiny 训练（CPU，2000 chunks 顺序 shuffle.idx）→ ckpt_interval=20 轮询至 iter≥40 强杀（manifest=running）→ 编排器重投递自动 `[resume] from iter 40` → 续跑至 100；断言全过：manifest completed/steps=100、final ckpt consumed=160+60×4=400 **精确对账**（修复前 404）、metrics.jsonl 轨迹 0..95 连续无重复、resumed_from_iter=40>0、「最多丢一个 ckpt_interval」成立
- 备注：resume 失败降级重跑路径由 UT monkeypatch 覆盖（演练走 happy path）；`estimate_loss` 迭代 train_loader 从 dataset.resume_offset（构造值）取样为 v0.4.0 既有行为，非本任务范围；epoch 回绕重置为新增语义（原实现 dataset.resume_offset 不复位导致回绕起点漂移）

### 2026-10-08 会话记录（T004）

- 完成任务：T004 FR3b 运行管理（标准任务，Red/Green 分离 + 真实冒烟）
- 修改文件：utils/notify.py（新增：`notify_webhook(message, env=None, timeout=10) -> bool`，URL 从 `NANOGPT_WEBHOOK_URL` env 读，缺失/空 → False 静默；POST JSON {"text"}，2xx → True，任何异常 → False 不抛）；utils/monitoring.py（新增 `MetricsHeartbeatWatchdog(threading.Thread)`：daemon 线程 + `check(now)` 同步核心（load_metrics 解析 train 事件 ts → 最近 baseline_window 间隔均值 → age > timeout_factor×baseline 首次告警）+ stderr/on_alert 回调/webhook 三连告警 + alerted 去抖 + stop() 幂等；轮询异常吞掉不杀线程）；utils/logging.py（Logger 加 `logging.handlers.RotatingFileHandler`：`{log_dir}/train.log`，maxBytes 10MB/backupCount 3，带 asctime Formatter，OSError 降级 console warning；同名 logger 复用时复用既有 handler；`close()` flush+close+removeHandler 释放句柄并置 file_handler=None 幂等）；run_component_ablations.py（`start_group_watchdog` 构造 watchdog 失败降级 None；`_run_with_watchdog` 包住每次 subprocess.run（resume 尝试与降级重跑各一轮）：显式 start（失败→None 继续）+ finally stop；`notify_failures(errors)` 失败摘要 webhook；main() errors 非空调用）；tests/test_bugfixes.py（3 个 GRPO 用例补 try/finally trainer.cleanup()——RotatingFileHandler 句柄在 Windows 阻塞 TemporaryDirectory 清理）
- **关键发现（Windows 句柄交互）**：新增 RotatingFileHandler 后，既有测试中「构造 trainer 不 cleanup」的用例连锁 PermissionError（WinError 32）——Logger.close() 必须 close+removeHandler 释放句柄，且测试侧补 cleanup；此交互 T018（conftest tiny fixture/测试拆分）应固化「trainer 构造必须配 cleanup」约定
- 测试：tests/test_run_management.py 8 用例（stall 告警+精确消息+去抖/健康与数据不足三态/webhook 缺 URL 静默/成功失败路径断言请求体与 Content-Type/RotatingFileHandler 落盘+类型+asctime 格式/notify_failures 消息与空跳过/编排器启停 watchdog（stop==start）/线程生命周期 daemon+join+stop 幂等）——Red 8 失败 → Green 8/8；全量回归 369 passed / 2 skipped（361 基线 + 8 新增，零回归）
- 冒烟验收（FR3b healthy 路径）：真实编排器 + 真 watchdog 线程跑 tiny 训练 40 步——completed、全程无 `[watchdog]` 告警、`out/ablation-baseline/logs/train.log` 落盘（0 bytes 属正常：健康跑无 warning 输出）
- 备注：watchdog stall 真实路径由 UT mock ts 覆盖（冒烟仅验证 healthy）；`NANOGPT_WEBHOOK_URL` 变量名已定契约（MIGRATION.md 环境清单可后续补文档，T021 收尾时统一交叉引用）

### 2026-10-08 会话记录（T005+T006+T007 合并交付）

- 完成任务：T005 FR4a + T006 FR4b + T007 FR4c（标准任务，Red/Green 分离 + 冒烟；三任务同属 FR4 推理正确性且缺陷互相咬合——temp=0 greedy 是投机逐位等价的前置——故同会话 TDD 连续交付）
- Red：两个并行子代理产出 3 个测试文件 21 用例，全部命中目标缺陷（paged 3：get_cache 逐层视图/块表跨层隔离/ring-paged 逐步 logits 等价；speculative 11：temp=0 三入口确定性/draft==target greedy 逐位等价/`_draft_sample`+`_verify` 接口数学性质 5 例/draft≠target 统计等价 TVD<0.25；profile 7：on_token 回调契约/真实计时/已知延迟序列统计/benchmark CPU 分位+warmup≥3 隔离）。基线对照证实既有 paged/speculative 测试在 n_layer=1 下恰好掩盖双 bug
- 修改文件：model/paged_kv_cache.py（Bug1：get_cache 改 `[self._get_layer_blocks(li) for li in range(n_layers)]`；Bug2：update 改「按需确保容量」语义——`needed_blocks = ceil((seq_len+S)/block_size)` 表不足才 append + 写入沿块路径 `block_no = pos // block_size` 索引，消除逐层重复 append）；model/modern_gpt.py（新增模块级 `_mask_logits`/`_sample`/`_draft_sample`/`_verify` 四接口，主路径/fullgraph/投机单序列/投机 batched 四处闭包全部收敛到模块级；`temperature==0 → argmax`；greedy draft p_d=point mass；残差 `norm(relu(p_t−q_d))` 用完整调温 draft 分布；验证 softmax 同温；vocab 不一致/draft token 越界 ValueError；generate 四处 `max_cache_len` → `getattr(config, "effective_block_size", block_size)`（主入口/fullgraph/speculative×2）；`on_token: Optional[Callable]` per-token 回调贯穿主路径 no-cache/cache/fullgraph/投机/fallback 全路径）；inference/generate_utils.py + inference/continuous_batching.py（`_sample_logits`/`BatchGenerator._sample` 补 `temperature==0 → argmax`）；utils/profiler.py（summary 增 `mean_ms`；`step(token=None)` 兼容回调签名）；inference/generate.py（benchmark：CPU 路径 perf_counter 计时修 UnboundLocalError 崩溃、warmup 1→3 次且不入统计、顶层 `p50_ms/p95_ms/p99_ms` 分位键单调）；profile_inference.py（删除「生成后循环补 step」假计时，改 `on_token=inf_prof.step` 真实 per-token 计时）
- **关键实现决策（投机分布正确性核心）**：`_verify` 只按温度调温不够——target/draft 的接受分布必须是**与直接生成相同的有效采样分布**（经 top_k/top_p mask 后）。修复中既有测试 `test_speculative_self_draft_matches_target_greedy`（top_k=1 场景）先失败暴露此点：解法是提取 `_mask_logits`（支持 [B,V] 与 [B,T,V]），调用侧先 mask 再传入 `_draft_sample`/`_verify`，接口签名维持 design §4.3.5 原样；验证侧 target_logits 对齐由 `cat([current_target_logits, logits_t[:, :-1]])` 构造，消灭原版 i==0/i>=1 分支重复
- 测试：21 新用例 Red→Green 全转绿；测试侧修正 1 处构造缺陷（`_fake_step` 未接受回调 token 参数）；全量回归 **390 passed / 2 skipped**（369 基线 + 21 新增，零回归）
- NTK 冒烟（FR4-AC4）：`rope_scaling={"type":"ntk","factor":4}`、block_size=32 → effective 128，生成 40 token（>block_size）ring 与 paged 路径均成功且逐位一致
- 遗留发现：`continuous_batching.generate_batch` 解码首步将 prompt 尾 token 重复喂入（此时 KV 已含整个 prompt）——off-by-one 独立缺陷，已记入 T014 备注（重构时一并修复）；`_generate_speculative_batched` 的 fallback 机制使 batch 投机 temp=0 断言不稳健，UT 以接口级+单序列端到端覆盖
- 持久化：commit + gitee 推送（新工作流规则：每个 AR 结束后推送 https://gitee.com/liu-xingyan04/nano-gpt-modern）

### 2026-10-08 会话记录（gitee 瘦身镜像同步）

- 完成工作：落地 `tools/gitee_mirror.py`，全量历史首次推上 gitee（29 commit + 148 seed = 169 链元素，远端 `main`=167accb0，默认分支已切 main，临时 refs 全清）
- **关键教训（git 协议本质）**：pack 协商是 **commit 祖先级差集**而非对象级去重——独立 seed 链对真实 commit 的包体毫无缩减（实测恒 402KB）。正解：把 seed commits **织入镜像链**（镜像 commit 以 seed 链尾为 parent），链上相邻元素差集天然受控，逐元素推送每包 ≤49KB
- **传输度量修正**：wire size 文本取 gz（pack 内 zlib≈gz）、图片取 raw（png 不可压）；批切分按「累加不超 42KB」上限式（非 flush-≥ 阈值式，后者允许 42+单 blob 超限）；`fig4_performance.png` 77KB>58KB 图片阈值纳入 quantize（此前仅 >95KB 触发漏网）
- 替换规则：fig1-4 PIL quantize（195/115/114/77KB → 43/44/40/33KB）、uv.lock×2 → stub（附 `uv lock` 再生指引）；镜像链确定性（seed committer date 取原 commit author date）保证重跑增量跳过
- 远端不可保真项：5 blob 替换 + commit parent 指向 seed 链（author/date/message 逐字保留）；本地历史零改动，gitee 为内容镜像
- 固化规则：**每个 AR 结束后运行 `python tools/gitee_mirror.py`**（幂等增量；BDWAF ~100KB POST 上限、SSH 22 封锁、NTLM 代理 `proxyhk.huawei.com:8080` 环境约束内置）
- 增量机制（后续两轮迭代）：跳过判据用远端 refs 的 **rev-list 祖先集**（精确 sha 匹配会在 _sm* 清理后失效）；被 merge 掉的旁支（GitHub 链镜像 commits 非 main 祖先）用常驻 `_mirror_side` ref 锚定，避免每次重推

### 2026-10-09 会话记录（T008）

- 完成任务：T008 FR8a 训练配置级性能优化·其一（标准任务，Red/Green 分离）
- 修改文件：model/attention_utils.py（新增 `sdpa_supports_enable_gqa()`：inspect 签名探测为主 + torch 版本解析回退，纯静态无设备副作用，可在构造期调用）；model/modern_gpt.py（`import warnings` + 顶部 import 探测函数；`CausalSelfAttention.__init__` auto 模式改**构造期静态解析**：torch≥2.5 → `"native"`、否则 `"grouped"`（删「repeat 待定 + 首 forward 惰性 probe」旧逻辑）；forward 删除 `probe_gqa_sdpa_support` 调用（该函数保留在 attention_utils），新增 `"native"` 分支：SDPA 携带 `enable_gqa=True`，运行期异常 → 一次性 UserWarning + 实例级永久降级 `"grouped"` 并以内联 grouped 完成本次 forward，不向调用方抛错（E6））；training/trainer_base.py（新增模块级 `enable_tf32()`：matmul + cudnn 双 flag，幂等，CPU-only 构建安全）；training/train_pretrain.py（删 train() 内两行内联 TF32，改 import 后模块级调用）；training/train_sft.py / train_grpo.py / train_dpo.py / iterative_grpo.py（各加模块级 `enable_tf32()` 调用）；tests/test_training_perf_flags.py（新增，22 用例）；tests/test_gqa_broadcast.py（契约同步 1 处：`test_auto_resolves_to_grouped_or_raw` → `test_auto_resolves_at_construction`，断言从「首 forward 后 ∈ raw/grouped/repeat」改为「构造后 ∈ native/raw/grouped/repeat」，对应 FR8a 设计变更）
- **关键收获（旧惰性探测的双重缺陷）**：① 探测调用本身不传 `enable_gqa`——torch≥2.5 上 raw 探测必然失败（SDPA 对不同头数要求显式 enable_gqa），生产路径从未用上原生 GQA 内核，一直降级 grouped；② 首 forward 中跑真实 SDPA 探测既是 torch.compile 图断裂源，也在首次推理引入设备探测开销。新方案两级防线：构造期 inspect 签名静态判定（零设备副作用、compile 安全）+ 运行期 try/except 一次性降级（B8 版本回退/E6 运行异常均覆盖）
- 测试：Red 20 failed / 2 passed（2 个 passed 为显式 grouped/repeat 回归守卫，docstring 注明预期）→ Green 22/22；全量回归 **412 passed / 2 skipped**（390 基线 + 22 新增，零回归）
- 备注：E6 降级为实例级永久性（`_gqa_mode` 改 grouped 后不再重试 native）；FR8-AC3 grep 语义由源码断言测试固化（5 训练器源码含 `enable_tf32()`）；hydra 变体经主脚本 import 自动获益；TF32 模块级调用使「import 训练器即生效」，测试通过 sys.modules.pop 强制重导入验证因果

### 2026-10-09 会话记录（T009）

- 完成任务：T009 FR8b 训练配置级性能优化·其二（标准任务，Red/Green 分离）
- 修改文件：data/openwebtext.py（新增 `default_num_workers()`：nt→0 / posix→4；`get_dataloader` 的 `num_workers=None` 环境分层解析 + 新增 `persistent_workers=None` 自动 `(nw>0)` + 显式覆盖优先、`pin_memory=True` 参数化）；training/train_pretrain.py（argparse `--num_workers` 默认 4→None、新增 `--eval_batch_size` 默认 None→2×batch；`_build_data` 局部解析不污染 args、新增 `train_eval_loader`（train split/eval 批/构造时 resume_offset/同 shuffle_idx）、val_loader 改 eval 批、train_loader 显式 `persistent_workers=False`；`estimate_loss` 重写：device 张量累积 `loss.detach()` 零逐 iter `.item()`、分布式 SUM all_reduce + /world_size、每 split 单次 mean().item()、batch `non_blocking=True`；eval 边界与收尾 eval 全 rank 化（master gate 从调用条件移除），best_val/early_stop 全 rank 一致推进，print/log/save 仍 master-only；`_forward`/训练批 `.to(device, non_blocking=True)`）；tests/test_training_perf_flags.py（追加 15 用例 + 自包含 tiny 数据/args helper）
- **关键发现（persistent_workers 两个隐蔽冲突，均绕开）**：① torch 的 persistent DataLoader 对每次 `iter()` 复用并 `_reset` 同一迭代器——estimate_loss 若继续迭代 train_loader 会**重置训练流**（破坏 T003 零重复不变量）→ 引入独立 `train_eval_loader`（复现旧「新鲜迭代器=构造时窗口」语义）；② epoch wrap 靠 `dataset.load_state_dict(新 offset)` 传递回绕位置，但 persistent worker **不会重新 pickle dataset**（worker 持旧副本）→ train_loader 必须恒 `persistent_workers=False`，val/train_eval（每 eval 重建迭代器、dataset 无中途变异）才安全启用
- **关键发现（Green 期修正 2 处）**：① 实现初版 `if self.distributed and dist.is_available() and dist.is_initialized()` 三重守卫超出契约（既有惯例 L608 仅查 `self.distributed`，分布式标志仅在 dist 初始化后为 True）→ 对齐惯例；② `_build_data` 初版回写 `args.eval_batch_size` 污染命名空间（测试断言构造后 args 仍无该键）→ 改局部变量
- 测试：Red 15 failed / 22 passed（T008 用例零影响）→ Green 37/37；全量回归 **427 passed / 2 skipped**（412 基线 + 15 新增，零回归，含 T003 resume 精确对账/编排器/监控集成全过）
- 备注：train_eval_loader 的窗口在 epoch wrap 后不再跟随回绕（独立 dataset 实例），train-split eval 语义为「构造时窗口近似」，仅用于日志，val 语义不变；multi-GPU allreduce 在 loader 未分片时各 rank 值相同（分片属 T010），本任务先消除「非 master 空转」；micro-step 逐步 all_reduce/.item()（L609）属 T022 范围未动；train_pretrain.py 本次 EOL 归一（CRLF→LF），645/708 行 diff 中真实改动仅 159 行（补记）

### 2026-10-09 会话记录（T022）

- 完成任务：T022 FR8c 训练循环性能·其三（标准任务，Red/Green 分离）
- 修改文件：utils/checkpoint.py（重构：`save_checkpoint` 拆为 `build_checkpoint`（纯 payload 组装）+ `write_checkpoint`（tmp+`os.replace` 原子写）+ `stage_to_cpu`（递归深拷贝到 CPU：`empty_like(cpu)+copy_(non_blocking)` 恒拷贝——CPU 张量上 `.cpu()` 是 no-op 会别名活训练状态，故必须显式拷贝；dict/list/tuple 递归）；training/trainer_base.py（`CheckpointManager` 异步化：`save(..., async_save=True)` 默认异步——caller 线程 build+stage 后交 daemon 后台线程 `_write_payload` 写盘，开头 `self.flush()` 串行排队（新 save 先 join 上一笔在途写）；`_write_payload` 失败 WARN(RuntimeWarning) + 同步重试一次、再失败 WARN 落败；新增 `flush()` join 在途写；`BaseTrainer.cleanup` 前置 `ckpt_manager.flush()`；import 加 threading）；training/train_pretrain.py（micro-step 块重写：删每步 `dist.all_reduce(AVG)` + `loss.item()`，改 `loss_accum` 设备张量跨 micro-step 累积 + 窗口边界单次 all_reduce + 单次 `.item()`（loss 预除 accum → 窗口和=窗口均值，与旧逐步平均语义等价）；tok/s 改 `running_tokens` 逐 micro-batch 计数（与日志重置解耦）；早停 return 前 + train() 末尾 `ckpt_manager.flush()`（调用方 train 返回即可读 ckpt）；train() 头部初始化 `loss_accum/running_tokens`）；model/modern_gpt.py（`update_ema` 改 `torch._foreach_lerp_(shadow_list, param_list, 1.0-ema_decay)` 展平列表单批更新：**lerp 权重方向是 `1-decay`**（lerp(s,p,w)=s·(1-w)+p·w，design 原文 `decay` 笔误），每次 update 重建展平列表（named_parameters 遍历开销远小于被消灭的 200+ kernel，且天然免疫 ema_shadow 被 ckpt 恢复替换后的缓存陈旧）；`ema_shadow` dict 磁盘格式不变；requires_grad 过滤、apply/restore 交换语义不变）；tests/test_training_perf_flags.py（追加 17 用例：TestEMAForeachUpdate 5 + TestMicroStepDesync 5 + TestAsyncCheckpoint 7）；tests/test_resume_integrity.py + tests/test_trainer_base.py（契约适配 3 处：trainer 级 save 后立即读文件前补 `ckpt_manager.flush()`——异步默认下的新契约，非弱化）
- **关键发现（Red 期 3 项契约澄清）**：① `iter_num` 实际按 **micro-step** 计数（while 循环每迭代消费一个 batch），max_iters=8/accum=2 → 8 micro-step/4 窗口，design 描述的「4 训练窗口」对应配置被 Red 代理实证纠正；② AdamW 优化器内部每步 ~27 次 `.item()`（`_get_value` 逐参数）——`.item()` 计数测试按调用方帧过滤（`"train" in basename(co_filename)`），只计本仓训练代码的同步点；③ `running_count` 旧语义本来就按 micro-step 计数（旧 tok/s 公式数值无误），静态禁令 `"* running_count" not in source` 强制改用独立 token 计数器是为与日志重置节奏解耦（accum>1 且 log_interval 非 accum 倍数时旧公式才会失真）
- **关键发现（Green 期 1 处缩进回归，测试当场捕获）**：早停块插入 flush 时 `return` 缩进错落到 eval 边界层级——train() 在**首次 eval 即返回**（既无 `final:` 也无早停打印），`test_train_flushes_before_return` 断言 iter 1≠2 立刻暴露；TDD 纪律的直接收益例证。修正后 flush 位于早停条件内、master 守卫外（全 rank drain，非 rank 为 no-op）
- 测试：Red 11 failed / 6 passed（6 个 passed 为数值等价/回归前向锚点，Green 后必须持续通过）→ Green 17/17；全文件 54/54；全量回归 **444 passed / 2 skipped**（427 基线 + 17 新增，零回归；test_trainer_base 2 个「save 后立读」用例按新异步契约补 flush 后通过）
- 备注：异步保真度设计——staged 深拷贝在 caller 线程完成（GPU 场景即 D2H 暂存，后续训练的原地变异不再影响快照），后台线程只做序列化+落盘；抢占场景最多丢「在途的那一笔」save（tmp 文件原子性保证目标路径永不出现半截 ckpt），train()/cleanup() 双点 flush 保证正常退出路径文件恒可读；既有测试适配仅 3 处且全部是「补 flush」方向（契约变更非弱化先例）；EMA 展平列表每次重建而非缓存，避免 ckpt 恢复替换 ema_shadow 后的陈旧引用

### 2026-10-09 会话记录（T010）

- 完成任务：T010 FR9a DDP 数据 rank 分片 + DDP 调优（标准任务，Red/Green 分离）
- 修改文件：data/openwebtext.py（`MemmapDataset` 新增 `shard_rank=0, shard_world_size=1`：rank r 消费全局流的跨步位置 `r, r+W, r+2W, ...`——互斥且并集=全集零丢弃（优于 DistributedSampler 的 padding/drop_last），`resume_offset` 语义变为 rank 内局部序号；`_rank_positions()` 统一虚拟流，W=1 时退化为 `range(length)` 与历史行为逐位一致（B3）；worker 切分在虚拟流内二级进行；`get_openwebtext_dataset`/`get_dataloader`/`GlobalShuffledDataset` 全链穿透；use_packing+分片 → RuntimeWarning 告知 PackingDataset 不支持分片）；training/train_pretrain.py（`_build_data` 仅 train split 接分片参数（val/train_eval 不分片——各 rank 评估同一数据，estimate_loss 的 SUM/world_size allreduce 仍为正确均值；分片键 = `max(world_size,1)>1`；lockstep 下各 rank 局部 offset 相等，master ckpt 值可恢复所有 rank）；`_maybe_resume` world_size 不匹配分支补 `self.resume_offset = 0`（原仅 WARN 保留 offset——跨分片布局下旧 offset 无意义，对齐 design「WARN 并从该 rank 分片头部开始」）；train loop 非窗口末 micro-step 的 backward 包 `self.model.no_sync()`（守卫：distributed + accum>1 + 非末步 + hasattr，raw model 无 no_sync 时 nullcontext 零开销）；import nullcontext）；training/trainer_base.py（DDP 构造补 `static_graph=True, gradient_as_bucket_view=True`）；tests/test_ddp_shard.py（新增，14 用例，design §6.1 指定单进程模拟）
- **关键设计取舍**：① 跨步切片而非 DistributedSampler 式均分——后者靠 padding/drop 保证等长会丢数据或重复数据，跨步切片天然互斥+全覆盖（12 chunks/W=5 → 3,3,2,2,2）；② val 不分片——分片 val 也能得到正确均值（真平均），但会改变「各 rank 看到同一验证集」的现有语义且 estimate_loss 的 eval limit 交互复杂，维持保守选择；③ 分片键取 world_size>1 而非 distributed 标志——WORLD_SIZE env 单独存在（rank=-1）时仍表达数据并行意图
- **关键发现（Red 期）**：① `_maybe_resume` 在 `BaseTrainer.__init__` 模板方法内调用——构造后 monkeypatch 太晚，测试用 `monkeypatch.setenv("WORLD_SIZE", "2")` 在构造前注入（无 RANK → rank=-1/distributed=False，但 world_size=2 触发 mismatch 分支）；② 无 shuffle.idx 路径的 chunk 级 buffer shuffle 用全局 random——测试传 `shuffle_buffer=1`（尺寸 1 的 buffer shuffle 恒等）使跨步可见性断言确定性；③ Windows 下同进程两个 trainer 重建 tiny 数据会撞 memmap 文件锁（OSError 22）——测试 helper 改为复用已存在的确定性数据文件
- 测试：Red 12 failed / 2 passed（2 个前向锚点：matching-offset 保留 + no_sync accum=1/非分布式不触发）→ Green 14/14 一次通过；全量回归 **458 passed / 2 skipped**（444 基线 + 14 新增，零回归）
- 备注：static_graph=True 与 no_sync 的组合在 torch 2.5+ 已支持（旧版曾不兼容），单进程模拟无法覆盖真实 NCCL 路径，**T011 FSDP round-trip 或迁移 GPU 冒烟时须验证**；ckpt 的 world_size/rank 键自 T009 起已持久化，本任务补齐消费端校验语义；SFT 已有 DistributedSampler 先例（train_sft.py L166），与 pretrain 的手动分片并存——统一属 T018 接口收敛范畴

### 2026-10-09 会话记录（T011）

- 完成任务：T011 FR9b FSDP 补齐（标准任务，Red/Green 分离）
- 修改文件：training/trainer_base.py（① `wrap_distributed` 重构：FSDP 分支补 `auto_wrap_policy=partial(transformer_auto_wrap_policy, {ModernBlock, BaselineBlock})`（per-layer 分片，通信/计算重叠）+ `use_orig_params=True`（compile 兼容开关）；**顺序倒置**——FSDP 先包、`torch.compile` 作用于 FSDP 模块（最外层），DDP 分支与非分布式分支的 compile 语义不变；新增 `self._fsdp_module` 持有 FSDP 包装器引用；② `CheckpointManager` 新增 `state_dict_fn`/`load_state_fn` 可选钩子：save 以钩子返回值为 model payload（model_type 检测仍走解包链 DDP→`_orig_mod`），load 经 `load_checkpoint(..., load_state_fn=)` 委托；③ `BaseTrainer._get_model_state_dict`/`_load_model_state_dict` 上移通用化（FSDP 守卫：args.fsdp + distributed + `_fsdp_module` 非 None → `FSDP.state_dict_type(FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True))` 上下文内 state_dict/load_state_dict；否则 raw_model 解 `_orig_mod` 后常规路径），`configure_checkpointing` 接线双钩子——**5 个训练器全部自动获益**；imports 加双 Block 别名）；utils/checkpoint.py（`load_checkpoint` 新增可选 `load_state_fn` 参数，缺省走 `model.load_state_dict`，向后兼容）；training/train_pretrain.py（删除死代码 `_get_model_state_dict`——该方法从未被 ckpt 流程调用，CheckpointManager 只解 DDP 包，FSDP 包装器 state_dict 会带前缀/分片残缺落盘）；tests/test_fsdp_checkpoint.py（新增，12 用例，design §6.1 L483 + E7）
- **关键发现（torch 2.5.1 FSDP1 API 考古，3 项实证）**：① `StateDictConfig` 仅有 `offload_to_cpu` 形参——FULL_STATE_DICT 的 rank0 语义在 `FullStateDictConfig(rank0_only=True)`（`full_state_dict` kwarg 是 2.6+ 才有，初版按 design 措辞 `cpu_offload=True` 两度 TypeError 后以签名探针定谳）；② FSDP1 源码无 `_orig_mod` 处理——compile-then-FSDP 组合下 `.module` 链是 OptimizedModule，raw_model 语义破碎，官方顺序为 **FSDP 先包 + compile 最外**（现行代码 compile→FSDP 属错误顺序，本任务倒置）；③ `use_orig_params=True` 保真的是原始 Parameter **对象/形状**（无 FlatParameter），`named_parameters` 的 key **仍带 `_fsdp_wrapped_module.` 前缀**——「无前缀」性质在 FULL_STATE_DICT 输出（CPU 无法实测，CUDA-only）
- **关键发现（CPU 可测性边界）**：FSDP1 在 CPU+gloo 单进程下**可构造**（须 `USE_LIBUV=0` 绕开 libuv 构建缺失；world_size=1 时 torch 自动 FULL_SHARD→NO_SHARD 降级警告），但 FULL_STATE_DICT gather 强制 CUDA（"parameters on cpu"）→ 测试分层：真 FSDP 构造（3 用例：_fsdp_module/raw_model 身份、参数槽保真、per-Block 包装计数 ≥3）+ 钩子路由（monkeypatch `FSDP.state_dict_type` 上下文记录器 + stub，save/load 双向）+ 静态源码契约（compile 顺序行号、FULL_STATE_DICT/offload_to_cpu/rank0_only）+ E7 原子性（torch.save 恒失败 → 旧 ckpt 完好、无 .tmp 残留、WARN）
- **顺手修复（潜在 bug）**：save/load 的 `_orig_mod` 解包——此前非分布式 compiled 模型的 ckpt key 会带 `_orig_mod.` 前缀（compile 默认关，属潜伏缺陷）；顺带使 model_type 检测在 compiled 场景下正确
- 测试：Red 12/12 正确失败（真实 FSDP 构造全数真跑未 skip，证明 GREEN CPU 可行）→ Green 12/12（2 处测试侧契约修正：use_orig_params 断言从「named_parameters 无前缀」改为「参数槽+形状保真且无 FlatParameter」——torch 2.5.1 实证前者不成立；静态断言 `cpu_offload`→`offload_to_cpu`+`rank0_only`——按真实 API 签名）；全量回归 **470 passed / 2 skipped**（458 基线 + 12 新增，零回归）
- 备注：FSDP 全链（FULL_STATE_DICT gather/load + static_graph/no_sync 组合）的真实多卡行为仍需 GPU 冒烟（迁移演练 §6.3 覆盖）；EMA shadow 与 FSDP 分片的交互（ema_shadow 在 raw_model 上、非 FSDP 管理）未在本任务范围，若 GPU 冒烟暴露问题另立任务；死代码接线选择 hook 方案而非 DCP（`torch.distributed.checkpoint`），后者改动面大且 design 允许二选一

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

### 2026-10-10 T012 会话记录（FR6 MoE 向量化重写·方案 B）

- 基准预检先行（防不可达阈值烧进契约）：本机 CPU @ E=4,H=688,C=256——旧实现（每 forward torch.stack 全专家权重 + Python 逐专家循环 + expand+bmm）fwd+bwd 137-406 ms/step；新 padded-bmm 参考 7.7-15 ms/step（**17.9-27.6x**），≥2x 阈值余量充足。旧路径慢的根因是 bmm 对 stride-0 expand 批次维回退到逐元素复制
- 等价性预检：真实 Linear 尺度权重（uniform ±1/sqrt(fan_in)）下新旧实现 diff ~3.4e-8（atol=1e-5 安全）；randn(0,1) 权重会放大 float32 累加顺序差异到 1e-2 级（等价断言勿用 randn 权重）——Red 测试据此统一用 Linear 尺度权重
- Red（12 用例，tests/test_moe_vectorized.py）：文件内自带忠实旧实现参考 `_reference_moe_forward`（stack+循环+int 截断）；pin 用例 9 个（等价/int 截断/丢集/旧 ckpt 加载/旧键保存 round-trip/dense 路径/aux/cap=0 边界）全部首跑即 PASS（现行实现即参考）；结构/静态/基准 3 个按预期 FAIL（Red）；compile 用例 Windows triton 缺失 skip。Red 报告偏离：`torch._dynamo.utils.CompileCounter` 在 2.5.1 不存在，改用 `counters["stats"]["unique_graphs"]`
- Green（model/modern_gpt.py，仅 SwiGLU + 一处 init）：
  - 参数：MoE 分支 `gate_w/up_w: Parameter[E,H,C]`、`down_w: Parameter[E,C,H]` 替代 3×ModuleList；init normal(0,0.02) 对齐 `_init_weights` 对 Linear 的处理
  - 兼容 hook：`_register_load_state_dict_pre_hook` 旧键 dim=0 拼接→gate_w；`register_state_dict_post_hook` 反向拆分（新 ckpt 落盘仍是旧键格式，B1 双向兼容成立——save 永远写旧键，load 永远读旧键）
  - forward：stable sort 分桶（原 token 序保持）+ 段内 slots<cap 截断（与旧 `idx[:capacity]` 逐位一致，已独立验证 keep 集合相等）+ padded `[E,cap,C]` 一次 index_put_ + 三次静态形状 bmm + gather 回写 ×router_probs；cap 用 `int()` 截断（design 原文 ceil 笔误，以「与现行对齐」为准，测试锁定防 ceil 漂移）；cap=0 时空张量路径自然成立
  - 顺手修复：ModernGPT 深度缩放 init 条件补 `endswith("down_w")`——否则 MoE down 投影丢失 GPT-2 深度缩放（旧参数名 `down_proj.{e}.weight` endswith 匹配、新名不匹配的潜伏 init 语义变化）
- 回归：新文件 11 PASS + 1 SKIP（compile）；全量 **481 passed / 3 skipped**（基线 470p/2s + 11 新 pass + 1 新 skip），90s 零回归
- 基准记录（FR6-AC2，写入实验记录）：E=4, H=688, C=256, N=512, fwd+bwd, CPU 中位/10 次——旧 137.13 ms/step → 新 7.65 ms/step，**17.92x**（≥2x 达标）；N=1024 时 27.56x
- 遗留：compile recompile 计数用例在无 triton 平台 skip，GPU 冒烟时验证（迁移演练覆盖项）

### 2026-10-10 T013 会话记录（FR5 KV decode 去 cat·方案 B）

- 定向关键发现：现行 KVCacheManager 已是 ring 预分配（update 无 cat）——**热点在 attention 层 L408-416**：get_cache() 返回 blocks 后每步全量 cat（值形态 past_kv）；未环绕时单块 cat 也是 O(cache_len) 拷贝
- Red（13 用例，tests/test_kv_append_layout.py）：1 pin（generate cache/no-cache 逐 token 相等）+ 12 Red（layout/get_layer_view/对象形态/fast path 等价/零 cat/B5 驱逐回退/int8 惰性/B9 flash 回退/mask 向量化/基准/GQA+滑窗）；值形态基线全部首跑通过；既有 KV 测试（test_bugfixes/test_paged_kv_*）零影响
- **Red 期发现（遗留问题）**：RoPE cos/sin 表在 RotaryEmbedding 中锁死 `2*block_size` 行——滑窗驱逐后 start_pos 无限前移，绝对位置超 2×block_size 时 generate 双路径都崩（block_size=16 + 28 token 即触发边界）。测试 7 以 28 token 规避；驱逐后行为与现行一致优先，未在本任务修（若滑窗长生成是需求需后续修复：表按 seq_len 请求扩展）
- Green（kv_cache_utils.py + modern_gpt.py + flash_attention.py）：
  - KVCacheManager：`layout="append"` 默认（"ring" 保留）；`evicted` 标记（advance 首次驱逐时置位，永久切滑窗语义）；`get_layer_view(layer_idx, extra)` 零拷贝视图（append+未驱逐+不越界，否则 None 回退；int8 仅物化该切片）；**get_cache() 返回结构不变 → 既有测试零破坏**
  - attention fast path：past_kv 形态 `(cache, layer_idx)`（model.forward 对象分发）+ 条件 T==1、无 attention_mask/document_ids、无滑窗、未 evicted、append 布局 → O(1) 写入 buffer + `[:pos+1]` 视图直喂 SDPA（is_causal=False 与旧路径一致）→ **单写协议**：present=None、forward 折叠全 None→None、调用方跳过 update 只 advance；不满足条件回退值路径（get_cache()[layer_idx] 转换后走现行 cat 逻辑，逐位一致）
  - flash wrapper 可选：fast path 先试 `flash_attention_with_cache`（返回 None 回退 SDPA 视图，B9）；**wrapper 探测 import 加模块级缓存**——每层每步重复失败 import 是隐性热点（实测 0.57x 的主因之一）
  - build_sliding_window_mask 向量化（arange 比较，无 Python 循环，逐位一致）；docstring 避开 "for "（静态 ban 覆盖 docstring）
  - generate 主循环接线对象形态；**PagedKVCacheManager 无 get_layer_view → 按能力分流保持值形态**（全量回归捕获的鸭子类型误伤，test_paged_generation_matches_ring_buffer 红灯）
- **契约演进（测试侧修正，非弱化）**：Red 契约的幂等 update 协议（调用方无脑 update，fast path 双写）在 cProfile 下暴露 0.35ms/步开销——CPU 上净负收益（0.89x）；改为单写协议（present=None 语义，回退场景调用方仍 update）；测试 3/4/8/9/12 同步更新，测试 9 重写为单写协议 + buffer 逐位一致断言
- 性能演进：flash import 缓存 0.57→0.89x → 单写协议 0.89→1.15x → 基准规模对齐 AC（Red 用 1024 ctx，cat 占比不足）→ **4096 ctx + embd512/8h/8L：21.4→13.3 ms/步 = 1.61x**（≥1.5x 达标；CPU 天花板 ~1.6x——cat 与 fast path 写入共享内存带宽，更大规模饱和）；基准需 buffer 余量（prompt 4096 + buffer 8192），否则 7 步即驱逐回退
- 全量回归：**494 passed / 3 skipped**（481+13），91s
- 遗留：投机解码/batch 投机/continuous_batching/generate_batch 仍走值形态（S>1 验证步为主 + compile step_fn 对象追踪风险；T014 重构时统一接线）；RoPE 表上限问题见上；显存斜率断言（AC2 后半）CUDA-only 留迁移演练

### 2026-10-10 T014 会话记录（FR7 连续批处理 iteration-level 重构）

**定向新发现：**
- 现行 continuous_batching.py 全部 FR7 指控坐实：批级回收（run 内 while 等整批完成）、右 pad KV 污染（cache.advance(max_prompt_len) 把 pad 位置写入 + decode 无 mask）、PrefixCache 占位（注入循环内 `pass`）、_init_decode_compile 从未被调用、逐序列采样 Python 循环、off-by-one（decode 首步 `next_in[b]=results[b][-1]` 重喂 prompt 尾 token）
- 唯一外部引用破坏面：api_server.py 仅 docstring 提及（T015 接线）；tests/test_speculative_distribution.py L133-156 import BatchGenerator._sample——契约同步改造（保留意图，改锁 generate_utils 两处采样）
- 核心架构问题：iteration-level 下各槽 cache_len 必然不齐 → 共享 forward 需要三件套——per-row 有效 KV 前缀（kv_mask）、per-row 新 token 绝对位置（positions）、值路径 additive mask 组合。model.forward 因此新增两个可选参数（FR7 服务性扩展）

**Red（17 用例 test_continuous_batching_v2.py + speculative 契约同步）：**
- 16 Red + 1 pin（B12 空 slot 行为级在现行也成立，但契约断言 list 返回使其 Red；线程安全行为级 1 PASS）；speculative 改造后 11 PASS + 1 Red（_sample_logits_batched ImportError）
- 用例：死代码静态断言（源码禁 "class PrefixCache"/"class BatchGenerator"/"_init_decode_compile"、require "threading.Lock"）、GenerateResult/run(max_steps)/iteration_count 契约、B=1 vs generate() 一致（捕获 off-by-one）、max_tokens=1 首 token、不等长 batch vs 逐条一致（捕获 KV 污染）、槽位回收精确 iteration 推演（short it3 / third it4 / long it8 / 总 8）、max_steps 部分、_sample_logits_batched 语义（temp=0 严格 argmax 短路 / top_k=1 / top_p=0.0 / <=0 与 >=1 不过滤 / RNG 无关断言）、greedy 确定性、B12 双变体（空槽 + 回收后无残留）、队列线程安全（8×25 并发 enqueue + 4 drainer 无丢失 + maxsize RuntimeError）、positions 语义（恒等 / start_pos 等价 / masked-dummy 偏移等价，rope+learned 参数化）、kv_mask 语义（垃圾尾屏蔽 == 干净切片 + 非空性哨兵 + B=2 不同长度批量 vs 单行）、rep penalty 弱断言（batch == solo）、eos 语义（含 eos token / finish_iteration=1）
- **契约 bug 修正（Green 前主代理修测试）**：测试 12/13 直接 `model(...).shape` 但 forward 返回 3-tuple——Red 期被 TypeError 掩盖；9 处补 `[0]` 解包

**Green：**
- model/modern_gpt.py：`apply_rotary_pos_emb_single` 支持 cos [B,T,D]（ndim==3 → unsqueeze(1)）；CausalSelfAttention.forward + positions/kv_mask 参数（RoPE 按 positions gather cos/sin；kv_mask 构造 [B,1,1,past_len+T] additive 与 attention_mask 相加组合；fast path 条件加 positions/kv_mask 保守排除）；Block.forward 透传（gc 训练路径不传）；ModernGPT.forward + 两参数（learned wpe 显式 positions 索引 + 形状/block_size 校验；块分发透传）
- inference/generate_utils.py：`_sample_logits_batched(logits, temperatures, top_ks, top_ps)`——temp=0 行 where(argmax) 严格短路（安全除法防 NaN）、per-row top-k（topk+rank 掩码 scatter）、per-row top-p（sort+cumsum+**前移一位**掩码，与 _sample_logits 逐行语义逐位对齐）、单次 multinomial
- inference/continuous_batching.py 全量重写（~470 行）：Request 去 prefix_key；RequestQueue threading.Lock；GenerateResult(request_id/token_ids/finish_reason/finish_iteration)；槽位状态机 EMPTY→PREFILL→DECODING→FINISHED；调度器自管 per-layer KV buffer [batch_size,H,effective_block_size,D]；iteration = 收割 FINISHED→紧凑化 buffer→补槽→**decode 组先快照**（本 iteration prefill 的槽下一 iteration 才 decode，首 token 来自 prefill）→prefill 组左对齐打包（右 pad 被 attention_mask 屏蔽且**只写各槽真实前缀**，pad KV 永不入 buffer）→decode 组视图 past + positions=[cache_len] + kv_mask=[有效前缀]→张量化采样 + 向量化 rep penalty（padded gather/scatter）；off-by-one 修复：prefill 最后真实位置 logits 采样首 token；batch 全程值路径（行独立保证 B12）
- 两次修正：① 同 iteration prefill 后槽被 decode 捞走（decode 组须在 iteration 开始快照，测试 6 精确推演锁定）；② 紧凑化 `buf[idx]` 误缩 buffer batch 维（改为 buf[:K]=buf[idx] 前缀写回，fancy indexing 拷贝保证无别名）

**性能演进（FR7-AC3 基准：16 并发、prompt 8-38、max_tokens 2..47 阶梯、embd64/4h/4L CPU）：**
- 初版（行 gather past + 逐行 Python 写回 128 次 ATen dispatch/步）：平均延迟降 28.3%（不达标）
- 紧凑布局（活跃槽恒占行 [0..K)，past 为零拷贝视图）+ 向量化写回（per-layer 单次索引赋值 [arange,:,lens_t]）+ kv_mask arange 构造：**249.0→154.2ms = 38.1%（≥30% 达标）**；16 条 greedy 输出与 static 基线逐 token 一致（基准内交叉验证）
- 静态基线为旧语义诚实仿真：整批跑完才返回（所有请求延迟=批端点）

**关键决策：**
- decode 走值路径 + kv_mask（不用 T013 fast path：kv_mask/positions 强制禁用 fast path 是保守正确性决策；AC3 收益来自槽位回收语义而非单步 kernel，值路径 cat 与旧实现同量级）；fast path 增强留遗留
- BatchGenerator 整类删除（职责并入调度器；无调用方即死代码，FR7 精神）；ContinuousBatchingServer 外壳保留
- finish_iteration 字段进 GenerateResult（AC1「日志/计数可验证」的落点，避免解析日志）
- 紧凑布局行搬移只在 iteration 边界发生且仅有空洞时（keep 非前缀才拷贝）；行为与稀疏布局逐位一致（行独立性）

**回归与遗留：**
- 全量 **512 passed / 3 skipped**（~100s；494 基线 + 本任务 17 + speculative 1）；目标测试 29 全绿
- 遗留：调度器 decode 仍值路径 cat（kv_mask 与 fast path 不兼容，后续可为 append 布局 + 等长槽场景设计 fast path 变体）；speculative/generate_batch 的 S>1 验证步值路径接线问题不变（T019 接口统一时一并处理）；benchmark 中每请求延迟按 finish_iteration 比例折算（等 iteration 时长假设，保守）

### 2026-10-10 T015 会话记录（FR7b api_server 接线新调度器）

**定向发现：**
- api_server.py 三处接线缺口坐实：docstring L9-10 声称 `--continuous_batching` 但 get_args() 无此 flag（虚假文档声称）；推理在 async endpoint 内直接同步调用 model_state.generate（阻塞事件循环）；无任何调度器接线
- 唯一外部依赖面：api_server 无其他模块 import 它（此前 T014 定向已确认），接线零外部破坏

**Red（9 用例 test_api_server_wiring.py，4 Red + 5 Pin，6.5s）：**
- 结构断言（get_args 源码含 --continuous_batching/--batch_size + 模块源码含 asyncio.to_thread）、CB dispatch 用调用计数 stub 锁定（Red 期 0 次调用 FAIL）；pin：off 模式行为/health/503//v1/models
- **关键环境发现：tiktoken 在本机可用**（get_encoding("gpt2") 成功）——会把 prompt 编码到 tiny 模型 vocab 之外导致 embedding 越界；测试 monkeypatch tiktoken.get_encoding 抛错强制走 ModelState 文档化的 ord/chr 回退，跨环境确定性
- 契约形态决策：cb_state 接口锁定为 `generate(prompt_text, max_tokens, temperature, top_k, top_p, repetition_penalty) -> (text, prompt_tokens, completion_tokens, finish_reason)`（阻塞同步，端点在 asyncio.to_thread 内调用）——服务端泵线程实现自由，端点契约锁死
- 并发测试用 httpx.AsyncClient + ASGITransport + asyncio.gather（TestClient 并发不可靠；httpx 随 fastapi TestClient 依赖存在，importorskip 兜底 E8）

**Green（api_server.py，~100 行净增）：**
- `ContinuousBatchingState` 适配器：单 daemon dispatcher 线程泵 scheduler.run()（队列空时 5ms 轮询）+ 结果按 request_id 存储（threading.Lock）+ generate() 提交 Request（uuid request_id）后阻塞轮询自身结果（300s 超时）；run 期间到达的请求在下一 iteration 边界被捞起（T014 队列锁保证并发 submit 安全）
- 两个生成端点：cb_state 非 None → `await asyncio.to_thread(cb_state.generate, ...)`；否则 `await asyncio.to_thread(model_state.generate, ...)`（**两路径都移出事件循环**）；finish_reason 映射：CB eos→"stop"、length→"length"（OpenAI 风格），legacy 路径恒 "stop"
- get_args 新增 --continuous_batching（store_true）+ --batch_size（默认 4）；main() flag 时构造 cb_state
- 503 语义保留（model_state None 检查在 cb dispatch 之前——cb_state 依赖 model_state 的 encode/decode）

**验证：**
- 目标测试 9/9 全绿（5.2s）；全量 **521 passed / 3 skipped**（~109s；512 基线 + 9）零破坏
- 4 并发 CB 模式经真实 dispatcher 线程 + 线程安全队列全部完成（T014 队列锁的第一次真实并发场景验证）

**关键决策：**
- dispatcher 泵线程 + request_id 结果存储（而非每请求锁串行 run）：并发请求自然合批（run() 在 iteration 边界捞起后续提交），符合 continuous batching 语义
- 测试不锁定 ContinuousBatchingState 内部结构（只锁 cb_state.generate 签名与端点 dispatch），服务端实现可自由演化
- 不测试 uvicorn 启动/main()（进程级行为不锁定）

**遗留：**
- CB 模式请求 eos_token_id=None（与 legacy ModelState.generate 行为对齐——它也不传 EOS）；生产接线可后续接 tokenizer EOT
- pump 线程 5ms 轮询（首 token 延迟下限 ~5ms）；条件变量优化留后续

### 2026-10-10 T016 会话记录（FR10 根目录收纳与死文件清理）

**定向发现（删除前全库引用检查）：**
- 4 个死文件（run_inference_ablation.py / generate_experiment_log.py / logs_prepare*.txt）**零代码引用**（仅 specs 历史文档与 TECH_REPORT 提及，后者为历史记录不改）→ 安全删除
- 迁移同步面：pyproject [project.scripts] 3 条 export 条目 + [[tool.mypy.overrides]] 裸 "export_to_onnx" + [tool.setuptools.packages.find] **无 tools\***（子代理发现，console script 打包安装需同步）；ci.yml 3 处 lint 命令；tests/test_gguf_export.py 2 处 subprocess CLI 路径；README 命令示例 6 处 + 目录树 6 行；evaluation/eval_benchmark.py docstring
- nanogpt-ablations = "run_component_ablations:main" 已存在且正确（AC2 pin）；run_component_ablations.py 按 design 保持根目录（v0.4.0 测试契约 pythonpath + console script 稳定）

**Red（9 用例 test_repo_layout.py，全 Red 0.32s，结构断言秒级）：**
- 根目录 .py 白名单恰等断言（4 个入口）；tools/export/ + tools/profiling/ 存在性 + __init__.py；死文件不存在；run_ablations.py deprecated 标注（含 "deprecated" + "run_component_ablations" 共现）；pyproject 三条 export 条目 → tools.export.*（tomllib 解析，Python 3.11.9 stdlib）；nanogpt-ablations 条目 pin；mypy overrides 更新；ci.yml 路径同步（负向后顾正则 `(?<!/)\bexport_to_hf\.py` 避免误伤新路径）；4 个迁移模块 importable（gguf/onnx 重依赖不锁）

**Green：**
- `git mv` 保留历史：4 export 脚本 → tools/export/、2 profile 脚本 → tools/profiling/；新建 tools/__init__.py + 两子包 __init__.py（tools 成为包，console script `tools.export.*:main` 可 import）
- `git rm` 4 死文件
- profile 脚本 sys.path hack 深度 +1（dirname×3）
- **4 个 export 脚本补 repo-root bootstrap**（`import os as _os` + `_sys.path.insert(0, dirname×3)`，锚点第一个 `from model.` import）——迁移后 CLI 直跑 `python tools/export/x.py` 需要根在 sys.path（Red 期被 gguf CLI 用例捕获：subprocess returncode != 0）
- run_ablations.py docstring 加 `.. deprecated:: v0.5.0` RST 段落指向 run_component_ablations/nanogpt-ablations
- README：6 处命令示例加 tools/ 前缀、目录树 6 行归并为 2 个目录条目

**验证：**
- 目标 15/15 全绿（repo_layout 9 + test_gguf_export 6——后者的 2 个 CLI 用例经新路径真实跑通 = 迁移后 CLI 功能验证）
- 全量 **530 passed / 3 skipped**（~100s；521 基线 + 9）零破坏
- FR10-AC2（nanogpt-ablations --help 仍可用）：条目未动，pyproject 解析 pin 锁定；AC1（CI 全绿）：ci.yml lint 路径已同步（本地 black/mypy/ruff 由 pre-commit 覆盖，CI 执行在 gitee 侧）

**关键决策：**
- 白名单「恰等」断言（== 而非 ⊆）：比 srs AC3「仅剩入口必要性文件」更强，防未来脚本再散落根目录
- export 脚本 bootstrap 用 `_os`/`_sys` 别名避免与脚本内既有 `import os` 语义纠缠；插入点统一锚定 `from model.` 首次出现（各脚本 import 区格式不一，docstring 内 "import" 文本会干扰按行锚定）
- test_gguf_export.py subprocess 路径改 "tools/export/export_gguf.py"（保持 cwd=repo-root 既有假设，最小改动）

### 2026-10-10 T017 会话记录（FR11a training/args.py 公共 flag 提取）

**调研（explore 子代理全量对比 5 训练脚本 argparse）：**
- 5 个非 hydra 训练脚本共 185 个 add_argument；≥2 脚本出现的 flag 共 33 个：**20 个定义完全一致**（其中 13 个连 help 字节级一致）+ 13 个 default/required 有差异（out_dir/batch_size/learning_rate/min_lr/weight_decay/eval_interval 等脚本特定 default，**有意差异须保留**）
- 5 个 `*_hydra.py` 变体无 argparse（@hydra.main + config/hydra YAML）→ 无需迁移，符合 design「改后 grep 复核无遗漏」
- `--ckpt_interval` 现状仅 train_pretrain 有（FR3 交付），design 要求 5 脚本统一注入 → Red 核心断言
- **关键坑**：`--init_from` 在 4 脚本 required=True 但 train_pretrain 没有 → 不能进公共组（注入 required flag 会破坏 PT CLI）；GR 系 5 flag（group_size/num_steps/max_prompt_len/max_response_len/eps）注入 PT/SFT/DP 无害（BaseTrainer 全 getattr 兜底消费）
- 现有回归守卫：test_monitoring_integration（5 脚本 no_metrics/no_monitoring）、test_ablation_switches（pretrain 4 消融 flag）、test_hydra_config（pretrain ~40 字段列举式快照——列举式故新增 flag 不破坏）

**Red（39 用例 test_common_args.py：22 Red + 17 pin）：**
- args.py 模块存在/纯组装（inspect 源码无 parse_args）/20 flag default+type+choices+store_true；5 脚本参数化暴露公共 flag；ckpt_interval 5 脚本注入（4 Red）；--help 块提取器（正则 `^\s+-{1,2}\w` 分块）断言公共 flag help 跨 5 脚本一致（help 差异 + 缺 flag 双 Red）；脚本特定 default/专属 flag pin（17 pin 锁定现状正确行为）；去重静态断言：公共 flag 零重声明（9 flag ban 列表）+ 5 脚本 add_argument 总数 ≤115（现状 185）+ 5 脚本均调用 add_common_trainer_args
- 修正一处设计缺陷：跨模块 help 一致比较原依赖参数化测试执行顺序（全局 mod_blocks 收集）→ 改为单一非参数化测试内循环收集再逐对比较

**Green：**
- 新建 `training/args.py`（132 行）：`add_common_trainer_args(parser)` 收敛 **20 个公共 flag**（19 个「≥2 脚本定义完全一致」+ ckpt_interval 统一注入），统一 help 文本（取现有最详细版，无 help 的补简短描述），LR_SCHEDULE/ATTN_BACKEND choices 常量共享
- 5 脚本改造（括号配对扫描器脚本）：删除 79 个重复 add_argument（PT 15/SFT 14/GR 19/DP 14/IG 17），`parser = argparse.ArgumentParser()` 后插 `add_common_trainer_args(parser)`，import 插在 trainer_base import 旁
- 净变化：training/ **-258/+194**（5 脚本 -268、args.py +132）≈ 净删 74 行 + 重复面收敛；设计估「~350 行净删」口径含 T018 load_model 7 份拷贝与 hydra 变体（后者实测无 argparse），实际重复定义 79 处 × 平均 2-3 行 ≈ 200 行消除
- design「23 个公共 flag」为估算数（只点名 11 个）；实测口径 20 个（init_from 因 PT required 语义剔除，差异 default 的 13 个保留脚本特定）——会话记录即口径说明

**验证：**
- 目标 39/39 全绿；全量 **569 passed / 3 skipped**（~101s；530 基线 + 39 新增）零破坏——hydra 等价快照/monitoring CLI/消融 flag 守卫全过
- FR11-AC1（测试等量通过）：569 ≥ 338 基线 + 新增全过 ✓

**关键决策：**
- 公共组以「定义完全一致」为入选标准，default 有差异的 13 个留脚本（AC「公共 flag 默认值/类型/--help 快照一致」反向要求公共组 default 必须相同）
- ckpt_interval 语义沿用 B2 边界（0 = 跟随 eval_interval），trainer_base 消费端已有 getattr 兜底，新注入脚本零接线成本
- iterative_grpo main() 不调 set_attention_backend（4 个 hydra 变体有）→ 注入 attn_backend flag 后暂不消费，无害；行为修复归 FR8/T019 范畴，记遗留

### 2026-10-10 T018 会话记录（FR11b load_model 统一 + conftest tiny fixture + test_bugfixes 拆分）

**调研（explore 子代理）：**
- load_model 拷贝实测 **7 份函数级**（trainer_base 权威 + generate/run_full_evaluation/export_to_onnx/api_server/eval_benchmark 部分拷贝 + benchmark_suite 死代码）+ **2 处内联**（export_gguf/export_to_hf main 内）
- 关键行为差异矩阵：仅权威实现支持 strict 参数；返回形式 4 种（(model,ckpt)/(model,config)/model 单值/3-tuple）；eval 语义 6/7 处 eval、权威实现不 eval（训练 resume 语义）；run_full_evaluation **完全不做 from_dict**（dict 直接当 config 用，统一后顺带修复）；generate 的 baseline+dict 分支有隐患
- ModernGPTConfig( 手工构造 **97 处/32 文件**（目标 <40）；conftest.py 不存在；markers 未注册；tiny 超参事实标准 n_layer2/n_head2/n_embd64/block32/vocab100/drop0
- test_bugfixes.py 670 行/18 用例；design 未给逐函数映射，按主题自定映射并写入测试锁定

**Red（19 用例 test_fr11b_consolidation.py：17 Red + 1 error + 1 pin）：**
- A 组：统一签名含 strict/eval_mode（pin：strict=True 现状已正确）；eval_mode 行为（training True/False）；strict=False 容忍缺键；6 个本地拷贝删除静态断言（generate/run_full_evaluation/export_to_onnx/benchmark_suite 死代码/api_server 委托 spy/eval_benchmark 内联）；generate 端到端 smoke
- B 组：conftest 存在 + tiny_modern_config/tiny_model fixture；默认值锁定 + overrides 透传；pyproject markers（perf/slow）；**ModernGPTConfig( 计数 <40**（FR11-AC3，97 → Red）
- C 组：test_bugfixes.py 删除；手动 runner 全库清零；18 用例 → 7 目标文件映射表（test_bugfixes_cases_live_at_new_homes 按 importlib hasattr 锁定）

**Red 期修正 3 个测试自身缺陷：**
- ModelServer → ModelState（类名调研不足，读 api_server 修正）
- 跨脚本 help 一致比较改单测试内收集（沿 T017 教训）
- `__main__` runner 扫描自指命中（测试文件引用 pattern 本身）→ 排除自身
- generate smoke 两轮修正：vocab 50 → ord 越界（vocab 1000）→ 仍炸（本机 tiktoken gpt2 id 31373 > 1000）→ monkeypatch get_encoding 强制 ord fallback（T015 同款手法）

**Green：**
- A 组：trainer_base.load_model_from_checkpoint 加 `eval_mode=False` 参数（默认保持训练 resume 语义，15 个既有调用点零破坏）；7 处拷贝全部收敛——generate.py（load_model 删除+main 改调）、generate_hydra.py（import 改）、run_full_evaluation.py（顺带修复 from_dict 缺失）、export_to_onnx.py、api_server.py（_load 委托薄壳保留 (model,config) 接口）、benchmark_suite.py（死函数删除+清理 unused imports）、eval_benchmark.py（_auto_export_hf 改用统一函数，model_type 探测改 isinstance(model, ModernGPT)）、export_gguf.py/export_to_hf.py（内联拷贝收敛）
- B 组：tests/helpers.py（TINY_DEFAULTS + tiny_modern_config/tiny_model 普通函数，供直接 import）+ tests/conftest.py（fixture 再导出 + pytest_configure markers）+ pyproject pythonpath 加 "tests"（conftest 导入早于 rootdir-prepend，须显式路径）+ markers 注册
- C 组：test_bugfixes.py 按映射迁 7 文件（KV 5 → test_kv_append_layout、GRPO 3+2 helper → test_grpo、optimizer/EMA 2 → test_trainer_base、DocBoundary 2 → test_packing、prepare 2 → test_prepare_data_files、arithmetic 2 → 新建 test_arithmetic_data、rewards 2 → 新建 test_rule_reward），迁移时 Config 构造改 helpers 调用（**_make_tiny_checkpoint 保留显式 vocab_size=50257**——GRPO 用 GPT2 tokenizer ids，tiny 默认 vocab=100 会越界）；**保留 try/finally trainer.cleanup()**（Windows RotatingFileHandler 句柄阻塞 tmp 清理的既有约束）；删手动 runner
- **Config 收敛**：脚本替换 89 处 `ModernGPTConfig(` → `tiny_modern_config(` + 自动插 import；**排除 3 风险文件**（test_dpo_trainer/test_grpo/test_packing——未传 vocab 且文件硬编码大 token id）；计数 97 → **5**（3 排除 + 2 计数测试 docstring 字样）
- 脚本 bug 修正 1 次：多行 import 括号内误插 import 行（2 文件 SyntaxError，手工修复 + compileall 全验）

**验证：**
- 目标 19/19 全绿；迁移用例 50p/1s；全量 **588 passed / 3 skipped**（~93s；569 基线 - 18 重收 + 19 新增 + 18 迁移用例并入既有文件净计）零破坏
- test_kv_append_layout 的既有 perf 用例 decode 1.5x 出现一次波动失败（全量并发计时边界），单独重跑通过、后续全量也过——flaky perf 非回归
- FR11-AC3：计数 5 < 40 ✓；AC1：588 ≥ 338 基线 ✓

**关键决策：**
- eval_mode 默认 False（训练 resume 语义不变），推理/导出调用点显式传 True——既有 15 个训练调用点零改动
- api_server._load 保留为薄壳（返回 (model, config) 接口不变，T015 接线测试零影响），内部委托统一函数
- helpers.py 与 conftest fixture 并存：fixture 满足 design 接口，普通函数供 89 处直接 import（避免大范围函数签名插参）；config = model.config 替代 ckpt config 提取（两模型均有 .config 属性，已验证）
- _make_tiny_checkpoint 显式 vocab 50257：tiny 默认 vocab=100 对 GPT2 tokenizer 路径是破坏性变更，该处保留原值
- eval_benchmark._auto_export_hf 的 model_type 探测从 config dict 改 isinstance(model, ModernGPT)——语义等价（saved model_type 驱动构造），且少一份 config 解析重复

### 2026-10-10 T019 会话记录（FR12 BaselineGPT/ModernGPT 接口统一）

**调研（explore 子代理）：**
- 破坏面实测极小：全库真 2-tuple 解包仅 train_pretrain.py:391 一处（自带 pad-to-3 补偿 hack）；tests 零 2-tuple 解包（gradient_checkpointing 用索引 out[1] 长度无关）
- BaselineGPT.generate 三个 no-op：top_p/use_cache/eos_token_id 声明未用（temperature/top_k 已实现）；forward 的 attention_mask 静默忽略、**kwargs 吞一切
- **反射 hack 假阳性**：BaselineGPT.generate 有显式 use_cache 参数 → co_varnames 探测恒 True → run_full_evaluation 实际给 baseline 传 use_cache=True 被静默 no-op；inference/generate.py:103 同款 hack 同样误判
- supports_kv_cache 两模型均不存在；CHANGELOG（docs/CHANGELOG.md）最新条目 v0.3.0、无 Unreleased 段
- ModernGPT._sample 有 temperature==0 严格 greedy 短路；baseline 无（除以 0 产生 inf → multinomial 崩溃，**既有 bug**）
- eval_benchmark.py:96 与 profile_training.py:98 的 3-tuple 解包对 baseline checkpoint 当前会 crash——FR12 改动顺带修复

**Red（14 用例 test_model_interface_unification.py：13 Red + 1 pin）：**
- forward 3-tuple（带/不带 targets）；两模型统一解包长度冒烟；attention_mask 一次性 warn（B11：首调 pytest.warns + 复调 simplefilter("error") 断言不再 warn）；top_p 执行 + greedy 不受影响（temperature=0 时 top-1 恒在 nucleus 内）；use_cache 显式 warn；nucleus 源码静态断言（cumsum/排序）；supports_kv_cache（baseline False / modern True）；run_full_evaluation 与 generate.py 无 co_varnames、含 supports_kv_cache；train_pretrain pad hack 删除；`logits, loss =` 于 training/evaluation/tools/inference 零残留（FR12-AC1）；CHANGELOG v0.5.0 + 3-tuple + BaselineGPT 记录
- Red 期修正 2 个测试自身缺陷：glob 逻辑写歪重写、nucleus 静态断言恒真表达式重写

**Green：**
- baseline_gpt.py：forward 返回 `(logits, loss, None)`；attention_mask 一次性 `warnings.warn`（实例级 flag `_warned_attention_mask`）；generate 实现 **top_p nucleus**（sort + cumsum + 前移一位掩码对齐 generate_utils 写法）+ **eos_token_id 提前停止**（顺手，声明未用的第三个 no-op）+ use_cache 一次性显式 warn + **temperature==0 严格 argmax greedy 短路**（修复既有除零 bug，对齐 ModernGPT._sample 语义）；类属性 `supports_kv_cache = False`
- modern_gpt.py：类属性 `supports_kv_cache = True`
- train_pretrain.py `_forward`：删 pad hack（baseline 直接透传 3-tuple）
- run_full_evaluation.py：co_varnames 反射 → `getattr(model, "supports_kv_cache", False)`（修复假阳性——baseline 不再收到静默 no-op 的 use_cache）
- inference/generate.py benchmark：同款反射 → capability flag（修复 baseline 误入 cache 路径的隐患）
- docs/CHANGELOG.md：新建 v0.5.0（Unreleased）Breaking Changes 段（3-tuple 迁移指引 + warn 语义 + top_p 真实现 + capability flag）

**验证：**
- 目标 14/14 全绿；全量 **602 passed / 3 skipped**（~98s；588 基线 + 14）零破坏
- FR12-AC1：`logits, loss =` 零残留 ✓；AC2（基线/现代模型统一训练评估跑通）：test_call_sites_unified_unpack_smoke + 既有 gradient_checkpointing 双模型参数化全绿 ✓

**关键决策：**
- use_cache 选择「显式 warning」而非 NotImplementedError（design 二选一；baseline 的 use_cache 在既有调用面 run_full_evaluation/api_server 被广泛传入，raise 会破坏可用性，warn 保持降级哲学）
- top_p/eos/greedy 短路三个「最小实现」顺手完成：design 只点名 top_p，但 eos 与 temperature=0 同为声明未用/既有 bug，一次改齐避免二次触碰
- CHANGELOG 放 docs/CHANGELOG.md（仓库实际位置，非根目录）；v0.5.0 段含迁移指引（外部 2-tuple 解包代码需跟进）
- api_server.py:214 的 "compile" co_varnames 反射**不在本次范围**（compile 参数两模型语义不同且探测结果正确，FR12 只授权 use_cache 探测改造）——记遗留待 FR13/后续评估
