# [AR-v0.7.5-precision-scaling-data] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.7.5-precision-scaling-data |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-14 |

> 用例编号对齐 design.md §6.6 的 ST-C1~C9（勘误与收口 ST 级用例），另补 ST-010~ST-013
> 覆盖 FR2/FR4/FR6 三工具端到端实测与 FR1/FR3/FR5 手册互证（srs §3 验收标准的
> 黑盒验证）。自动化证据引用：tests/test_{precision_probe,fit_scaling,data_probe}.py
> （50 用例）、T007 全量实测（EXPERIMENTS.md §6.10）、review 两轮记录。

## 测试用例列表

### ST-001：主题 6 锚点核验

**关联需求：** srs.md §3.7 FR7-1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 勘误后的 ai-infra/README.md 主题 6 行

**测试步骤：**
1. 读 README.md 主题 6 行引用的文件+函数名
2. 与 model/modern_gpt.py / tools/experiments/run_inference_suite.py 实际位置 grep 对照

**期望结果：**
- Then 锚点（generate(draft_model/draft_tokens)、_verify、run_inference_suite.py、api_server.py）全部存在

**实际结果：** README.md:23 含 `generate(draft_model/draft_tokens)`+`_verify` 锚点；model/modern_gpt.py:1528 `def _verify(`、generate(draft_model/draft_tokens) 路径与 run_inference_suite.py E-B-3 suite 均核实存在；api_server.py 在仓根

**状态：** PASS

---

### ST-002：B 组命令 argparse 核验

**关联需求：** srs.md §3.7 FR7-2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 勘误后的 RUNNING_EXPERIMENTS.md E-B-3 命令

**测试步骤：**
1. 解析命令的每个 flag（--checkpoint/--output_json/--draft_iters/--draft_out）
2. 与 run_inference_suite.py argparse 定义集合比对

**期望结果：**
- Then 0 个不存在参数（E-F-4 教训固化）

**实际结果：** 四 flag 全部存在于 run_inference_suite.py:22-29 argparse（T008 逐参核对；原 --speculative/--draft_model/--draft_len 三参确认不存在于 inference/generate.py:19-51）

**状态：** PASS

---

### ST-003：注记结构检查

**关联需求：** srs.md §3.7 FR7-3
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given ADR-006 与 TECH_REPORT §10.3 勘误后文本

**测试步骤：**
1. 检查 ADR-006 状态行 superseded 注记 + 决策原文保留
2. 检查 TECH_REPORT §10.3 交叉注存在 + 原估算数字未改动

**期望结果：**
- Then 注记清晰且不破坏原文档结构

**实际结果：** ADR-006 状态行「Accepted（决策第 4 条已被取代）」+ Superseded 注记块（决策第 1-4 条原文保留）；TECH_REPORT §10.3 交叉注在 §10.3.2 代码块后，原「总计 ≈ 694 MB」数字未改动（实测 grep 命中）

**状态：** PASS

---

### ST-004：链接存在性 100%

**关联需求：** srs.md §3.8 FR8-1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 索引更新后的 ai-infra/README.md 与 README.md 及本 AR 改动文档

**测试步骤：**
1. 脚本提取全部本地 markdown 引用并核验存在性

**期望结果：**
- Then 引用的仓内文件 100% 存在

**实际结果：** ai-infra/README.md 26/26 存在；9 个改动文档（EXPERIMENTS/RUNNING_EXPERIMENTS/两 README/三手册/ADR-006/TECH_REPORT）合计 119 引用 0 缺失（两轮脚本核验）

**状态：** PASS

---

### ST-005：全量回归零回归

**关联需求：** srs.md §3.8 FR8-2
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given 全部任务完成（T001-T011 + review 修复）

**测试步骤：**
1. `python -m pytest tests/ -q` 全量执行

**期望结果：**
- Then 711+2 基线零回归（新增测试全过或合规 skip）

**实际结果：** **760 passed / 3 skipped**（2026-10-14 实测，171s）＝基线 711 passed/2 skipped + 本 AR 新增 49 passed/1 skipped（47 契约用例 + 3 review 补充 abort 用例；1 skip 为 Windows 只读目录 skipif 合规），零回归

**状态：** PASS

---

### ST-006：gitee 同步验证

**关联需求：** srs.md §3.8 FR8-3
**测试类型：** 正常路径
**优先级:** Medium

**前置条件：**
- Given AR 交付完成

**测试步骤：**
1. `python tools/gitee_mirror.py`（proxy 环境变量 + PAT）
2. 核对远端 refs 与本地镜像 HEAD 一致

**期望结果：**
- Then 镜像链推送成功

**实际结果：** T011 轮同步成功：102 链元素分批推送（每 HTTP POST ≤70KB 过公司代理）、main push ok、mirror head 164c048d（2026-10-14）；review 修复提交（24e613c/655c6b7）与归档提交按用户既定序列「归档后 gitee 同步」执行终同步

**状态：** PASS

---

### ST-007：手册-AMP 逐行对照

**关联需求：** srs.md §3.1 FR1-2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given numerical-precision.md §3 的 AMP/GradScaler 机理描述

**测试步骤：**
1. 与 training/trainer_base.py `build_amp_context`/`enable_tf32` 逐行对照

**期望结果：**
- Then 逐行一致（autocast 语义、GradScaler 循环、TF32 开关位置）

**实际结果：** 手册 §3「AMP 逐行（autocast 进程内 + GradScaler 循环）」与 trainer_base.py:49 `enable_tf32`、:208 `build_amp_context` 锚点一致（T001 交付时逐行核对；ST 复审 grep 命中 L49/L54/L208 系列锚点；review S1 复核 YES）

**状态：** PASS

---

### ST-008：手册-工具数值互证

**关联需求：** srs.md §3.1 FR1-1/FR1-3、§3.3 FR3-1~3、§3.5 FR5-1~3
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 三手册引用的全部数值断言

**测试步骤：**
1. 与 docs/results/*.json 数据文件逐一对照

**期望结果：**
- Then 数值一致（2.10×/16B/1e-2/记账三件套/MIGRATION §3 判据）

**实际结果：** 精度手册 2.10× ↔ system.json bf16 147,727.89 tok/s；bf16 容差 1e-2 ↔ triton_kernels.json consistency；16B/param ↔ memory-flops §2.1；scaling 手册记账三件套 ↔ budget.json（corpus 1,244,464,437/consumed 589,824,000）与 ablation-summary.json 七组；数据手册 EOT 8.87e-4/均长 1128 ↔ MIGRATION §3；T007 实测互证链闭合（EXPERIMENTS §6.10 三表）

**状态：** PASS

---

### ST-009：三绿零新增

**关联需求：** srs.md §3.8 FR8-2（T011 口径）
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given CI lint 三件套口径

**测试步骤：**
1. black --check（line-length=100）/ ruff（E,F,I,B,UP）/ mypy 按 CI 范围执行

**期望结果：**
- Then 零新增

**实际结果：** CI 口径实测：black 86 files unchanged、ruff all passed、mypy Success: no issues found in 86 source files（2026-10-14）；三个测试文件 black/ruff/mypy 亦全过（review 第 2 轮复审证据）

**状态：** PASS

---

### ST-010：precision_probe 端到端实测

**关联需求：** srs.md §3.2 FR2（三条验收标准）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 本机纯 CPU 环境

**测试步骤：**
1. full 模式运行 `precision_probe.py --output-json docs/results/precision_probe.json`
2. 核对耗时、JSON 产出、tolerance_refs 口径
3. pytest 测试套件执行

**期望结果：**
- Then 全部实验产出 JSON（≤2 分钟）；bf16 容差口径与 triton_kernels.json 一致并记入 meta；pytest 全过

**实际结果：** full 3 秒（≪2 分钟界）；meta.tolerance_refs={bf16_atol:0.01, bf16_rtol:0.01, source:triton_kernels.json}；tests/test_precision_probe.py 14 用例 13 passed/1 skipped（Windows skipif 合规）

**状态：** PASS

---

### ST-011：fit_scaling 端到端实测

**关联需求：** srs.md §3.4 FR4（三条验收标准，含 Round 4 精化）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 已有 7 组数据（ablation-summary.json）与本仓 budget 配置

**测试步骤：**
1. full 模式运行 fit_scaling.py
2. 核对拟合残差报告与 compute-optimal 对照
3. pytest 测试套件执行

**期望结果：**
- Then 产出 JSON 且残差合理（R²>0.9 或诚实记录不达）；对照比例与手册一致；pytest 全过

**实际结果：** full 1 秒；真实 7 点**诚实记录不达**：converged=True 但 R²≈-1.7e-10、Jacobian cond=inf → param_uncertainty.reliable=false + limitations 第四条「参数与不确定度不具引用价值」+ summary "not meaningful"（srs FR4 Round 4 精化的「诚实降级」路径，escape 条款兑现）；合成联合式 α_N 恢复 0.65% 误差证明拟合器正确；tokens_per_param 24.9 与手册 §6.2 双口径一致；tests 18 用例全过（17+1 abort）

**状态：** PASS

---

### ST-012：data_probe 端到端实测

**关联需求：** srs.md §3.6 FR6（三条验收标准）
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 本机 train.bin 存在（17,853,806,718 B）

**测试步骤：**
1. full 模式运行 data_probe.py
2. 核对真实+合成实验产出与 EOT 频率对照 MIGRATION §3
3. pytest 测试套件执行

**期望结果：**
- Then 真实+合成两类实验产出 JSON（≤2 分钟）；pytest 全过零大文件依赖；EOT 频率与判据表量级一致

**实际结果：** full 7 秒；bin_profile（真实，200,003 token 跳跃采样）+doc_length_dist（14,052 docs）+4 合成实验齐备；**EOT 9.20e-4 vs 判据 8.87e-4 偏差 4% 同量级**、doc 均长 1138.4 vs ~1128 贴线；tests 17 用例全过（16+1 abort，零 17GB 依赖——守卫降级用例覆盖 bin 缺失分支）

**状态：** PASS

---

### ST-013：手册-生产代码语义对照

**关联需求：** srs.md §3.5 FR5-1/FR5-3
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given data-pipeline.md 的 packing/shard 机制描述与 MinHash/Jaccard 去重推导

**测试步骤：**
1. 与 data/openwebtext.py（PackingDataset/_build_samples/MemmapDataset）逐锚点对照
2. 与 data/dedup.py（MinHash L31/LSH L69/Jaccard L117）语义对照

**期望结果：**
- Then 逐行锚点一致；去重语义一致

**实际结果：** 手册锚点 openwebtext.py L110/L326/L349/L485 与 dedup.py L31/L69/L117 全部核实（T003 交付时逐锚点核对+review S1 复核）；data_probe 实调用生产 MinHashDeduplicator 且 keep/drop mask 语义一致（T006 修复记录 + review D4 核实）

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-14 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | 精度手册（表示学/AMP/存量数据） | ST-007, ST-008, ST-010 | PASS |
| §3.2 FR2 | precision_probe 工具 | ST-010, ST-008 | PASS |
| §3.3 FR3 | scaling 手册 | ST-008, ST-011 | PASS |
| §3.4 FR4 | fit_scaling 工具 | ST-011 | PASS |
| §3.5 FR5 | 数据手册 | ST-008, ST-013 | PASS |
| §3.6 FR6 | data_probe 工具 | ST-012, ST-008 | PASS |
| §3.7 FR7 | 勘误三件 | ST-001, ST-002, ST-003 | PASS |
| §3.8 FR8 | 索引与质量收口 | ST-004, ST-005, ST-006, ST-009 | PASS |

**需求覆盖率：** 8 / 8（100%；srs 24 条验收标准经 design §6.7 追溯表全映射到 UT/ST 用例）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 9 | 9 | 0 | 0 |
| 边界条件 | 0 | 0 | 0 | 0 |
| 异常处理 | 0 | 0 | 0 | 0 |
| 回归测试 | 2 | 2 | 0 | 0 |
| 端到端实测 | 2 | 2 | 0 | 0 |
| **合计** | **13** | **13** | **0** | **0** |

> 单元/契约层：50 用例（49 passed/1 skipped）；全量回归 760 passed/3 skipped 零回归；
> 三绿 CI 口径全过。异常处理与边界条件类在本 AR 由 UT 层覆盖（design §6.1/§6.4：
> 数据源缺失/不收敛/非法入参/中断注入/极小 bin 退化等 20+ 用例），ST 层不重复立项。

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | quantize_model 无生产调用方（model/quantization.py） | 延后处理：tasks.md 延迟项登记，归 v0.9.0 量化总论 |
| Minor | mypy pyproject.toml unused sections note（历史存在，非本 AR 引入） | 延后处理：不影响 CI 三绿判定 |

### 结论

> **Go**：FR1-FR8 需求覆盖 100%（8/8），13 个 ST 用例全 PASS，无 Critical/Major 缺陷；
> 全量回归 760/3 零回归、三绿零新增、链接 100%、gitee 同步机制验证成功；
> 2 个 Minor 已登记延后（用户既定裁定：quantize_model 归 v0.9.0）。
> 建议归档 AR-v0.7.5，归档提交后执行 gitee 终同步。
