# [AR-v0.7.0-ai-infra-kernels] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.7.0-ai-infra-kernels |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（设计定稿：D1-D7，design 门控第 3 轮 PASS 2026-10-14）|
| 创建日期 | 2026-10-13 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | FR1 docs/ai-infra/attention-kernels.md 手册（SDPA 生态/flash tiling+online-softmax 推导↔ring_attention.py 逐行对照/GQA 三路径深潜含 KV 物化全案/MoE grouped GEMM static-shape 教学；存量数据嵌入：5.8×/12,335/−75%/1160×） | - | passing | 2026-10-14 完成：8 节，推导↔代码对照表（ring L145-156 六行）、数据全溯源（moe_kernel.json/MIGRATION §8.2/EXPERIMENTS §3.2/roofline.json）、测试指针 5 组已验证存在、思考题 3 |
| T002 | FR2 docs/ai-infra/compile-stack.md 手册（dynamo→inductor→cudagraphs 三层/dynamo 层 recompile 实测演示/inductor fusion 理论/triton 版本断裂案例教学化/compile 1.59×与负结果分层归因/FSDP 包裹顺序） | - | passing | 2026-10-14 完成：7 节，dynamo 内联实验 3 段复制即跑（explain/计数/fullgraph 约束）、断裂案例含版本矩阵表、摊平边界归因、E-F-4 四要素对齐 F 组口径 |
| T003 | FR3+FR4 Triton 微实验（TDD）：tests/test_triton_kernels.py（eager 参照纯函数 CPU 单测+强度公式+形状校验+importorskip/skipif 用例）→ tools/infra/triton_kernels.py 单文件（fused SwiGLU/fused RMSNorm/GQA decode 三 @triton.jit kernel——D2 完整复刻 _decode_gqa_attention（scores+softmax+×V，沿 S 分块在线 softmax）+ eager 参照 + fp32/bf16 两口径一致性（atol=1e-5,1e-4 / 1e-2,1e-2）+ 算术强度标注 + A/B cuda.Event 中位数计时 + stdout Markdown 摘要 + CPU 冒烟降级（D7：无 CUDA/triton→eager 冒烟 exit 0 不写 JSON）+ JSON 原子落盘） | - | passing | 2026-10-14 完成：TDD Red（子代理 19 测试函数/28 用例点）→ Green（修 GQA head 索引漏乘 rep、bench swiglu bf16 量级 0.25→0.0625）→ 28/28 绿 + 三绿 + 全量 711/2 零回归；W1 负例已参数化落实 |
| T004 | FR3 验收+FR5：本机实测三 kernel（≤5min）→ docs/results/triton_kernels.json；EXPERIMENTS.md 新增记录节（含能说/不能说边界）；RUNNING_EXPERIMENTS.md 新增 E-F-4 inductor 云端 SOP（四要素+状态表+更新历史） | T003 | passing | 2026-10-14 完成：全量 bench 18 组（3 kernel×3 档×2 dtype）全 allclose，<2min；§6.9 三教学点（RMSNorm 反超 1.3-2.9× / 朴素 GEMM 0.12× / GQA 随 S 劣化）+诚实边界；E-F-4 四要素+状态表+更新历史齐备 |
| T005 | FR6a 索引更新：ai-infra/README.md 主题 2/3 转「已交付」+挂手册链接；修正「7 篇 ADR」→6 篇笔误；链接存在性检查 | T001, T002 | passing | 2026-10-14 完成：主题 2/3 转本期已交付（锚点换 attention/compile 手册+Triton 微实验+§6.9+E-F-4）；L45 7→6 篇；路线图 AR-v0.7.0 标本期；14 个引用文件 Test-Path 全 OK（ADR 实数 6：docs/adr 001-006 + 目录 README） |
| T006 | FR6b README.md 文档索引挂两本新手册 + badge 测试数更新（681+→实际值） | T005 | passing | 2026-10-14 完成：文档索引表新增 attention-kernels/compile-stack 两行（L535 后）；badge+L32+L167 681+→711+；L510 600+→700+ |
| T007 | FR7 质量收口：全量回归零回归（基线 683 passed / 2 skipped）+ 三绿（新文件已在 CI 范围）+ gitee 同步 + 链接终检 | T001-T006 | passing | 2026-10-14 完成：711 passed / 2 skipped 零回退（147.85s）；black 164 files unchanged / ruff All checks passed / mypy 零新增（19 处遗留全在 gitee_mirror.py+run_component_ablations.py，db3b7ff 前旧账）；链接终检含 E-F-4 命令目标（profile_inference.py/test_moe_vectorized.py::test_compile_no_recompile_growth）全过；gitee：分叉处理——云端应急 checkpoint 450f8ca 钉 _cloud_ckpt 后 force-push main=ef8359a（镜像 89b0787），_sm1..228 清理完毕，远端剩 main/_cloud_ckpt/_cloud_origin/_mirror_side/notes |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-13 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（7 YES + 2 Minor WARN）
- 修复的问题：G2 Minor——srs §3.7 FR7 补「描述/触发条件/期望行为/异常处理」小节（结构对齐 FR1-FR6）；G5 Minor——T006 备注 badge 前提数字修正（681+→实际值，非 683+）
- 事实主张核查：代码锚点行号（GQA 三路径 163-297/MoE 988-1046/ring_attention/attention_utils）、6 篇 ADR 笔误、5.8×/12,335/−75%/1.59×/1160× 数据、triton 3.8.0 环境断裂、683/2 基线、F 组 SOP 格式——100% 与仓库一致
- 审查代理：sdd-gate-reviewer

### 2026-10-14 design 门控记录

- 门控结果：FAIL（第 1 轮）
- 失败项：G2（Important）——FR3 期望行为 4 项未入设计（bf16 一致性口径/算术强度标注+JSON 强度字段/stdout Markdown 摘要/无 CUDA·无 triton 的 CPU 参照冒烟路径），且 design（bench 无 CUDA 报错退出）与 tasks T003（CPU 冒烟降级）与 srs 三方口径不一；D5 误称 fp32-only 为「srs FR3 既定」
- Minor WARN：G3/G5/G15（接口异常未逐一声明、swiglu/rmsnorm 无形状校验用例、CLI 形式与 srs `[--quick]` 有差）；G4（无边界矩阵）；G6（无 PlantUML）；G8（未声明覆盖率目标）；G12/G14（文档型 FR 验收未清单化）；事实核查 3 处（1.59× 源应为 EXPERIMENTS §6.4 与 inference.json 双源；roofline.json 实无 meta/measurements 键应改「松散承袭」；「主题 4（Triton）」编号错误应为主题 3=compile/Triton）
- 修复计划：design.md §4.3 补齐四项并统一 CPU 冒烟三方口径、修 D5；§4.3.5 CLI 对齐 srs 触发形式；§4.3.6 补边界矩阵+PlantUML；§4.4 补 4 用例（swiglu/rmsnorm 形状、强度公式、CPU 冒烟模式、bf16）+覆盖率口径；§6 补验收核验清单；修正 3 处事实引用
- 审查代理：sdd-gate-reviewer

### 2026-10-14 design 门控记录（第 2 轮）

- 门控结果：FAIL（第 2 轮）
- 已修复验证：第 1 轮全部 FAIL 项与 Minor WARN 核实闭环（FR3 四项/三方口径/D5/is_causal/cudagraphs/边界矩阵/流程图/覆盖率/验收清单/CLI 形式/3 处事实引用），事实核查 16 项 15 项一致
- 失败项：G13（Important）——「测量中断（exit 1 不写部分 JSON）」与「CPU 冒烟断言失败（exit 1）」两条异常分支在 §4.4 无对应用例
- Minor WARN：G15（argparse exit 2 无触发用例、--selftest 入口无用例）；事实核查（E-F-4 四要素措辞应为「环境/命令/产物/判据」，对齐 RUNNING_EXPERIMENTS.md L90 与 srs FR5 口径）；tasks.md L7 元数据过期
- 修复计划：§4.4 补 test_bench_interrupt_no_partial_json / test_cpu_smoke_assertion_failure / test_cli_usage_error_exit_2 三用例（--selftest 注明由 T004 本机实测覆盖）；§4.2 §7 与 §4.5 E-F-4 措辞对齐「环境/命令/产物/判据」；tasks.md L7 更新
- 审查代理：sdd-gate-reviewer

### 2026-10-14 design 门控记录（第 3 轮）

- 门控结果：PASS（第 3 轮）
- 审查项数：15 项全 YES
- 修复的问题：第 2 轮 G13（Important，测量中断/CPU 冒烟断言失败两条异常分支无用例）→ §4.4 补 test_bench_interrupt_no_partial_json、test_cpu_smoke_assertion_failure；G15 Minor（argparse exit 2 无用例、--selftest 未注明）→ 补 test_cli_usage_error_exit_2 + 注明 T004 覆盖；E-F-4 四要素措辞对齐「环境/命令/产物/判据」；tasks.md L7 元数据更新
- 第 1 轮修复一并验证：FR3 四项（bf16 两口径/强度标注/stdout Markdown 摘要/CPU 参照冒烟）、三方口径统一（design D7 = tasks T003 = srs FR3 异常处理，逐字一致）、D5 修正、is_causal 折叠、cudagraphs 层全部闭环
- 事实核查：抽查 16 项（数据 6 + 行号锚点 9 组 + 环境 1）100% 与仓库一致
- 遗留 Minor（不阻塞）：W1——包装层 dtype/device ValueError 两条契约的负向用例在 T003 TDD 实施时以参数化并入 invalid_shape 用例；W2——srs FR1-AC2 溯源列表含 ablation-summary.json 为 srs 侧笔误（−75% 实录在 EXPERIMENTS.md L52），AR 归档时回勘
- 审查代理：sdd-gate-reviewer

## Review 记录

### 2026-10-14 审查记录

- 审查结果：PASS（第 2 轮；第 1 轮 FAIL 1 Major，修复后复审 PASS）
- 审查方式：独立子 Agent 全量合规审查（S1-S5 / D1-D4 / C1-C3 / 手册数值抽查 M 13 项），实测复跑 28/28、全量 711/2、三绿、git diff 影响范围、行号锚点逐点抽查
- S 维度：S2-S5 全 PASS；S1 首轮 NO——E-F-4 命令③引用 profile_inference.py 不存在的 `--context_lengths`/`--new_tokens` 参数（`--compile` 为 store_true），SOP「命令」要素失效（Major）→ 修复为 `python -m tools.experiments.run_inference_suite --checkpoint <ckpt> --output_json docs/results/inductor_cloud.json --prompt_lengths 128 256 --new_tokens 64`（参数经 argparse 定义核验、`-m` 形式实机可解析、含 variant=compile/eager 对照行），复审 PASS
- D 维度：D1-D4 全 PASS（D1-D7 决策逐条核验闭环；D3 影响范围恰为声明 12 文件，model/training/utils 零改动）
- C 维度：C1-C3 全 PASS（三绿实测复现；E741 规避 grep 证实）
- M 数值抽查：13 项中 12 项一致（MoE 5.8×/KV 12,335→−1,090/114×/GQA-2 −75%/roofline 1160×·77.9%/compile 1.59×·294 vs 185·64tok 负结果/§6.9 全部数值/E-F-4 判据），1 项即上述 E-F-4 命令缺陷
- 遗留 Minor（不阻塞，记录待办）：①E-F-3 命令同源问题（profile_inference.py 无 `--context_lengths`，v0.6.0 遗留）——建议后续 lite AR 对全文 profile_inference.py 调用块做参数一致性扫描；②design §4.3.3 文字「mean/min 同时记入 JSON」与 §4.3.4 schema（仅 mean）不一致，实现从 schema——AR 归档时勘误 design 文字；③rmsnorm N 非 2 的幂档无专测（现有 CUDA 用例 N=64/1024 均为 2 的幂，掩码尾部路径未被实际触发）；④gqa_decode_bytes 将 q 计为 2× 元素量（docstring 自述，物理偏保守，k/v 项主导，强度值不受实质影响）
- 审查代理：合规审查子 Agent（2 轮）
