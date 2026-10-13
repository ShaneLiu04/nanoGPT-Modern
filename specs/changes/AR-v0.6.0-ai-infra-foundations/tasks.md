# [AR-v0.6.0-ai-infra-foundations] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.6.0-ai-infra-foundations |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（2026-10-13 完成，方案 1B+2A+3B：逐张量激活建模/三精度口径/roofline 三段式） |
| 创建日期 | 2026-10-13 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | FR1 docs/ai-infra/README.md 学习路径索引（7 主题地图 + 4-AR 路线图 + 本仓锚点 + 延伸阅读；未交付主题标「计划中」） | - | passing | 2026-10-13 完成；7/7 主题条目含锚点+AR 标注；链接存在性终检并入 T007 |
| T002 | FR2 docs/ai-infra/memory-flops-arithmetic.md 显存/算力心算手册（五项分解推导 + FLOPs + roofline + 本仓数据逐条验证） | T001 | passing | 2026-10-13 完成；含 §3.3 单开关组间差经验分解（swiglu +1.52/GQA −0.62/qk-norm +1.89 公式闭合 ≈1.0GB 下界）与 §7 心算卡（7~9GB vs 实测 26.26GB 的诚实残差注记） |
| T003 | FR3 utils/infra_math.py 计算器（param_count/training_memory_breakdown/inference_kv_bytes/gqa_kv_materialization_slope/flops_per_token/step/arithmetic_intensity + CLI `python -m`） | - | passing | 2026-10-13 完成；TDD Red（子代理 31 用例）→ Green（2 轮修复：MoE 增量漏乘层数、CLI precision 改函数校验）；31 passed + black/ruff/mypy 三绿 + 覆盖率 96%（目标 ≥95%）；锚点全部精确断言通过 |
| T004 | FR4 tools/infra/run_roofline.py 本机微实验（三段式：GEMM 尺寸扫描跨 L2 4MB + GEMV 对照 + decode 步带宽反推；--quick ≤5min；CPU fallback 可测；结果 docs/results/roofline.json + EXPERIMENTS.md 记录） | T003 | passing | 2026-10-13 完成；TDD Red（子代理 8 用例，测试契约 bug 修复 gemm_flops(2,3,4)=48 非 24）→ Green（B023 partial 闭包 + abort 不写部分 JSON + cpu-smoke）；实测 --quick 10 点 <1min：GEMM 8.5→9825.7 GFLOPs/s（1160×，cache-bound 可见）、GEMV 349.2 GB/s=77.9% 规格、decode 4.81 GB/s=1.1%（bound=memory 方向成立，绝对利用率受启动开销主导已注记）；MIGRATION.md「8GB」→16 GiB ×2 处；ci.yml 三行加 tools/infra/；EXPERIMENTS §6.8+§8+§10 索引 |
| T005 | FR5 RUNNING_EXPERIMENTS.md 新增 F 组 SOP（E-F-1 通信带宽 / E-F-2 多卡 roofline+MFU / E-F-3 KV 物化 GPU 复测；环境/命令/产物/判据四要素 + 状态表登记） | T003 | passing | 2026-10-13 完成；F 组表 + 三实验各四要素段（E-F-1 内联 torchrun 脚本不新增仓库文件、总线带宽 2(N-1)/N 口径；E-F-2 复用 FR4 工具；E-F-3 复用 profile_inference，判据 ±5%）；状态表 3 行（E-F-2 part-done：本机 --quick 已跑）+ 更新历史登记 |
| T006 | FR6 README 文档索引挂载 ai-infra 系列（不破坏三条阅读线） | T001, T002 | passing | 2026-10-13 完成；「文档索引」表插入 ai-infra/ 与心算手册两行（MIGRATION 之后）；「实验与结果」段挂 F 组指引并把 14→17 个实验口径同步；badge 与质量保障行 628+→681+ ×3 处 |
| T007 | FR7 质量收口：全量回归零回归 + mypy/black/ruff 三绿（新文件入 CI 范围） + gitee 同步 + RUNNING_EXPERIMENTS 状态表更新 + T001 索引链接存在性终检 + README badge 测试数陈旧（628+ → 640+）顺带修正 | T001-T006 | passing | 2026-10-13 完成；全量 681 passed / 2 skipped 零回归（新基线口径 681 = 640 + infra_math 31 + roofline 8 + 会话增量 2）；CI 新范围本地三绿（82 文件）；链接终检 25 相对链接全存在（修复 1 处 `./adr/`→`../adr/`）；badge 修正至 681+（口径顺带更新）；gitee 同步见会话记录 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

### 2026-10-13 设计阶段完成

- design.md 生成（Step 1-4）：领域知识重建（无组件详设/无模板→内置默认；pyproject [project.scripts] 不采用、`python -m` 足够）→ 输入分析 → 三方案对比（1B 逐张量激活建模 / 2A 三精度口径 / 3B roofline 三段式）→ 全文落稿（时序图×1 + 流程图×2 + 接口 8 函数签名 + 测试设计 23 用例）
- 设计验证强化：param_count 公式与 model/modern_gpt.py 逐行比对，修正 2 处（qk_norm 实为 RMSNorm(head_dim) 跨头共享 → 2·hd/层 非逐头；baseline 实为 legacy 架构 → 补 use_bias/norm_variant 参数）；三锚点手工复算**精确闭合**（59,560,960 / 64,085,504 / 77,881,344，均与 EXPERIMENTS E-A 报告值一致）
- Step 5 审批（自动化授权自审通过）、Step 6 tasks.md 对齐、Step 7 组件详设跳过（specs/component-detail-design/ 不存在）

### 2026-10-13 开发会话记录（T001-T003）

- 完成任务：T001 学习路径索引（7 主题地图+4-AR 路线图+锚点+延伸阅读）；T002 心算手册（8 节：单位底账/参数量精确账/三精度口径/激活逐张量表+经验分解/KV+物化案例/FLOPs/roofline/心算卡）；T003 计算器（TDD Red 子代理 31 用例 → Green，2 轮定向修复）
- 修改文件：docs/ai-infra/README.md、docs/ai-infra/memory-flops-arithmetic.md（新增）；utils/infra_math.py、tests/test_infra_math.py（新增）
- 测试覆盖率：utils/infra_math.py 语句 96%（目标 ≥95%）；31 passed
- 备注：锚点全部精确闭合（59,560,960 / 64,085,504 / 77,881,344 / +1,536 / 12,288 / +1.25GB）；Green 阶段 2 个缺陷（MoE 增量漏乘 n_layer、argparse choices 吞掉 ValueError 契约）均在 3 次尝试内修复

### 2026-10-13 开发会话记录（T004）

- TDD 纪律执行：Red（子代理 8 用例：sizes 子集/公式/schema/args/空入参/cpu-smoke/abort）→ Green（tools/infra/__init__.py + run_roofline.py 三段式实现）；测试契约 bug 1 处（gemm_flops(2,3,4) 应为 48，子代理 prompt 算术错）
- 实现缺陷修复：B023（计时 lambda 改 functools.partial）、E501、无效 noqa；运行入口 sys.path 修复（以文件方式运行时仓库根入 path，profile_inference.py 先例）；「PowerShell Set-Content 二次编码损坏 UTF-8 中文」事故 1 起——整体重写恢复（逻辑未动）
- 本机实测（Quadro RTX 5000，--quick 10 尺寸点 <1min，满足 FR4 验收 1）：段1 GEMM 8.5→9825.7 GFLOPs/s（1160× 提升，L2 边界前后 3 个数量级，cache-bound 区间可见）；段2 GEMV 349.2 GB/s = 规格 448 GB/s 的 77.9%；段3 decode tiny 7,286,016 参数单步 6.05 ms → 有效带宽 4.81 GB/s（1.1%，bound=memory 方向成立，绝对利用率受 kernel 启动开销主导，与手册 §7 残差结论互证）
- 顺带交付：MIGRATION.md「8GB」→ 16 GiB 修正 ×2 处（门控 W5 兑现）；ci.yml mypy/black/ruff 三行加 tools/infra/（lint 契约扩展，T007 验收项）；EXPERIMENTS.md §6.8 记录节 + §8 复现命令 + §10 索引行
- 质量门：39 passed（8 roofline + 31 infra_math）+ 三绿（black/ruff/mypy）
- 修改文件：tools/infra/__init__.py、tools/infra/run_roofline.py、tests/test_roofline_script.py、docs/results/roofline.json、docs/EXPERIMENTS.md、docs/MIGRATION.md、.github/workflows/ci.yml、tasks.md

### 2026-10-13 开发会话记录（T005-T007，AR 收口）

- T005：RUNNING_EXPERIMENTS.md 新增 F 组（E-F-1 通信带宽 / E-F-2 多卡 roofline+MFU / E-F-3 KV 物化复测）：矩阵表 + 三实验各四要素段；E-F-1 内联 torchrun 脚本（总线带宽 2(N-1)/N 口径）不新增仓库文件；E-F-2/E-F-3 复用本 AR 交付工具（run_roofline.py / profile_inference.py）；状态表 3 行（E-F-2 part-done）+ 更新历史登记
- T006：README 挂载——「文档索引」表 +2 行（ai-infra/ 索引、心算手册）；「实验与结果」段 14→17 实验口径 + F 组指引；三条阅读线未动
- T007：全量回归 681 passed / 2 skipped 零回归（新基线 681，原口径 640 + 本 AR 新增测试）；CI 新范围（含 tools/infra/）本地三绿 82 文件；T001 索引链接存在性终检 25/25（修复 ai-infra/README.md 第 4 步 `./adr/`→`../adr/` 1 处）；README badge 与质量保障行 628+→681+ ×3；RUNNING_EXPERIMENTS 状态表已在 T005 内更新
- 事故与修复：① PowerShell 管道向 git commit 传中文变 `?`（$OutputEncoding 陷阱）——amend 改用 `-F <UTF-8 文件>` 修复，此后提交一律走文件；② edit 替换标题行吞标题 ×2（EXPERIMENTS §7、tasks 门控记录）——均在当步发现补回
- 修改文件：docs/RUNNING_EXPERIMENTS.md、README.md、docs/ai-infra/README.md（链接修复）、tasks.md

### 2026-10-13 审查记录（第 1 轮，FAIL → 修复）

- 审查结果：FAIL（S1/S4/D1/D2/D4 NO；S2/S3/S5/D3/C1-C3 YES）；审查代理独立复跑三绿/39 用例/全量 681+2perf/CLI/锚点 25 项存在性/roofline.json 数值比对
- Major（S1/D2/D4）：激活逐张量模型缺 attn scores [B,H,T,T] 项（srs FR2 期望行为 1 + design §4.1 表均有）——修复：`_activation_bytes_per_token_per_layer` 与 `training_memory_breakdown` 增加 `materialize_scores` 开关（默认 False=fused SDPA 不物化口径；True=math 后端 2·n_head·T B/token/层），design §4.1 scores 行改两口径、手册 §3.1 表补行，新增 ut-mem-scores 用例（增量精确断言）
- Minor（S4/D1）：design §4.3.2「非法 device → ValueError」未实现——修复：三段函数加 `_validate_device` 白名单校验（先于 torch.device 的 RuntimeError），新增 ut-invalid-device 用例
- Minor（D1/D4）：`summarize` 签名细化未记录——修复：docstring + design §4.3.2 同步记录（mode 为 schema 必需键、三段分参便于单测）
- Minor（D2/D4）：GELU 口径实现 8d（两张量）vs design 初稿 I=4d——修复：design §4.1 行更正为 8d 并注明「审查修正，与手册 §3.1 一致」
- Minor（S1 文档质量）：手册 §3.3 交叉引用 §5.2 → §4.2（KV 物化机理实际位置）已更正
- 顺带发现（审查外）：tests/test_infra_math.py 存量 I001（import 分组缺空行；tests/ 不在 CI lint 范围故 T003 未暴露）——已 ruff --fix + black 修复
- 修复后：41 passed（39+2 新用例）+ 三绿 + 全量 683 passed / 2 skipped 零回归

### 2026-10-13 审查记录（第 2 轮，PASS）

- 复审结果：PASS（S1/S4/D1/D2/D4 全 YES，证据闭合）
- 审查代理独立复算：scores 增量公式（2·H·T 元素/token/层 × L × tokens × dtype）与 ut-mem-scores 差值断言精确吻合；_validate_device 三段一致且先于张量构建；summarize 签名双记录（design §4.3.2 + docstring）；GELU 8d 三处（design/实现/手册）一致；手册 §3.3 交叉引用指向 §4.2 正确
- 可运行验证复跑：41 passed / ruff+black+mypy 三绿；未发现修复引入的新问题
- 至此 AR-v0.6.0 开发 + 审查全部完成，进入 ST 验收（sdd-task-st）

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-13 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（G1/G2/G4/G5/G6/G8 = YES；G3/G7/G9 = Minor WARN）
- 修复的问题：W1 显存锚点来源错误（15.49GB 为本机 WDDM 溢出非云端 → 改引云端 E-A 23.1/26.3/29.3GB 并拆解口径，15.49GB 改作反面案例）；W2 KV 物化口径矛盾（12,288·L → 斜率 12,335 B/token 层数无关，单层瞬时 12,288=3×16×64×4，断言口径已统一）；W3 基线测试数统一为 2026-10-13 实测 640 passed；W4 T001/T002 链接次序矛盾（首版仅锚既有文件、终检并入 T007）；W5 GPU 规格修正（nvidia-smi 16384 MiB → 桌面版 16GB/448GB/s，顺带修 MIGRATION「8GB」误载）——均已在 srs.md/tasks.md 修正
- 审查代理：sdd-gate-reviewer

### 2026-10-13 design 门控记录（第 1 轮）

- 门控结果：FAIL
- 失败项：G14（Important NO）——srs FR3 验收 2 optimizer 计账口径错误（写成 bf16 权重流派「optimizer 12 B/param、参数/梯度 bf16 2+2」，与本仓 trainer_base.py autocast 实际（参数即 fp32 主权重，4+4+8=16）矛盾；design 决策 2A 口径正确，属 srs 缺陷）；另 G2/G3/G7/G12/G13/G15 六项 Minor WARN
- 修复：srs FR3 验收 2 改为 autocast 口径（optimizer 项 = 8 B/param、总静态 16 B/param）+ bf16-pure 12 B/param 并列双口径；FR2 期望行为 1 同步澄清；design.md 补 Minor——G2（MFU/绘图取舍入 §4.5 决策 D4/D5）、G3（gqa 异常声明 + flops_per_step 完整签名 + roofline 函数级异常）、G7（§4.5 决策记录 6 条内联）、G12（FR1 索引完整性 / FR6 README 挂载入 6.3）、G13（ut-mem-fp32-16b + bound 判定行 + ut-roof-abort）、G15（ut-flops-step / ut-flops-lmhead / ut-gqa-invalid / ut-roof-args）
- 审查代理：sdd-gate-reviewer

### 2026-10-13 design 门控记录（第 2 轮，复审）

- 门控结果：PASS
- 审查项数：15 项（G1-G15：13 YES + G2/G15 Minor WARN；全维度重审 + 审查代理独立复算锚点全部闭合 59,560,960/64,085,504/77,881,344/1,536/12,288）
- 第 1 轮失败项修复验证：G14 已修复（srs FR3 验收 2 autocast 口径 + design/ut-mem-amp-16b/trainer_base.py 三方一致）；G2/G3/G7/G12/G13/G15 Minor 全部修复
- 遗留 Minor（第 2 轮后顺手闭合）：srs §2「理论 MFU」措辞对齐 D4；ut-roof-args 并入空入参校验断言；tasks.md 用例数同步（23+7=30）
- 审查代理：sdd-gate-reviewer
