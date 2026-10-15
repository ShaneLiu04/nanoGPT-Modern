# [AR-v0.8.5-moe-long-context] 任务清单

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.8.5-moe-long-context |
| 日期 | 2026-10-14 |
| 流程 | harnessX TDD：Red（子代理写测试）→ Green（主代理实现）→ 三绿 → commit |

## 任务列表

| ID | 任务 | 依赖 | 规模估计 | 状态 |
|----|------|------|----------|------|
| T001 | compile-stack.md 新增 §0 GPU 执行模型前置章（SM/warp/occupancy/层级/cp.async/TMA，roofline.json 算术闭合，sm75 无 TMA 明标） | 无 | S | Done |
| T002 | RotaryEmbedding rope_scaling 扩展（none/ntk/pi/yarn，TDD：Red 测试先行→实现→config_models 校验→train CLI flag） | 无 | M | Done |
| T003 | 手册 long-context.md（NTK 复盘/PI/YaRN 机制与本仓实现对照/滑窗+cache 驱逐/needle 方法论/MLA 理论节含内嵌 MQA/GQA/MLA 对比表/KV 复习） | T002 | M | Done |
| T004 | 新建 run_context_eval.py 四档对照入口（naive/ntk/pi/yarn，--quick 低算力档，JSON 落 docs/results/）+ rope_retune.py 共享模块（D4：run_ntk_eval.py 不改，E-A-7 命令字节稳定）+ 本地低算力档实测 | T002 | M | Done |
| T005 | run_needle_eval.py（包装 needle_in_haystack.py 的 depth×context 网格 CLI，--quick 档）+ 本地低算力档实测 | 无 | M | Done |
| T006 | 手册 moe-systems.md（机制深化/容量负载微实验解读/专家并行理论节/E-A-5 未赢归因四因子分解/grouped GEMM 5.8× 复盘） | 无 | M | Done |
| T007 | tools/infra/moe_probe.py（纯 CPU：capacity 扫描溢出率/router 负载分布/偏斜注入，TDD，JSON 原子写+退出码语义+守卫降级） | 无 | M | Done |
| T008 | RUNNING_EXPERIMENTS 登记 E-A-8/E-A-9/E-C-6（逐参核对 argparse，E-F-4 教训）+ EXPERIMENTS 记录节（本地低算力档实测+云端回填占位） | T004, T005, T007 | S | Done |
| T009 | 索引收口：ai-infra/README 主题地图挂深化手册+roadmap v0.8.5 状态、根 README 索引 | T003, T006, T008 | S | Done |
| T010 | 质量收口：全量回归（基线 760/3 零回归）、三绿（black 100/ruff E,F,I,B,UP/mypy 零新增）、链接存在性检查、gitee 同步 | T001-T009 | S | Done |

## 依赖图

```
T001 ──────────────────────────────────────────────┐
T002 ──→ T003 ──────────────────────────────────┐   │
T002 ──→ T004 ──┐                                │   │
T005 ──────────┤                                │   │
T006 ──────────┼──→ T008 ──→ T009 ──→ T010 ←───┴───┘
T007 ──────────┘
```

## 状态说明

- Pending：未开始
- In Progress：进行中
- Done：三绿通过并 commit
- Blocked：受阻塞（注明原因）

## 阶段门控记录

### 2026-10-15 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项（G1-G9 全 YES）
- 修复的问题：无（2 项 Minor WARN 留待 design 阶段：FR6「同数量级」需补容差定义；FR2 rope_scaling config 形态（现有 dict `{"type":"ntk"}` vs srs 字符串枚举）需 design 明确向后兼容方案）
- 审查代理：sdd-gate-reviewer

### 2026-10-15 design 门控记录

- 门控结果：FAIL（第 1 轮）
- 失败项：G12（Important，F6 缺 context_eval e2e 用例）、G14（Important，7 条 srs 验收无溯源用例：FR5-AC1/FR1-AC1/FR1-AC3/FR3-AC1/FR3-AC3/FR6-AC2/FR7-AC1）、G15（Important，I4 factor<1 退出 2/I5 非法 method/I1 train_len 越界三个声明错误无触发用例）；WARN：G9（T004 文字与 D4 矛盾）、G13（pi 分支无分发用例）、G15（rope_type 无用例）
- 审查代理：sdd-gate-reviewer
- 修复：§6.2 新增 IT11 e2e；§6.3 新增 BS7-BS12 溯源核查项；§6.1 新增 UT10(CLI)/UT12/UT13/UT14/UT15 并修正 UT9 交叉引用；T004 文字同步 D4

### 2026-10-15 design 门控记录（第 2 轮）

- 门控结果：PASS（第 2 轮；第 1 轮 3 项 Important 已全部修复核验通过）
- 审查项数：15 项（G1-G8/G10-G13 全 YES；G9/G14/G15 仅 Minor WARN）
- 修复的问题：第 1 轮 G12（IT11 e2e）/G14（BS7-BS12 溯源+FR-AC 映射）/G15（UT10/UT12/UT13/UT14/UT15）；第 2 轮 3 项 Minor 已顺手修复（§4.4 补 roadmap-gap-analysis.md、映射补 FR4-AC4→IT5/T010、UT10 补 none 缺省分支断言、新增 IT12 needle 非法入参 exit 2）
- 审查代理：sdd-gate-reviewer

### 2026-10-15 审查记录（sdd-task-review）

- 审查结果：PASS（第 1 轮；独立审查子代理，无 Critical/Important，4 项 Minor 全部当场修复）
- S 维度：S1/S2/S3/S5 全 YES（FR1-FR8 验收均可溯源到测试/交付物；缺省行为 bit-exact；run_ntk_eval/needle_in_haystack/MoE 前向零改动经 diff 核实）
- D 维度：D2/D3 YES；D1/D4 Minor——design I8 文本（--config_json、metrics schema）与实现（--output-json、契约 schema）偏离，已补 I8 勘误注记（实现以 tests/test_moe_probe.py TDD 契约为准）；顺手修复 §4.2.2 YaRN ramp 方向括注写反（与「注意」句自相矛盾，UT5 锁定正确方向）
- C 维度：C1/C2/C3 全 YES（旧签名不变/exit 语义区分清晰/三绿实测复核通过）
- S4 Minor×2 已修复：UT10 组装断言（train_pretrain 提取纯函数 build_rope_scaling+新测试）、IT10 retune 联动（新 spy 测试断言 retune_rope 真实调用且 inv_freq 改变，不再只看 JSON echo）
- 修复后验证：三绿（black/ruff/mypy）+ 相邻回归 105 passed（needle/rope_pi_yarn/ablation_switches/common_args/hydra_config）
- 审查代理：general（独立上下文，只读）

## 记录

- 2026-10-14 创建（需求阶段产出，Round 1/2 澄清裁定 + 暂停点自动化指令）
- 2026-10-15 design.md 生成后对齐确认：F1→T001、F2/F3/F4→T002、F5/F6→T004、F7→T005、F8→T007、F9→T003、F10→T006、F11→T008、F12→T009/T010；任务分解与设计一致无需调整。两项 req WARN 已在 design 落定：D1（rope_scaling 保留 dict、type 扩展枚举，WARN-2）、D5（roofline 闭合容差比值 ∈ [0.5,2]，WARN-1）。组件详设不存在，Step 7 跳过。
