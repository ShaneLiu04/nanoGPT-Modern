# [AR-v0.8.5-moe-long-context] 任务清单

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.8.5-moe-long-context |
| 日期 | 2026-10-14 |
| 流程 | harnessX TDD：Red（子代理写测试）→ Green（主代理实现）→ 三绿 → commit |

## 任务列表

| ID | 任务 | 依赖 | 规模估计 | 状态 |
|----|------|------|----------|------|
| T001 | compile-stack.md 新增 §0 GPU 执行模型前置章（SM/warp/occupancy/层级/cp.async/TMA，roofline.json 算术闭合，sm75 无 TMA 明标） | 无 | S | Pending |
| T002 | RotaryEmbedding rope_scaling 扩展（none/ntk/pi/yarn，TDD：Red 测试先行→实现→config_models 校验→train CLI flag） | 无 | M | Pending |
| T003 | 手册 long-context.md（NTK 复盘/PI/YaRN 机制与本仓实现对照/滑窗+cache 驱逐/needle 方法论/MLA 理论节含内嵌 MQA/GQA/MLA 对比表/KV 复习） | T002 | M | Pending |
| T004 | run_ntk_eval.py 扩展四档对照（none/ntk/pi/yarn，--quick 低算力档，JSON 落 docs/results/）+ 本地低算力档实测 | T002 | M | Pending |
| T005 | run_needle_eval.py（包装 needle_in_haystack.py 的 depth×context 网格 CLI，--quick 档）+ 本地低算力档实测 | 无 | M | Pending |
| T006 | 手册 moe-systems.md（机制深化/容量负载微实验解读/专家并行理论节/E-A-5 未赢归因四因子分解/grouped GEMM 5.8× 复盘） | 无 | M | Pending |
| T007 | tools/infra/moe_probe.py（纯 CPU：capacity 扫描溢出率/router 负载分布/偏斜注入，TDD，JSON 原子写+退出码语义+守卫降级） | 无 | M | Pending |
| T008 | RUNNING_EXPERIMENTS 登记 E-A-8/E-A-9/E-C-6（逐参核对 argparse，E-F-4 教训）+ EXPERIMENTS 记录节（本地低算力档实测+云端回填占位） | T004, T005, T007 | S | Pending |
| T009 | 索引收口：ai-infra/README 主题地图挂深化手册+roadmap v0.8.5 状态、根 README 索引 | T003, T006, T008 | S | Pending |
| T010 | 质量收口：全量回归（基线 760/3 零回归）、三绿（black 100/ruff E,F,I,B,UP/mypy 零新增）、链接存在性检查、gitee 同步 | T001-T009 | S | Pending |

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

## 记录

- 2026-10-14 创建（需求阶段产出，Round 1/2 澄清裁定 + 暂停点自动化指令）
