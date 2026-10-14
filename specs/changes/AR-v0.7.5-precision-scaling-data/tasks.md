# [AR-v0.7.5-precision-scaling-data] 任务跟踪

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.7.5-precision-scaling-data |
| 关联 srs.md | ./srs.md |
| 关联 design.md | ./design.md（2026-10-14 生成，design 门控待执行）|
| 创建日期 | 2026-10-14 |

## 任务列表

| ID | 任务描述 | 依赖 | 状态 | 备注 |
|----|---------|------|------|------|
| T001 | FR1 手册《numerical-precision.md》：四精度表示学（位布局/ulp/RNE）、AMP/GradScaler 逐行对照（build_amp_context L208）、累加精度、E-C 2.10× 复盘、与 compile/triton 口径交叉、fp8 理论展望（验证归 v0.9.0 注记） | - | passing | 2026-10-14 完成（commit 7b91999，262 行）；每节「本仓验证」小节；锚点：trainer_base.py:49/208、system.json、memory-flops §2.1、triton bf16 1e-2 |
| T002 | FR3 手册《scaling-laws.md》：Kaplan vs Chinchilla 全脉络、compute-optimal、B_crit、cosine/WSD、本仓 budget/7 组消融拟合复盘（诚实标注 7 点非受控） | - | passing | 2026-10-14 完成（commit 3b11576，171 行）；锚点：budget.json（记账三件套 corpus 1.24B/split 1.18B/consumed 0.59B）、ablation-summary.json、EXPERIMENTS §6.1/6.2、LRScheduler（utils/lr_scheduler.py L14/L81） |
| T003 | FR5 手册《data-pipeline.md》：获取/流式→tokenize→去重→过滤→packing→shard/resume 全链路逐环节代码锚点 + MIGRATION §3 判据表复盘 + 质量边界 | - | passing | 2026-10-14 完成（commit fc88e4a，202 行）；锚点：openwebtext.py L110/159/326/485、prepare.py L307/330/415/441、dedup.py L31/69/117、filter.py、test_packing |
| T004 | FR2 precision_probe.py + tests/test_precision_probe.py（TDD：Red 先行）：表示剖析（范围/ulp/RNE 统计）、四精度误差注入分布、bf16 累加 vs fp32 累加对照、fp16 下溢 + loss scaling 模拟 → docs/results/precision_probe.json（原子写、退出码语义化、纯 CPU） | T001 | passing | 2026-10-14 完成（commit 2a37573）；Red 子代理 14 用例+Green 主代理；13 passed/1 skipped（Windows skipif 合规）；三绿；tolerance_refs 从 triton_kernels.json consistency 读取 |
| T005 | FR4 fit_scaling.py + tests/test_fit_scaling.py（TDD）：真实 7 点 Kaplan 式 L(N)=E+A/N^α 多初值拟合（D 恒定→B/D^β 不可辨识，联合式在合成数据演示）、外推预测、Chinchilla 预算计算器（文献参数默认+本仓拟合对照，6ND 口径对照 infra_math） → docs/results/scaling_fit.json（fit_quality 诚实报告；数据源缺失报错退出 1、不收敛诚实降级） | T002 | passing | 2026-10-14 完成（commit 96c92ed）；design D3：7 组同预算 D 恒定是拟合策略核心依据；多初值策略经原型调参（α 恢复 0.7% 误差）；17 passed；三绿；数据源守卫 ablation-summary.json/budget.json |
| T006 | FR6 data_probe.py + tests/test_data_probe.py（TDD）：train.bin memmap 抽样（top-K 直方图/EOT 频率/doc 长度/dtype 核验/污染检测演示）+ 合成 corpus（MinHash 去重率曲线/packing 填充率/doc_boundary 统计） → docs/results/data_probe.json（train.bin 缺失守卫降级、测试零大文件依赖） | T003 | passing | 2026-10-14 完成（commit 944f9b6）；EOT 频率对照 MIGRATION §3（1/1128≈8.87e-4 量级）；memmap 零拷贝；dedup 曲线调用生产 MinHashDeduplicator（keep/drop mask 语义核实）；16 passed；三绿 |
| T007 | FR2/4/6 验收：三工具本机全量实测（各 ≤2 分钟）落 JSON；EXPERIMENTS.md 新增记录节（三工具表格+能说/不能说）；scaling 拟合残差与 data EOT 频率对照判据核验 | T004, T005, T006 | pending | 诚实报告：拟合优度不达标如实记录 |
| T008 | FR7 勘误三件：①主题 6 锚点漂移（ai-infra/README.md:23 → modern_gpt.py/run_inference_suite.py 实际位置 + RUNNING_EXPERIMENTS B 组命令改为实际可执行）；②ADR-006 superseded 注记；③TECH_REPORT 容量规划口径交叉注；quantize_model 待办登记（不修，归 v0.9.0） | T007 | pending | B 组命令勘误须与 run_inference_suite.py argparse 核对（吸取 E-F-4 教训） |
| T009 | FR8a 索引更新：ai-infra/README.md 主题地图 7→10（主题 8/9/10 转已交付+挂手册链接）、路线图补 v0.8.5 行（MoE+长上下文）、阅读顺序衔接；链接存在性检查 100% | T008 | pending | 全文引用文件 Test-Path 核验 |
| T010 | FR8b README.md 文档索引挂三手册 + badge 测试数更新；纯增量不破坏结构 | T009 | pending | badge 现已为 711+（v0.7.0 T006 更新）；本 AR 以 T011 全量实测为准回填（req 门控 W1 修正） |
| T011 | FR8c 质量收口：全量回归零回归（基线 711 passed / 2 skipped）+ 三绿（black/ruff/mypy 零新增）+ gitee slim-mirror 同步 + 链接终检（含 B 组勘误命令目标核验） | T010 | pending | mypy 遗留 19 处为 db3b7ff 前旧账不新增即可 |

## 状态说明

- `pending`：待开始
- `in_progress`：进行中（当前会话）
- `passing`：开发完成，测试通过
- `failed`：测试失败，需修复

## 进度记录

> 每个开发会话结束后追加，记录完成情况。

## 阶段门控记录

> 由 sdd-phase-gate skill 在阶段门控审查后追加，记录每轮审查结果（PASS/FAIL + 轮次）。格式见 sdd-phase-gate SKILL.md Step 6。

### 2026-10-14 req 门控记录

- 门控结果：PASS（第 1 轮）
- 审查项数：9 项全 YES；事实主张核查 16+ 处全一致（含三处文档漂移主张反向验证成立：README:23 主题 6 锚点 / B 组命令 --draft_iters 非 --draft_model / ADR-006 无 superseded 注记；train.bin 本机 17,853,806,718 B 在位）
- Minor WARN：W1——tasks T010 badge 备注前提陈旧（681→711），已当场修正为「以 T011 实测回填」；W2——FR4「R²>0.9 或诚实记录不达」逃生条款的落盘格式（fit_quality 字段）转 design 阶段明确；W3——711/2 基线与 mypy 19 处遗留为引用性主张，T011 实测自愈
- 审查代理：sdd-gate-reviewer

### 2026-10-14 design 门控记录

- 门控结果：FAIL（第 1 轮）
- 失败项：G12（功能点 7/8 勘误与收口无测试用例，High）、G14（8 条 srs 验收标准无追溯用例：FR1-2/FR7-1/2/3/FR8-1/2/3，High）、G15（--output-json 校验语义未定义+无用例，Medium）、G13（实验异常分支无具体用例+两工具非法入参缺口，Medium）
- 严重性：High×2 / Medium×2（另 Low×4：G1/G2/G4/G8——边界矩阵/不确定度与时长字段/覆盖率目标/D3 偏差回写）
- 审查代理：sdd-gate-reviewer
- 修复（同日）：design.md §4.3.1 补全三工具 --output-json 统一校验语义；§4.3.2 补 param_uncertainty 与训练时长字段；§6.1 补共有用例组（异常注入/--output-json 触发/非法入参/冒烟 ×3）+ fit_scaling 单独缺失用例 + 覆盖率目标声明（≥90%）；新增 §6.4 边界条件矩阵/§6.6 勘误与收口 ST 用例（ST-C1~C8）/§6.7 srs 验收标准追溯表（24 条全覆盖）；追溯附录补 D3 显式偏差记录；srs FR4 期望行为同步精化（附 Round 4 澄清记录）

### 2026-10-14 design 门控记录（第 2 轮）

- 门控结果：PASS（第 2 轮，全维度重审）
- 审查项数：15 项（G1-G9/G11-G15 全 YES；G10 组件详设目录不存在按规则跳过）；第 1 轮失败项 G12/G14（High）G13/G15（Medium）核实修复到位；事实核查 22 处引用文件/数据源全部在位
- Minor WARN（已当场修复）：W1——--output-json「不可写」触发补只读目录用例（Windows ACL 不支持时 skipif）+ 校验分支代覆盖注记；W2——FR4-1 追溯措辞对齐（合成参数恢复+fit_quality 字段存在性断言，真实 smoke 补 r_squared 落盘与 limitations 非空断言）；W3——§6.6 补 ST-C9 三绿零新增（black/ruff/mypy 与 19 处旧账基线比对）
- 审查代理：sdd-gate-reviewer
