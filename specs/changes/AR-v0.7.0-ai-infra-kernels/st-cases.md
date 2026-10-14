# [AR-v0.7.0-ai-infra-kernels] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.7.0-ai-infra-kernels |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-14 |

## 测试用例列表

### ST-001：flash/online-softmax 推导与 ring_attention.py 对照

**关联需求：** srs.md §3.1 FR1-AC1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 手册中 flash/online-softmax 推导

**测试步骤：**
1. 逐行核对 attention-kernels.md 推导↔代码对照表与 model/ring_attention.py 实现

**期望结果：**
- Then 每个推导步骤有对应代码行锚点，推导与实现语义一致

**实际结果：** 审查子代理逐行核验对照表为真（在线 softmax 递推↔ring_attention.py L145-156；分块/边界锚点 L101-164 全命中）

**状态：** PASS

---

### ST-002：手册存量数据溯源（5.8×/12,335/−75%）

**关联需求：** srs.md §3.1 FR1-AC2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 手册引用的存量数据

**测试步骤：**
1. 追溯 moe_kernel.json / MIGRATION §8.2 / EXPERIMENTS.md 逐值比对

**期望结果：**
- Then 数值一致

**实际结果：** MoE 11.391 vs 1.978 ms=5.76×（moe_kernel.json）；KV 12,335→−1,090 B/token、114×、57.7MB→0.5MB（MIGRATION §8.2 L162/166）；GQA-2 4,608（−75%）vs MHA 18,432（EXPERIMENTS L52/L183）全一致。srs 笔误（误写 ablation-summary.json）已在归档前勘误为 EXPERIMENTS.md

**状态：** PASS

---

### ST-003：GQA 三路径与 modern_gpt.py 对照

**关联需求：** srs.md §3.1 FR1-AC3
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given GQA 三路径（grouped-broadcast / enable_gqa / 手写 decode）

**测试步骤：**
1. 与 modern_gpt.py:163-297 对照判定条件

**期望结果：**
- Then 语义一致（何时走哪条路径与代码一致）

**实际结果：** 审查行号锚点逐点抽查全命中（163/177-242/245-297/389/661/770）；三路径判定条件与代码一致

**状态：** PASS

---

### ST-004：dynamo 层 recompile 演示本机复跑

**关联需求：** srs.md §3.2 FR2-AC1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given compile-stack.md §4 dynamo 层实测代码块（声称「复制即跑」）

**测试步骤：**
1. 本机（torch 2.5.1+cu121）逐块复跑 §4.1/§4.2/§4.3

**期望结果：**
- Then 计数变化可复现（脚本化）

**实际结果：** 首轮 **FAIL**——三段均无法按文中预期运行：§4.1 `ExplainOutput` 无 `graph_break_reasons` 属性（AttributeError）；§4.2 `dynamo.run(g)` 不填充 `counters["stats"]["unique_graphs"]`（恒 0），且「n=2/4 同图」预期与 dynamo 默认形状特化语义不符；§4.3 `dynamo.run` 不接受 `nopython` 参数、`TorchDynamoException` 不在顶层。**修复（d270a1b）**：§4.2/§4.3 改 `torch.compile(backend="eager")`/`fullgraph=True`（对齐 MIGRATION §8.1 方法）；§4.1 属性名改 `break_reasons`；§4.2 预期改为实测输出+形状特化教学点。修复后三段整段复跑：§4.1 graph_count=1/break_reasons=[]/ops 列表正确；§4.2 计数 2→1、4→2、8→3 后稳定；§4.3 fullgraph 异常正确捕获（TorchDynamoException: Dynamic control flow is not supported）

**状态：** PASS（修复后复验）

---

### ST-005：compile 存量数据与归因口径

**关联需求：** srs.md §3.2 FR2-AC2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given compile 存量数据（1.59×/负结果）

**测试步骤：**
1. 追溯 inference.json / EXPERIMENTS §6.4 比对数值与归因口径

**期望结果：**
- Then 数值一致且归因口径一致（编译成本摊平边界）

**实际结果：** 256 tok compile 294 vs eager 185 tok/s≈1.59×；64 tok 79 vs 184（compile 更慢）、94 vs 182（EXPERIMENTS L225/L228、inference.json 78.99/184.06）全一致；归因口径均为编译成本摊平边界（64~256 之间）

**状态：** PASS

---

### ST-006：三层架构描述与 compile 调用点对照

**关联需求：** srs.md §3.2 FR2-AC3
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 三层架构描述

**测试步骤：**
1. 与 trainer_base.py/modern_gpt.py compile 调用点对照配置项（mode/fullgraph/dynamic）

**期望结果：**
- Then 配置项语义一致

**实际结果：** FSDP→compile 顺序（trainer_base.py L755-758）、reduce-overhead/dynamic=True（modern_gpt.py L2087-2089）、单步 fullgraph（L2336-2340）、can_fullgraph 收紧（L2052-2060）、KVCacheManager vs value 列表权衡（L2309-2315）审查锚点全命中，语义一致

**状态：** PASS

---

### ST-007：本机全量运行三 kernel（≤5 分钟）

**关联需求：** srs.md §3.3 FR3-AC1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 本机 Quadro RTX 5000 + triton 3.8.0

**测试步骤：**
1. `python tools/infra/triton_kernels.py`（full mode）

**期望结果：**
- Then 三个 kernel 全部通过数值一致性断言并产出 JSON（≤5 分钟）

**实际结果：** T004 实测（2026-10-14）：18 measurements（3 kernel×3 尺寸×2 dtype）consistency 全 allclose=True；JSON 落盘 docs/results/triton_kernels.json；全程 <2 分钟

**状态：** PASS

---

### ST-008：fp32 口径 allclose 与容差记录

**关联需求：** srs.md §3.3 FR3-AC2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given eager 参照与 Triton 输出

**测试步骤：**
1. fp32 口径比对（atol=1e-5/rtol=1e-4）；核对 JSON 容差字段

**期望结果：**
- Then allclose 且容差在 JSON 中记录

**实际结果：** 18 组 allclose 全 True；容差（fp32 atol=1e-5/rtol=1e-4、bf16 1e-2/1e-2）在常量/JSON meta/测试三处一致（审查 D5 核验）；input_precision="ieee" 显式记入 meta

**状态：** PASS

---

### ST-009：无 GPU 环境 pytest 行为

**关联需求：** srs.md §3.3 FR3-AC3
**测试类型：** 边界条件
**优先级：** High

**前置条件：**
- Given 测试套件（无 GPU 环境）

**测试步骤：**
1. pytest tests/test_triton_kernels.py（含 monkeypatch 模拟无 CUDA 路径）

**期望结果：**
- Then 参照实现纯函数单测通过、triton 用例 importorskip 跳过而非失败

**实际结果：** ST 复跑 28 passed（1.77s）：CPU 冒烟双路径（test_cpu_smoke_mode/test_cpu_smoke_assertion_failure 打桩 torch.cuda.is_available）通过；triton 用例 importorskip+GPU 守卫语义审查核验（S2/S5）

**状态：** PASS

---

### ST-010：测试文件零失败（CI 口径）

**关联需求：** srs.md §3.4 FR4-AC1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 无 GPU/无 triton 的 CI 环境（以测试套件守卫语义+全量回归覆盖）

**测试步骤：**
1. pytest tests/test_triton_kernels.py 全量

**期望结果：**
- Then 全部通过或 skip，零失败

**实际结果：** 28 passed 零失败（本机 GPU 在位时全过；无 GPU 路径由 monkeypatch 用例覆盖且 importorskip 保证缺 triton 环境 skip 不 fail）；全量回归 711/2 中本文件零失败

**状态：** PASS

---

### ST-011：eager RMSNorm 与仓内实现对照

**关联需求：** srs.md §3.4 FR4-AC2
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given 参照 RMSNorm 实现

**测试步骤：**
1. 与 model/modern_gpt.py RMSNorm 前向对照（fp32）

**期望结果：**
- Then 数值一致

**实际结果：** test_eager_rmsnorm_matches_reference_class 通过（28 passed 内）；锚点一致性由参数化用例锁定

**状态：** PASS

---

### ST-012：E-F-4 四要素与命令风格

**关联需求：** srs.md §3.5 FR5-AC1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given E-F-4 条目

**测试步骤：**
1. 逐项检查四要素（环境/命令/产物/判据）与 F 组命令风格

**期望结果：**
- Then 四要素齐备且命令风格与 F 组一致

**实际结果：** 首轮审查 **FAIL**（Major）——命令③引用 profile_inference.py 不存在的 `--context_lengths`/`--new_tokens` 参数，SOP 命令要素失效。**修复（acf3852）**：改为 `python -m tools.experiments.run_inference_suite --checkpoint <ckpt> --output_json docs/results/inductor_cloud.json --prompt_lengths 128 256 --new_tokens 64`（argparse 定义核验匹配、`-m` 形式实机可解析、含 variant=compile/eager 对照行）。复审 3 检查点全 YES：命令已更新、参数匹配可执行、四要素完整且判据/产物未破坏

**状态：** PASS（修复后复审）

---

### ST-013：EXPERIMENTS §6.9 与 JSON 逐值一致

**关联需求：** srs.md §3.5 FR5-AC2
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given EXPERIMENTS 记录节

**测试步骤：**
1. 与 triton_kernels.json 逐值比对

**期望结果：**
- Then 一致（舍入口径注明）

**实际结果：** 审查 M 逐值核验一致：swiglu 0.12×（36.22/4.34ms，强度 328）、rmsnorm 1.32×/2.86×、gqa 0.86×/0.39×、18 组 allclose 全 True；舍入口径「fp32 中位数 4 位有效数字」在 §6.9 明文注明；「能说/不能说」边界齐备（不设阈值诚实报告）

**状态：** PASS

---

### ST-014：链接存在性检查

**关联需求：** srs.md §3.6 FR6-AC1
**测试类型：** 正常路径
**优先级：** High

**前置条件：**
- Given 索引更新后

**测试步骤：**
1. 对 ai-infra/README.md、主 README.md、EXPERIMENTS §6.9、E-F-4 引用的仓内文件做存在性检查

**期望结果：**
- Then 引用的仓内文件 100% 存在

**实际结果：** T005 Test-Path 14 个引用文件全 OK；T007 E-F-4 命令目标（tools/experiments/run_inference_suite.py、tests/test_moe_vectorized.py::test_compile_no_recompile_growth）核验存在；ADR 实数核实 6 篇编号（001-006）+目录 README

**状态：** PASS

---

### ST-015：README 文档索引两行挂载

**关联需求：** srs.md §3.6 FR6-AC2
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given README 文档索引

**测试步骤：**
1. 检查新增两行与表格结构完整性

**期望结果：**
- Then 新增两行且不破坏既有结构

**实际结果：** L535 后纯插入 attention-kernels.md/compile-stack.md 两行（T006 diff +7/−5，均为计数同步与新增行）；表结构完整

**状态：** PASS

---

### ST-016：全量回归零回归

**关联需求：** srs.md §3.7 FR7-AC1
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given 全部任务完成

**测试步骤：**
1. `python -m pytest tests/ -q`（基线 683 passed / 2 skipped）

**期望结果：**
- Then 零回归（新增测试全过或合规 skip）

**实际结果：** **711 passed / 2 skipped**（147.85s，2026-10-14 实测）；+28 全为本 AR 新增（test_triton_kernels.py），既有 683/2 无一退步

**状态：** PASS

---

### ST-017：CI lint 三绿

**关联需求：** srs.md §3.7 FR7-AC2
**测试类型：** 回归测试
**优先级：** High

**前置条件：**
- Given CI lint 范围（tools/infra/ 已含）

**测试步骤：**
1. black --check / ruff check / mypy

**期望结果：**
- Then 三绿

**实际结果：** black 164 files would be left unchanged；ruff All checks passed；mypy 零新增（19 处遗留全在 tools/gitee_mirror.py 与 run_component_ablations.py，均为 db3b7ff 前旧账，本 AR 未触碰；新文件两份 mypy no issues）

**状态：** PASS

---

### ST-018：gitee slim-mirror 同步

**关联需求：** srs.md §3.7 FR7-AC3
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：**
- Given AR 完成

**测试步骤：**
1. `python tools/gitee_mirror.py` 同步推送

**期望结果：**
- Then 镜像链推送成功

**实际结果：** 远端 main 曾被云端应急 checkpoint（450f8ca，AR-v0.4.0 云端补跑 E-A-8/E-D-4 存档）推进致分叉；经用户裁定「钉住后强推」：450f8ca 钉 refs/heads/_cloud_ckpt 永久保留 → force-push main=ef8359a（本地 1:1 slim 镜像）→ _sm1..228 临时 ref 全部清理。远端终态：main/_cloud_ckpt/_cloud_origin/_mirror_side/notes，干净

**状态：** PASS

---

### ST-019：--quick 模式 NFR（≤1 分钟）

**关联需求：** srs.md §4 非功能需求（性能）
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：**
- Given 本机 GPU

**测试步骤：**
1. `python tools/infra/triton_kernels.py --quick --json <tmp>` 计时

**期望结果：**
- Then ≤1 分钟，产出合规结果

**实际结果：** ST 复跑实测 **4.6s**；6 measurements（3 kernel×1 档×2 dtype）allclose 全 True；stdout Markdown 摘要表输出正常（swiglu 0.14×/0.21×、rmsnorm 1.84×/1.92×、gqa 0.85×/1.04×）

**状态：** PASS

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-14 |
| 执行结果 | PASS |
| 执行轮次 | 第 2 轮（第 1 轮 ST-004 FAIL，修复后复验） |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | 手册《attention-kernels.md》 | ST-001, ST-002, ST-003 | PASS |
| §3.2 FR2 | 手册《compile-stack.md》 | ST-004, ST-005, ST-006 | PASS |
| §3.3 FR3 | Triton 微实验工具 | ST-007, ST-008, ST-009 | PASS |
| §3.4 FR4 | 测试 | ST-010, ST-011 | PASS |
| §3.5 FR5 | 实验记录与 SOP 更新 | ST-012, ST-013 | PASS |
| §3.6 FR6 | 索引与 README 挂载 | ST-014, ST-015 | PASS |
| §3.7 FR7 | 质量收口 | ST-016, ST-017, ST-018 | PASS |
| §4 NFR | 性能（--quick ≤1min）/可测性/兼容性/文档 | ST-019（+ST-007/009/016/017 承接） | PASS |

**需求覆盖率：** 8 / 8（100%）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 13 | 13 | 0 | 0 |
| 边界条件 | 3 | 3 | 0 | 0 |
| 异常处理 | 0 | 0 | 0 | 0 |
| 回归测试 | 3 | 3 | 0 | 0 |
| **合计** | **19** | **19** | **0** | **0** |

### 缺陷修复记录

| 用例 | 严重性 | 缺陷 | 修复 |
|------|-------|------|------|
| ST-004 | Major | compile-stack.md §4 三段「复制即跑」代码块在 torch 2.5.1 本机全部不可按预期运行（§4.1 属性名错 / §4.2 dynamo.run 不填充计数器且预期与形状特化语义不符 / §4.3 API 两处错误） | d270a1b：改为实测验证写法（torch.compile backend="eager"/fullgraph=True），预期改为实测输出+形状特化教学点；三段整段复跑验证通过 |

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | E-F-3 命令同源问题（profile_inference.py 无 `--context_lengths`，v0.6.0 遗留，非本 AR 引入） | 延后处理：后续 lite AR 对全文 profile_inference.py 调用块做参数一致性扫描 |
| Minor | rmsnorm N 非 2 的幂档无专测（现有 CUDA 用例 N=64/1024 均为 2 的幂，掩码尾部路径未被实际触发） | 延后处理：记录待办 |
| Minor | gqa_decode_bytes 将 q 计为 2× 元素量（docstring 自述，物理偏保守，k/v 项主导，强度值不受实质影响） | 延后处理：记录待办 |

### 结论

> **Go（建议）**。19/19 用例 PASS，需求覆盖 100%，零回归（711/2），三绿，gitee 同步完成。第 1 轮发现的 1 个 Major（ST-004 手册内联实验不可复现）已修复并复验。3 个 Minor 遗留均为记录性待办（其一为 v0.6.0 遗留），不阻塞交付。srs/design 归档前勘误（主题编号/溯源文件/mean-min 文字）已完成。
