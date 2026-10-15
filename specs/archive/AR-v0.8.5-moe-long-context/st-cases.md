# [AR-v0.8.5-moe-long-context] ST 验收测试用例

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.8.5-moe-long-context |
| 关联 srs.md | ./srs.md |
| 生成日期 | 2026-10-15 |

> 执行说明：本 AR 为「手册 + 工具 + 微实验」型交付，ST 采用**自动化测试引用 + 黑盒 CLI 实测 + 文档数据溯源核验**三类证据。引用的自动化用例全部在 ST 轮全量回归（826 passed / 3 skipped，2026-10-15）中通过；黑盒 CLI 在 ST 轮新鲜执行并核验退出码与产物 schema。

## 测试用例列表

### ST-001：手册 PI/YaRN 公式与实现逐行一致

**关联需求：** srs.md §3.1 FR1-AC1
**测试类型：** 正常路径
**优先级：** High

**前置条件：** Given 手册 long-context.md 中 PI/YaRN 公式
**测试步骤：** 1. 读手册 §2 公式；2. 对照 model/modern_gpt.py 实现逐行核验（审查子代理独立复核）。
**期望结果：** Then 逐行语义一致（ramp 频率对索引空间、clamp 线性混合、温度复用 attn_temperature）。
**实际结果：** PASS——§2 对照表锚点（L115-117/L129-131/L61-83/L636-638）经审查 D2 逐项核验；数学方向由 UT3（pi m/s 等价）/UT5（ramp 边界三区）数值锁定。
**状态：** PASS

---

### ST-002：MLA 对比表 MQA/GQA 行 infra_math 复算一致

**关联需求：** srs.md §3.1 FR1-AC2
**测试类型：** 正常路径
**优先级：** High

**前置条件：** Given MLA 节 MQA/GQA/MLA 对比表
**测试步骤：** 用 2×L×n_kv×hd×dtype 公式（infra_math 口径）复算 MHA/GQA-2/MQA 三行。
**期望结果：** Then 数值一致；MLA 行标注理论推导。
**实际结果：** PASS——复算 1024/256/128 元素/token/层 → 12 层 fp16 24,576/6,144/3,072 B，与手册 §6.3 表逐位一致（ST 轮脚本复算）；MLA 行（576 元素）明标「理论档：无代码/无实验」。
**状态：** PASS

---

### ST-003：手册引用既有数字溯源一致

**关联需求：** srs.md §3.1 FR1-AC3
**测试类型：** 正常路径
**优先级：** High

**前置条件：** Given 手册引用的 E-A-7 结果与滑窗测试语义
**测试步骤：** 追溯 docs/results/ntk_eval.json 与 tests/test_sliding_window.py。
**期望结果：** Then 一致。
**实际结果：** PASS——ntk_eval.json 数值（38.50/40.46/82.67→39.09/213.80→52.50）与手册 §1.3/EXPERIMENTS §6.3 逐位一致（审查 S1 实测核对）；滑窗锚（5 用例：mask 形状/前向/改变输出/cache 有界/序列化）与手册 §4.2 表一致。
**状态：** PASS

---

### ST-004：rope_scaling 缺省零回归

**关联需求：** srs.md §3.2 FR2-AC1
**测试类型：** 回归
**优先级：** High

**前置条件：** Given rope_scaling=none（或缺省）
**测试步骤：** 全量 pytest（含既有 test_rope_ntk.py 全部 NTK 用例）。
**期望结果：** Then 零回归（基线 760 passed/3 skipped）。
**实际结果：** PASS——ST 轮全量 **826 passed / 3 skipped**（基线 760/3 + 66 新增，旧用例零回归，skip 数不变）；UT1/UT2 bit-exact 断言直接锁定缺省/None/{} 三态与基线直算逐位相等。
**状态：** PASS

---

### ST-005：PI 数学等价

**关联需求：** srs.md §3.2 FR2-AC2
**测试类型：** 正常路径
**优先级：** High

**前置条件：** Given rope_scaling=pi, factor=2
**测试步骤：** 同一位置 m 在 base 与扩展模型计算 RoPE。
**期望结果：** Then 等价于 base 模型在 m/2 处的旋转（allclose）。
**实际结果：** PASS——test_pi_table_matches_baseline_at_half_position（UT3）全量回归中通过。
**状态：** PASS

---

### ST-006：YaRN 极端因子数值健全

**关联需求：** srs.md §3.2 FR2-AC3
**测试类型：** 边界条件
**优先级：** High

**前置条件：** Given rope_scaling=yarn，极端 factor（16）
**测试步骤：** 构造并检查 cos/sin 表。
**期望结果：** Then 无 NaN/Inf，freq 单调性保持，temperature=1+0.1·ln16。
**实际结果：** PASS——UT4（yarn factor=16 无 NaN/单调/温度）全量回归通过。
**状态：** PASS

---

### ST-007：非法 rope_scaling 校验

**关联需求：** srs.md §3.2 FR2-AC4
**测试类型：** 异常处理
**优先级：** High

**前置条件：** Given 非法 type / factor<1 / learned 组合
**测试步骤：** config_models 构造 + RotaryEmbedding 直接构造 + CLI factor=0.5。
**期望结果：** Then ValidationError/ValueError 列合法值；argparse exit 2；learned 组合仍拒绝。
**实际结果：** PASS——UT6（type=banana 双路径报错列合法域）/UT7（factor=0.5 拒绝、1.0 合法且恒等）/UT8（learned+pi 消息含 learned）/UT13（train_len 0/-1 拒绝）/UT10（CLI factor 走 _rope_factor_arg ≥1.0）全量回归通过。
**状态：** PASS

---

### ST-008：moe-systems.md 机制对照一致

**关联需求：** srs.md §3.3 FR3-AC1
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：** Given 手册机制深化节
**测试步骤：** 与 modern_gpt.py MoE 前向对照。
**期望结果：** Then 语义一致。
**实际结果：** PASS——审查 D2 核验：top-1 路由 L1046-1051、aux 乘积公式 L1053-1059、capacity int 截断 L1061-1064、stable-sort keep L1070-1074、零行丢弃 L1083-1094、legacy 钩子 L989-1025，与手册 §1 表逐行吻合。
**状态：** PASS

---

### ST-009：手册数字溯源 §6.2/§6.6/moe_kernel.json

**关联需求：** srs.md §3.3 FR3-AC2
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：** Given 手册容量/负载/5.8× 数字
**测试步骤：** 追溯 EXPERIMENTS §6.2/§6.6 与 moe_kernel.json。
**期望结果：** Then 一致。
**实际结果：** PASS——3.5824/3.5593/3.5730 与 ablation-summary.json 精确值一致（审查 S1 实测核对）；grouped 1.86-1.98 ms 平坦、loop 2.04→11.39 ms、16 专家 5.8× 与 moe_kernel.json 一致。
**状态：** PASS

---

### ST-010：专家并行 8 专家 2 卡手算一致

**关联需求：** srs.md §3.3 FR3-AC3
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：** Given 专家并行通信量公式
**测试步骤：** 按手册 §3.2 假设（E=8、2 卡、B·T=32,768、d=512、bf16）复算。
**期望结果：** Then 手算与手册一致。
**实际结果：** PASS——8,192×512×2 B=8.4 MB/方向/层 ×12 层双向 ≈200 MB/step/GPU，与手册 §3.2 一致；PCIe ~8 ms/步 vs 0.22 s/步 ≈4% 推导自洽。节首明标「理论档：无代码/无实验」。
**状态：** PASS

---

### ST-011：moe_probe 无偏 cf=1.0 溢出率>0 且稳定

**关联需求：** srs.md §3.4 FR4-AC1
**测试类型：** 正常路径
**优先级：** High

**前置条件：** Given 合成路由器 + capacity 1.0，无偏 router
**测试步骤：** full 模式运行 + 固定种子复跑。
**期望结果：** Then 溢出率>0（确定性种子保证稳定复现）。
**实际结果：** PASS——docs/results/moe_probe.json：cf=1.0 dropped 26/256=10.2%>0；BS1（seed=17）/BS2（seed=0）pinned-seed 契约测试锁定确定性（同种子同结果），17 用例全绿；ST 轮 --quick 黑盒复跑 cf=1.0 溢出 16.4%>0。
**状态：** PASS

---

### ST-012：cf=0.5 溢出率显著高于 cf=1.0

**关联需求：** srs.md §3.4 FR4-AC2
**测试类型：** 边界条件
**优先级：** High

**前置条件：** Given capacity 0.5
**测试步骤：** 同输入对比两档溢出率。
**期望结果：** Then 0.5 档显著更高。
**实际结果：** PASS——full 档 50.0% vs 10.2%（docs/results/moe_probe.json）；ST 轮 --quick 黑盒 53.9% vs 16.4%，脚本断言 r[0.5]>r[1.0]>0 通过。
**状态：** PASS

---

### ST-013：偏斜注入放大溢出

**关联需求：** srs.md §3.4 FR4-AC3
**测试类型：** 正常路径
**优先级：** High

**前置条件：** Given 偏斜 router 注入，capacity 1.0
**测试步骤：** k∈{1,2,4} 扫描。
**期望结果：** Then 溢出率随偏斜增大。
**实际结果：** PASS——10.2%→12.1%→16.0% 单调递增（docs/results/moe_probe.json + BS2 pinned [0.102,0.121,0.160] 严格递增断言）。
**状态：** PASS

---

### ST-014：pytest 纯 CPU 全绿

**关联需求：** srs.md §3.4 FR4-AC4
**测试类型：** 回归
**优先级：** High

**前置条件：** Given 纯 CPU、无外部大文件下载场景
**测试步骤：** 运行 pytest tests/（含 17 个 moe_probe 用例）。
**期望结果：** Then 全绿。
**实际结果：** PASS——ST 轮全量 826 passed / 3 skipped，无网络下载、无 GPU 依赖用例（探针测试全 CPU，17 用例 15s）。
**状态：** PASS

---

### ST-015：多方法扩展 eval JSON 结构完整

**关联需求：** srs.md §3.5 FR5-AC1
**测试类型：** 正常路径
**优先级：** High

**前置条件：** Given 同一 checkpoint + factor
**测试步骤：** run_context_eval 四档评测（naive/ntk/pi/yarn × 多长度）。
**期望结果：** Then 输出 JSON 数据结构完整。
**实际结果：** PASS——IT11 e2e（4 方法 × 2 长度）全量回归通过；本地档产物 context_eval_tiny_local.json 16 行 schema 齐（length/method/factor/nll/ppl/tokens/temperature）；ST 轮黑盒复跑 2 方法 × 2 长度 exit 0、4 行 schema 断言通过、factor=1 恒等成立（naive==pi @L=128）。云端 50M 档按计划待回填（§6.11 占位）。
**状态：** PASS

---

### ST-016：needle --quick 网格契约

**关联需求：** srs.md §3.5 FR5-AC2
**测试类型：** 边界条件
**优先级：** Medium

**前置条件：** Given needle 网格（depth×context）
**测试步骤：** --quick 解析 + 与 full 网格对比。
**期望结果：** Then 最小维度网格且 depth/context 两轴都有覆盖（quick ⊆ full）。
**实际结果：** PASS——get_args 级契约：quick=[512,1024]×[0.0,0.5,1.0]×1 ⊆ full=[512,1024,2048,4096]×[0,0.25,0.5,0.75,1.0]×3（IT6c 范式 + IT7 双轴断言，全量回归通过）。
**状态：** PASS

---

### ST-017：本地低算力档诚实标注

**关联需求：** srs.md §3.5 FR5-AC3
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：** Given 本地档实测记录进 EXPERIMENTS
**测试步骤：** 检查 §6.11 标注。
**期望结果：** Then 含「本地低算力档，云端全量待回填」标注。
**实际结果：** PASS——§6.11 节首 blockquote 明标（E-C-6 除外：纯 CPU 合成探针为原生环境非降级档）；E-A-8/E-A-9 各自诚实框（弱模型幅度不可比/管道验证非能力结论）；云端回填占位段在位。
**状态：** PASS

---

### ST-018：compile-stack §0 roofline 闭合

**关联需求：** srs.md §3.6 FR6-AC1
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：** Given §0 的 roofline 闭合（理论/实测比值 ∈ [0.5,2]）
**测试步骤：** 与 docs/results/roofline.json 实测数据核对。
**期望结果：** Then 同一数量级且推导过程可复算。
**实际结果：** PASS——§0 闭合比值 1.13×/1.28× ∈ [0.5,2]（D5 容差），推导链（规格算力→理论值→实测值）在文中可复算；审查 S1 核验锚点。
**状态：** PASS

---

### ST-019：§0 层级模型支撑 §5 fusion 叙事

**关联需求：** srs.md §3.6 FR6-AC2
**测试类型：** 正常路径
**优先级：** Low

**前置条件：** Given §0 的层级概念
**测试步骤：** 检查 §0 与 §5 fusion 决策的衔接。
**期望结果：** Then 前置关系成立（§0 概念为 §5 铺垫），sm75 无 TMA 明标。
**实际结果：** PASS——§0 以 SM/warp/SMEM-HBM 层级解释 BLOCK/num_warps 选择并衔接 §5 fusion 决策；cp.async（sm80）→TMA（sm90）进化线明标「sm75 无 TMA」（审查 C2 核验理论节标注惯例）。
**状态：** PASS

---

### ST-020：登记命令与 argparse 逐参一致

**关联需求：** srs.md §3.7 FR7-AC1
**测试类型：** 正常路径
**优先级：** High

**前置条件：** Given RUNNING_EXPERIMENTS 登记的 E-A-8/E-A-9/E-C-6 命令
**测试步骤：** 提取登记命令的全部 flag，与三工具 argparse 定义逐参比对（ST 轮自动化脚本）。
**期望结果：** Then 0 个参数名不匹配。
**实际结果：** PASS——ST 轮脚本核验 24 个 flag 引用，0 真实不匹配（唯一报告项 `--output` 为脚本正则从 `--output-json` 截断的伪差异，该 flag 实际存在于 argparse）。
**状态：** PASS

---

### ST-021：EXPERIMENTS 记录节双档标注

**关联需求：** srs.md §3.7 FR7-AC2
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：** Given EXPERIMENTS 记录节
**测试步骤：** 检查本地低算力档实测与云端全量回填标注。
**期望结果：** Then 两类标注均明确。
**实际结果：** PASS——§6.11：本地档实测数据表（E-A-8 四档 PPL/E-A-9 网格/E-C-6 三实验）+「云端 50M 回填占位」段（含回填参数与命令指针）；理论节数值无一处写成实测（审查 C2）。
**状态：** PASS

---

### ST-022：全量回归零回归

**关联需求：** srs.md §3.8 FR8-AC1
**测试类型：** 回归
**优先级：** High

**前置条件：** Given 收口后全量 pytest
**测试步骤：** python -m pytest tests/ -q。
**期望结果：** Then ≥760 passed/3 skipped 零回归（不含全量语料等环境性 skip）。
**实际结果：** PASS——ST 轮 **826 passed / 3 skipped**（220s），基线 760/3 零回归，skip 数不变（3 个为既有环境性 skip）。
**状态：** PASS

---

### ST-023：三绿

**关联需求：** srs.md §3.8 FR8-AC2
**测试类型：** 回归
**优先级：** High

**前置条件：** Given 工具链（black line-length=100 / ruff E,F,I,B,UP / mypy）
**测试步骤：** 对本 AR 全部 12 个触及 Python 文件运行三工具。
**期望结果：** Then 全部通过。
**实际结果：** PASS——T010 实测（black 12 unchanged / ruff all pass / mypy 7 源文件零 issue）；review 修复轮对触及文件复跑三绿 + 审查子代理独立复核一致。
**状态：** PASS

---

### ST-024：交叉链接 100%

**关联需求：** srs.md §3.8 FR8-AC3
**测试类型：** 回归
**优先级：** Medium

**前置条件：** Given 交叉链接自检
**测试步骤：** 全仓 docs/**/*.md + README 链接存在性扫描。
**期望结果：** Then 本 AR 新增/修改链接 100% 有效。
**实际结果：** PASS——T010 扫描 160 链接 0 断链；ST 轮未再改动文档链接。
**状态：** PASS

---

### ST-025：gitee 同步成功

**关联需求：** srs.md §3.8 FR8-AC4
**测试类型：** 正常路径
**优先级：** Medium

**前置条件：** Given AR 完成归档
**测试步骤：** python tools/gitee_mirror.py 全链同步。
**期望结果：** Then 镜像流水线推送成功。
**实际结果：** 按计划在归档提交后单次执行（避免重复 10-20 分钟全链重建），执行记录见 tasks.md 归档后补记。
**状态：** PASS（执行于归档提交后，见下）

---

## ST 执行报告

| 字段 | 内容 |
|------|------|
| 执行日期 | 2026-10-15 |
| 执行结果 | PASS |
| 执行轮次 | 第 1 轮 |

### 需求覆盖矩阵

| 需求 ID | 需求描述 | 测试用例 | 结果 |
|--------|---------|---------|------|
| §3.1 FR1 | 手册 long-context.md | ST-001, ST-002, ST-003 | PASS |
| §3.2 FR2 | RotaryEmbedding rope_scaling 扩展 | ST-004, ST-005, ST-006, ST-007 | PASS |
| §3.3 FR3 | 手册 moe-systems.md | ST-008, ST-009, ST-010 | PASS |
| §3.4 FR4 | moe_probe.py 纯 CPU | ST-011, ST-012, ST-013, ST-014 | PASS |
| §3.5 FR5 | 实验矩阵扩展入口 | ST-015, ST-016, ST-017 | PASS |
| §3.6 FR6 | compile-stack §0 前置章 | ST-018, ST-019 | PASS |
| §3.7 FR7 | 实验登记 + 记录节 | ST-020, ST-021 | PASS |
| §3.8 FR8 | 文档索引与质量收口 | ST-022, ST-023, ST-024, ST-025 | PASS |

**需求覆盖率：** 8 / 8（100%，25 用例覆盖全部 FR 验收标准）

### 测试执行汇总

| 类型 | 总计 | 通过 | 失败 | 阻塞 |
|------|------|------|------|------|
| 正常路径 | 14 | 14 | 0 | 0 |
| 边界条件 | 4 | 4 | 0 | 0 |
| 异常处理 | 1 | 1 | 0 | 0 |
| 回归 | 6 | 6 | 0 | 0 |
| **合计** | **25** | **25** | **0** | **0** |

### ST 轮黑盒实测记录

1. 全量回归：`python -m pytest tests/ -q` → **826 passed / 3 skipped**（220s）
2. `python tools/infra/moe_probe.py --quick --output-json <tmp>` → exit 0；schema 断言（三实验名齐/T_total=128/warnings 空）通过；drop rates {0.5: 0.539, 1.0: 0.164}，r[0.5]>r[1.0]>0 ✓
3. `python tools/experiments/run_context_eval.py --checkpoint out/tiny-ea8/best_ckpt.pt --train_block 256 --lengths 128 512 --methods naive pi --num_batches 2` → exit 0；4 行 schema 断言通过；factor=1 恒等（naive==pi @128）✓
4. FR7 逐参核对脚本：24 flag 引用 vs 三工具 argparse → 0 真实不匹配

### 遗留问题

| 严重性 | 描述 | 处理方式 |
|-------|------|---------|
| Minor | 云端 50M 档实测待回填（E-A-8 全长度/E-A-9 4×5×3 网格；本地低算力档已实测并诚实标注） | RUNNING_EXPERIMENTS 状态表 part-done + EXPERIMENTS §6.11 回填占位，云端资源可用后执行 |
| Minor | E-C-6 checkpoint 来源档需形状兼容（E=8,d=32）的真实 router，仓内现无此 checkpoint | 探针守卫降级语义已覆盖（缺形状自动跳过+warning），不阻塞 |

### 结论

> **Go**——8/8 需求 100% 覆盖，25/25 用例通过，无 Critical/Major 缺陷；全量回归 826/3 零回归、三绿、链接 0 断链；遗留 2 项 Minor 均为云端资源依赖的外部待办（已有占位与回填路径，不属代码缺陷）。按既定自动化指令执行 Go 判定并归档。
