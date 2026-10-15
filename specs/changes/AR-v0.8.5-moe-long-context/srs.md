# [AR-v0.8.5-moe-long-context] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.8.5-moe-long-context |
| AR 主题 | MoE 系统工程 + 长上下文（主题 2/8 深化）+ GPU 执行模型前置章 |
| 关联 SR | 无（学习路径深化，规划输入 docs/ai-infra/roadmap-gap-analysis.md 建议 #2/#6/#7 及用户裁定） |
| 日期 | 2026-10-14 |
| 状态 | Draft |

## 1. 背景与目标

AR-v0.7.5 归档后对学习路径做了系统性缺口分析（roadmap-gap-analysis.md），三大结论
驱动本 AR：① 实验矩阵滞后于代码现状（MoE capacity 机制完整无实验、needle 网格
183 行现成未登记、NTK 只有单点无 PI/YaRN 对照）；② 已交付主题的深化按「并入优于
铺开」处理（MoE/长上下文排 v0.8.5）；③ GPU 执行模型是读懂 Triton BLOCK/num_warps
与 inductor fusion 决策的前置知识缺口。

本 AR 是首个**修改 model/ 代码**的 ai-infra AR（v0.7.5 为纯 docs+tools）：PI/YaRN
需要扩展 RotaryEmbedding——这是「真实代码锚点」原则的必然要求（教学推导必须能对照
真实实现，否则 PI/YaRN 只能纸上谈兵）。

**目标**：主题 2（Attention kernel）/8（数值精度）深化为两本新手册 + 三件工具/
实验入口 + compile-stack §0 前置章，实验矩阵补齐滞后条目，本地低算力档实测先行、
云端全量档待补跑回填。

## 2. 需求范围

**In Scope（本 AR 要做的）：**
- 手册 docs/ai-infra/long-context.md（主题 2 深化）
- 手册 docs/ai-infra/moe-systems.md（主题 8 深化）
- docs/ai-infra/compile-stack.md 新增 §0 GPU 执行模型前置章
- model/modern_gpt.py RotaryEmbedding 扩展（rope_scaling 配置）
- tools/infra/moe_probe.py（纯 CPU）
- tools/experiments/ 两个实验入口（RoPE 扩展对照、needle 网格）
- RUNNING_EXPERIMENTS 实验矩阵登记（E-A-8/E-A-9/E-C-6）+ EXPERIMENTS 记录节
- 索引收口 + gitee 同步

**Out of Scope（本 AR 不做的）：**
- 主题 11 评测系统工程（后续独立 AR）
- 专家并行 / MQA / MLA 的任何代码实现（理论节明标「无代码」）
- KV 家族显存矩阵独立实验（仅 MLA 节内嵌手算对比表）
- F 组云端实验的执行（只登记；云端全量档由 gitee 侧补跑后回填记录节）
- 主题 10 数据构成增补（multilingual/code/mixer/chat_templates，另立小 AR）
- 量化 E-G 组（v0.9.0 前置）
- MTP 消融（归主题 6）

## 3. 功能需求

### 3.1 FR1 手册《long-context.md》（主题 2 深化）

**描述：** 长上下文系统工程手册：NTK 复盘 → PI/YaRN 机制与实现 → 滑窗+cache 驱逐 →
needle 检索 → MLA 理论节。

**触发条件：** T002（RotaryEmbedding 扩展）完成后编写实现节。

**期望行为：** 覆盖六部分：① NTK 复盘（既有 E-A-7 + RotaryEmbedding 锚点 +
run_ntk_eval.py）；② 位置插值 PI 与 YaRN 的机制推导与**本仓实现对照**
（rope_scaling config 路径）；③ 滑窗注意力+cache 驱逐深化（L543-546 +
build_sliding_window_mask + test_sliding_window.py 锚点）；④ needle 检索实验
方法论（depth×context 网格语义、E-A-9 登记）；⑤ **MLA 理论节**：KV 投影压缩
原理、内嵌 MQA/GQA/MLA B/token 手算对比表（infra_math 口径，无代码明标
「理论档」）；⑥ 长上下文显存压力（KV 复习，锚 memory-flops §4）。

**异常处理：** 理论节无代码锚点处必须明标「无实验/无代码」，禁止把文献数值
写成实测。

**验收标准：**
- Given 手册中 PI/YaRN 公式，When 与 RotaryEmbedding 实现（T002）对照，Then 逐行语义一致
- Given MLA 节的 MQA/GQA/MLA 对比表，When 用 utils/infra_math.py 复算 MQA/GQA 行，Then 数值一致（MLA 行标注理论推导）
- Given 手册引用的既有数字（E-A-7 NTK 结果/滑窗测试语义），When 追溯 docs/results/ 与 tests/，Then 一致

### 3.2 FR2 模型代码：RotaryEmbedding rope_scaling 扩展

**描述：** RotaryEmbedding 加 PI/YaRN scaling 选项。

**触发条件：** 开发阶段 TDD。

**期望行为：** `rope_scaling ∈ {none, ntk, pi, yarn}`（config 字段 + 构造参数）：
- none：现状默认，行为逐位不变（向后兼容硬约束）；
- ntk：既有路径保留（现 L69-91 语义不变）；
- pi：位置插值——位置 m 映射 m/scale_factor（线性压缩）；
- yarn：YaRN——NTK-by-parts 插值 + attention temperature 因子
  （文献参数：beta_fast=32, beta_slow=1, factor 由 config 给出）。
- config_models.py 增加 rope_scaling/rope_factor 字段校验；train_pretrain 暴露
  CLI flag。

**异常处理：** 非法 rope_scaling 值 → 配置校验报错（config_models 既有模式）；
yarn 温度因子作用于 attention scale 的方式与文献一致并在手册注明。

**验收标准：**
- Given rope_scaling=none（或缺省），When 既有全部测试（含 NTK 用例）跑，Then 零回归
- Given rope_scaling=pi, factor=2, When 同一位置 m 在 base 与扩展模型计算 RoPE，Then 等价于 base 模型在 m/2 处的旋转（数学等价断言）
- Given rope_scaling=yarn, When 极端 factor（如 16），Then 不出现数值溢出/NaN
- Given rope_scaling=banana（非法值），When 配置构造，Then 明确报错

### 3.3 FR3 手册《moe-systems.md》（主题 8 深化）

**描述：** MoE 系统工程手册：机制深化 → 容量/负载微实验 → 专家并行理论 →
E-A-5 未赢归因 → grouped GEMM 复盘。

**期望行为：** 覆盖五部分：① top-1 路由/aux 负载均衡损失/capacity 溢出丢 token
机制深化（modern_gpt.py:1007-1031 + test_moe_load_balance.py 锚点，Switch
Transformer 语义对照）；② 容量/负载微实验解读（moe_probe.py 产物）；③ **专家
并行理论节**：all-to-all 通信量推导、EP vs DP+replica 的取舍（无代码明标）；④
**E-A-5 未赢归因分析**：moe 组 loss 3.5824 vs 稠密 3.5593（差 0.023）——按
「欠训练（18000 iters 半纪元）/capacity 丢 token/负载不均/参数量口径
（137.4M total vs 59.6M active）」四因子分解，引用 EXPERIMENTS §6.2/§6.6 数据
诚实归因，能说/不能说边界明确；⑤ grouped GEMM 5.8× 复盘（ADR-006 + E-C-3 +
moe_kernel.json，含 superseded 注记的历史脉络）。

**验收标准：**
- Given 手册机制描述，When 与 modern_gpt.py L1007-1031 对照，Then 逐行语义一致
- Given 归因分析的数字，When 追溯 EXPERIMENTS §6.2/§6.6 与 moe_kernel.json，Then 一致
- Given 专家并行通信量公式，When 手算本仓假想配置（如 8 专家 2 卡），Then 与手册算例一致

### 3.4 FR4 工具 moe_probe.py（纯 CPU）

**描述：** MoE 路由与容量的纯 CPU 微实验探针。

**期望行为：** 三实验：① capacity factor 扫描（0.5/1.0/1.25）——固定合成输入下
的溢出 token 率/丢 token 数（调真实 MoE 前向，CPU）；② router 负载分布——top-1
路由的每专家 token 占比、负载数值（合成输入 + 可选真实 checkpoint 状态字典）；
③ 负载不均注入——构造偏斜 router 权重观察 capacity 溢出的放大效应。
JSON schema 沿 v0.7.5 惯例（meta/experiments[].name/source、原子写 tmp+rename、
退出码 0/1/2 语义、--quick 档、数据源缺失守卫降级）。

**异常处理：** checkpoint 缺失→守卫降级跳过该实验退出 0；非法入参退出 2；
实验中断退出 1 不写部分 JSON。

**验收标准：**
- Given 合成输入 + capacity 1.0，When 无偏 router，Then 溢出率≈0（留裕度断言 <5%）
- Given capacity 0.5，When 同输入，Then 溢出率显著高于 1.0 档（方向断言）
- Given 偏斜 router 注入，When capacity 1.0，Then 溢出率高于无偏档（方向断言）
- Given pytest（纯 CPU、无大文件依赖），Then 全过

### 3.5 FR5 实验入口两件：RoPE 扩展对照 + needle 网格

**描述：** ① run_ntk_eval.py 扩展（或并列新入口）支持 none/ntk/pi/yarn 四档对照：
同一 checkpoint 在 base 与扩展上下文下的 PPL/检索对比；② run_needle_eval.py：
包装 evaluation/needle_in_haystack.py 的 depth×context 网格 CLI。

**期望行为：** 两入口均支持 --quick 低算力档（本机 RTX 5000 可跑 3.3M/50M 短程），
产物 JSON 落 docs/results/；云端全量档由 gitee 侧补跑后回填 EXPERIMENTS。

**异常处理：** checkpoint/GPU 缺失→明确报错退出 1（实验工具不同于探针，不做
静默降级）；--quick 档参数集 ⊆ full 档。

**验收标准：**
- Given 同一 50M checkpoint 与 factor，When pi/yarn/ntk 三档扩展 eval，Then 产物 JSON 含四档对照行
- Given needle 网格（depth×context），When --quick，Then 网格规模缩小但维度完整（depth/context 两轴都保留）
- Given 本地低算力实测，When 记录入 EXPERIMENTS，Then 明标「本地低算力档，云端全量待回填」

### 3.6 FR6 compile-stack.md §0 GPU 执行模型前置章

**描述：** SM/warp/occupancy/SMEM-HBM 层级/cp.async/TMA 的教学前置章。

**期望行为：** 覆盖：SM 与 warp 调度模型、寄存器/共享内存/HBM 层级与带宽数量级、
occupancy 概念、异步拷贝（cp.async 一代/二代 TMA——本机 sm75 无 TMA 明标）、
「为什么 BLOCK/num_warps 这样选」与 triton_kernels.py 参数对照、roofline 实测点
的算术闭合（roofline.json 本机数据反推理论峰值）。

**验收标准：**
- Given §0 的 roofline 算术闭合（理论峰值推导），When 用 docs/results/roofline.json 实测点对照，Then 同数量级且推导过程可复算
- Given §0 的层级描述，When 与 compile-stack 既有 §5 fusion 决策衔接，Then 前置关系成立（§0 无依赖可独立先读）

### 3.7 FR7 实验矩阵登记 + EXPERIMENTS 记录节

**描述：** RUNNING_EXPERIMENTS 补三行（E-A-8 RoPE 扩展对照/E-A-9 needle 网格/
E-C-6 MoE 容量扫描——命令与实际 argparse 一致，吸取 E-F-4 教训）；EXPERIMENTS
新增记录节承载本地低算力档实测 + 云端回填占位。

**验收标准：**
- Given 登记的三条命令，When 与两入口+moe_probe 的 argparse 逐参核对，Then 0 个不存在参数
- Given EXPERIMENTS 记录节，When 读，Then 本地档实测数字齐备且云端全量档有明确「待回填」标注

### 3.8 FR8 索引与质量收口

**描述：** ai-infra/README 主题地图挂两深化手册（主题 2/8 行加深化链接）、
roadmap v0.8.5 状态更新；根 README 文档索引；全量回归零回归；三绿；gitee 同步。

**验收标准：**
- Given 收口后全量 pytest，Then 基线 760 passed/3 skipped 零回归（新增全过或合规 skip）
- Given 三绿（black line-length=100/ruff E,F,I,B,UP/mypy），Then 零新增
- Given 链接存在性检查，Then 本 AR 新增/修改引用 100% 存在
- Given AR 完成，When gitee 同步，Then 镜像链推送成功

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 兼容性 | RotaryEmbedding 默认行为 | rope_scaling 缺省=none，既有 NTK 测试与全部 760 用例零回归 |
| 可执行性 | moe_probe full 模式 | 纯 CPU ≤2 分钟 |
| 可执行性 | 实验入口 --quick 档 | 本机 RTX 5000 可跑（50M 短程 ≤30 分钟/网格档） |
| 诚实报告 | 实验记录分档 | 本地低算力档与云端全量档明确区分，理论节明标无代码 |

## 5. 约束与假设

**约束：**
- TDD 纪律：Red（子代理写测试）→ Green → 三绿 → commit；Windows 提交信息用
  UTF-8 临时文件 + `git commit -F`
- 实验矩阵命令登记前必须逐参核对 argparse（E-F-4 教训固化为 ST-C2 惯例）
- 理论节（专家并行/MLA/TMA）一律明标「无代码/无实验」，文献数值不得写成实测

**假设：**
- 本机 Quadro RTX 5000（sm75, 16GB）可跑 50M 短程 eval 与 needle 小网格
- 云端 GPU 资源在位（gitee 侧正按最新仓补跑实验），全量档可回填
- E-A-7 NTK 实验数据（docs/results/ntk_eval.json）与本仓 50M checkpoint 可用

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| PI | Position Interpolation（位置插值，Chen et al., 2023）：位置 m→m/scale 线性压缩 |
| YaRN | Yet another RoPE extensioN（Peng et al., 2023）：NTK-by-parts 插值 + attention 温度因子 |
| NTK | Neural Tangent Kernel 视角的 RoPE 频率外推（既有 E-A-7 路径） |
| capacity factor | Switch Transformer 语义：每专家最多处理 capacity=cf×T/E 个 token，超出丢弃 |
| aux loss | 负载均衡辅助损失 E·Σ(f_i·P_i)（modern_gpt.py:1007） |
| EP | Expert Parallelism：专家分卡 + all-to-all 分发 |
| MQA/MLA | Multi-Query Attention（KV 头=1）/ Multi-head Latent Attention（KV 低秩压缩） |
| TMA | Tensor Memory Accelerator（Hopper 二代异步拷贝；sm75 无，明标） |

## 附：需求澄清记录

- **Round 1（2026-10-14，question 四问）**：交付形态=手册+工具+微实验（v0.7.5
  三段式）；长上下文=PI/YaRN 对照+needle 网格+MLA 理论节（KV 家族矩阵未选）；
  MoE=容量/负载微实验+机制深化+专家并行理论+未赢归因（全四项）；GPU 执行模型
  前置章随本 AR 交付。
- **Round 2（2026-10-14，question 四问）**：载体=两新手册+一前置章；接受修改
  model/（首个改模型代码的 infra AR）；实验执行=本地低算力+云端回填；MLA 节
  内嵌 MQA/GQA/MLA 小型对比表（不单独立项）。
- **暂停点（2026-10-14）**：用户裁定沿用全程自动化指令（免逐步确认、每步反馈
  记录），srs 按 Round 1/2 裁定生成。
