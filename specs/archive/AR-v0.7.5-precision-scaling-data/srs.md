# [AR-v0.7.5-precision-scaling-data] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.7.5-precision-scaling-data |
| AR 主题 | AI infra 学习内容第三阶：数值精度与混精 + scaling laws 与训练预算 + 数据管道与 tokenization（纯 CPU） |
| 关联 SR | 无（学习向内容增强，非业务需求）；上游 AR-v0.6.0-ai-infra-foundations、AR-v0.7.0-ai-infra-kernels（均已归档） |
| 日期 | 2026-10-14 |
| 状态 | Draft |

## 1. 背景与目标

AR-v0.6.0/v0.7.0 交付了主题 1-3（心算 / attention kernel / compile-Triton）。2026-10-14 三路子代理
缺口分析（现状盘点 / 代码资产盘点 / 领域版图评估）确认三个高价值缺口：

- **数值精度（领域版图 D2）**：E-C bf16 2.10× 实测与 AMP/GradScaler 代码（trainer_base.py:208）
  在仓，但 fp16/bf16/fp32/tf32 表示学、loss scaling 机理、累加精度全无手册——大厂面试高频题；
- **Scaling laws（D3）**：budget.json（1,244,464,437 tokens 云端预算）与 ablation-summary.json
  （7 组 loss）是现成拟合素材，但 Kaplan/Chinchilla/compute-optimal/WSD 全无手册，预算规划
  是 infra 岗 per-day 问题；
- **数据管道（D4）**：data/ 12 文件全链（memmap/分片/packing/去重/过滤）+ MIGRATION §3 判据表
  全在，零手册承接——「代码在、教学缺」密度的又一高点。

本 AR 目标：三本分立手册 + 三个纯 CPU 微实验工具闭合上述缺口；顺带修复三项文档漂移/勘误
（缺口分析发现），并把 AI infra 主题地图从 7 主题扩容到 10 主题。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- `docs/ai-infra/numerical-precision.md`：fp32/bf16/fp16/tf32 表示学（范围/尾数/舍入/ulp）、
  AMP autocast 与 GradScaler 机理（trainer_base.py build_amp_context 逐行对照）、fp32 master
  weights 累加精度、E-C bf16 2.10× 复盘、与 compile/roofline/triton 容差口径交叉；fp8 理论
  展望一节（无实验）
- `docs/ai-infra/scaling-laws.md`：Kaplan vs Chinchilla 两条脉络、compute-optimal 推导、
  临界 batch size（McCandlish）、LR schedule（cosine/WSD）、本仓 budget.json/ablation-summary
  拟合复盘（诚实标注样本点少，教学拟合非生产 scaling study）
- `docs/ai-infra/data-pipeline.md`：下载/流式 → tokenize（BPE/vocab 权衡）→ 去重（MinHash/LSH）
  → 过滤 → packing → shard/resume 全链路，逐环节接 data/ 代码锚点 + MIGRATION §3 判据表复盘
- `tools/infra/precision_probe.py` + `tests/test_precision_probe.py`：dtype 表示可视化、
  误差注入/fake-quant 对照、loss scaling 模拟 → `docs/results/precision_probe.json`
- `tools/infra/fit_scaling.py` + `tests/test_fit_scaling.py`：幂律 L(N,D)=E+A/N^α+B/D^β
  拟合、外推预测、预算计算器 → `docs/results/scaling_fit.json`（用已有数据，不新跑训练）
- `tools/infra/data_probe.py` + `tests/test_data_probe.py`：train.bin memmap 抽样（EOT 频率/
  top-token 直方图/doc 长度分布/污染检测演示）+ 合成 corpus 去重/packing 分析 →
  `docs/results/data_probe.json`
- 勘误三件：①主题 6 锚点漂移（ai-infra/README.md:23 speculative 实际在 model/modern_gpt.py:1504+
  与 tools/experiments/run_inference_suite.py；RUNNING_EXPERIMENTS.md B 组示例命令同步勘误）；
  ②ADR-006 补 superseded 注记（旧 bmm+外层循环方案 → 手册 §6 padded [E,cap,C] 方案）；
  ③TECH_REPORT 容量规划 bf16 口径（≈694 MB/50M）与手册 autocast 口径（16 B/param）交叉注
- 索引收口：主题地图 7→10（主题 8 数值精度 / 9 scaling laws / 10 数据管道）、路线图补
  v0.8.5（MoE 系统工程 + 长上下文）行、README 挂载三手册、EXPERIMENTS 新增记录节、
  全量回归 + 三绿 + gitee 同步

**Out of Scope（本 AR 不做的）：**

- 任何 GPU 依赖的实验（全纯 CPU；GPU 侧精度实测如 bf16 训练曲线只引 E-C 既有数据）
- 新跑任何训练（scaling 拟合只用已有 ablation-summary/budget/EXPERIMENTS 数据）
- fp8 实验验证（仅理论展望；fake-quant 实验→量化总论归 v0.9.0 主题 6）
- quantize_model 无生产调用方的改造（仅登记待办，归 v0.9.0 量化总论一并处理）
- 主题 4 分布式/collectives、v0.8.5 MoE/长上下文（后续 AR）
- 修改 data/、model/、training/ 生产代码（只作锚点引用）

## 3. 功能需求

### 3.1 FR1 手册《numerical-precision.md》（主题 8）

**描述：** 数值精度与混精深潜手册。

**触发条件：** 读者理解「bf16 为什么稳、fp16 为什么需要 scaler、混精到底省在哪」。

**期望行为：** 至少覆盖：
1. **四精度表示学**：fp32/bf16/fp16/tf32 的位布局（符号/指数/尾数）、表示范围与 ulp、
   舍入模式（RNE）；bf16 vs fp16 的稳定性根因（指数位 8 vs 5）
2. **AMP 机理**：autocast 的算子白名单/黑名单语义、fp32 master weights、GradScaler 的
   scale/unscale/infskip 循环——与 trainer_base.py build_amp_context(L208) 逐行对照
3. **累加精度**：bf16 直接累加 vs fp32 累加的误差演示（教学级数值实验）；与 triton kernel
   fp32 累加 + input_precision="ieee" 口径互证
4. **本仓验证锚点**：E-C bf16 2.10×（system.json：147.7k vs 70.2k tok/s）、三口径显存账
   （memory-flops §2.1 的 16B vs 12B）、enable_tf32(L49)、test_pretrain_precision、
   triton bf16 容差 1e-2 口径
5. **fp8 理论展望**：E4M3/E5M2 位布局、per-tensor/per-row scaling、为何训练前沿而推理先行
   （无实验，标注「验证归 v0.9.0 量化总论」）

**异常处理：** 无（纯文档）。

**验收标准：**
- Given 手册中的表示学推导（ulp/范围/舍入），When 与 precision_probe.py 工具复算对照，Then 数值一致
- Given AMP/GradScaler 机理描述，When 与 trainer_base.py build_amp_context 对照，Then 逐行一致
- Given 手册引用的存量数据（2.10×/16B/1e-2），When 追溯 system.json/memory-flops/triton_kernels.json，Then 数值一致

### 3.2 FR2 微实验工具 precision_probe.py

**描述：** `tools/infra/precision_probe.py`：纯 CPU 数值精度探针。

**触发条件：** `python tools/infra/precision_probe.py [--quick]`（纯 CPU，零 GPU 依赖）。

**期望行为：**
- dtype 表示剖析：fp32/bf16/fp16 的范围边界、ulp 曲线、RNE 舍入统计
- 误差注入对照：同一组随机张量在四精度下的逐元素相对误差分布（max/mean/p99）
- fake-quant 演示：bf16 权重 + fp32 累加 vs bf16 累加的累积误差对照（小型 matmul，纯 CPU）
- loss scaling 模拟：fp16 梯度下溢演示 + scale 补偿（GradScaler 语义的最小可复现）
- 产物 `docs/results/precision_probe.json`（schema：实验名/dtype/统计量/容差口径）；
  stdout Markdown 摘要
- 无重依赖（仅 torch CPU + 标准库）

**异常处理：** 非法入参 → ValueError；测量中断 → 不写部分 JSON、退出 1。

**验收标准：**
- Given 本机纯 CPU 环境，When 运行，Then 全部实验产出 JSON（≤2 分钟）
- Given bf16 容差口径，When 与 triton_kernels.json 的 atol/rtol=1e-2 对照，Then 口径一致并记入 JSON meta
- Given 测试套件（纯 CPU），When pytest，Then 全过零失败

### 3.3 FR3 手册《scaling-laws.md》（主题 9）

**描述：** scaling laws 与训练预算深潜手册。

**触发条件：** 读者回答「给定预算，模型多大、数据多少、要训多久」。

**期望行为：** 至少覆盖：
1. **Kaplan 脉络**：幂律 L(N)=E+A/N^α、参数效率结论（大模型样本高效）、无数据瓶颈假设
2. **Chinchilla 脉络**：L(N,D)=E+A/N^α+B/D^β、compute-optimal 条件（D/N≈20）、
   与 Kaplan 结论相反的根因（固定 N 扫 D vs 联合拟合）
3. **临界 batch size**（McCandlish）：B_crit 与梯度噪声尺度、小 batch 并行免费/大 batch
   边际递减；与本仓 batch_size=32/梯度累积的对照
4. **LR schedule**：cosine 与 WSD（warmup-stable-decay）退火语义、衰减段与退火损失的
   关系；本仓 cosine 配置锚点
5. **本仓拟合复盘**：budget.json（1,244,464,437 tokens、max_iters=18000、batch 32）的
   compute-optimal 对照；ablation-summary.json 7 组 val loss 的拟合演示与外推——诚实标注
   「7 个点、非受控扫参，教学拟合而非 scaling study」
6. **能说/不能说**：50M/1.24B token 的比例 vs Chinchilla 20:1 的偏差及原因（教学预算 vs
   compute-optimal 预算）

**异常处理：** 无（纯文档）。

**验收标准：**
- Given 手册中的 compute-optimal 推导，When 用 fit_scaling.py 复算本仓 budget.json 对照，Then 结论一致
- Given 手册拟合复盘，When 与 fit_scaling.py 产出的 scaling_fit.json 对照，Then 参数与预测值一致
- Given 手册引用的预算数据，When 追溯 budget.json/ablation-summary.json，Then 数值一致

### 3.4 FR4 微实验工具 fit_scaling.py

**描述：** `tools/infra/fit_scaling.py`：纯 CPU scaling 拟合与预算计算器。

**触发条件：** `python tools/infra/fit_scaling.py [--quick]`（纯 CPU）。

**期望行为：**
- 幂律拟合：ablation-summary.json 7 组 **D 恒定（同预算训练），L(N,D) 的 B/D^β 项不可辨识**，
  故主拟合 Kaplan 式 L(N)=E+A/N^α（scipy 多初值最小二乘，报告拟合优度与参数不确定度）；
  Chinchilla 联合式 L(N,D)=E+A/N^α+B/D^β 在合成数据上演示与验证（2026-10-14 design 阶段
  澄清，见附 Round 4）
- 外推预测：给定 (N*, D*) 预测 loss 并给出区间（教学级，标注方法局限）
- 预算计算器：给定 FLOPs 预算，输出 Chinchilla 口径的 N*/D*/训练时长（时长按本仓吞吐
  口径换算；6ND 口径复用/对照 utils/infra_math.py）
- 产物 `docs/results/scaling_fit.json`（schema：拟合参数/数据源/预测表/预算算例）；
  stdout Markdown 摘要
- 数据源硬编码指向 docs/results/ablation-summary.json + budget.json（存在性守卫，缺失时报错退出）

**异常处理：** 数据源缺失 → 报错退出 1 不写 JSON；拟合不收敛 → 诚实记录并降级报告；
非法入参 → ValueError。

**验收标准：**
- Given 已有 7 组数据，When 运行拟合，Then 产出 JSON 且拟合残差合理（R²>0.9 或诚实记录不达）
- Given 本仓 50M/budget 配置，When compute-optimal 对照计算，Then 输出比例与手册复盘一致
- Given 测试套件（纯 CPU，内置小合成数据），When pytest，Then 全过零失败

### 3.5 FR5 手册《data-pipeline.md》（主题 10）

**描述：** 数据管道与 tokenization 全链路深潜手册。

**触发条件：** 读者理解「token 从语料到 loss 曲线的全旅程」。

**期望行为：** 至少覆盖，每环节接 data/ 代码锚点：
1. **获取与流式**：OpenWebText parquet → curl range 并行下载 → 流式 tokenize 不落盘
   （prepare_openwebtext_shards.py）；1M 默认子集陷阱（prepare.py resolve_max_docs）
2. **Tokenize**：BPE/tiktoken encode_ordinary_batch、vocab 大小权衡（50k vs 100k：embedding
   参数 vs 序列效率）、uint16/uint32 dtype 自动选择（_select_dtype L307）
3. **去重**：MinHash 签名 → LSH 桶 → Jaccard 判定（dedup.py L31/L69/L117 逐行）
4. **过滤**：长度/重复度/正则/fastText 组合过滤链（filter.py CompositeFilter.stats）
5. **Packing**：贪心打包跨文档拼接、doc_boundary 稀疏记录、跨文档 mask 语义
   （PackingDataset L326/_build_samples L349）
6. **Shard 与 resume**：memmap 零拷贝、rank 跨步分片互斥覆盖、rank-local resume_offset
   （MemmapDataset L110/_rank_positions L159）；GlobalShuffledDataset 全局置换表
7. **本仓验证锚点**：MIGRATION §3 判据表（8,926,903,359 tokens/7,913,597 docs/avg 1128
   tokens/EOT 频率）、test_packing 6 用例、data_probe.py 实测对照
8. **数据质量边界**：污染检测思路、质量分 vs 数量的权衡（quality_filter.py 熵分/分层采样）

**异常处理：** 无（纯文档）。

**验收标准：**
- Given 手册中 packing/shard 机制描述，When 与 data/openwebtext.py 对照，Then 逐行锚点一致
- Given MIGRATION §3 判据表数字，When 用 data_probe.py 对本机 train.bin 抽样复测，Then 同类统计量（EOT 频率等）量级一致
- Given 手册去重推导（MinHash/Jaccard），When 与 dedup.py 实现对照，Then 语义一致

### 3.6 FR6 微实验工具 data_probe.py

**描述：** `tools/infra/data_probe.py`：纯 CPU 数据管道探针。

**触发条件：** `python tools/infra/data_probe.py [--bin <path>] [--quick]`（纯 CPU）。

**期望行为：**
- 真实数据剖析（train.bin memmap 抽样，默认 data/openwebtext/train.bin，存在性守卫）：
  token 频率 top-K 直方图、EOT 频率与文档长度分布、dtype/大小/词表范围核验、
  简单污染检测演示（合成 needle 语料注入对照）
- 合成 corpus 实验（不依赖大文件）：MinHash 去重率曲线（重复率扫描）、贪心 packing
  的填充率/padding 浪费对照、doc_boundary 溢出/截断统计
- 产物 `docs/results/data_probe.json`（schema：实验名/数据源（real|synthetic）/统计量）；
  stdout Markdown 摘要
- 测试不依赖 17GB train.bin（合成路径全覆盖 + 真实路径存在性守卫 skip）

**异常处理：** train.bin 缺失 → 跳过真实剖析并警告（合成实验照跑，退出码 0）；非法入参 →
ValueError；测量中断 → 不写部分 JSON、退出 1。

**验收标准：**
- Given 本机 train.bin 存在，When 运行，Then 真实+合成两类实验产出 JSON（≤2 分钟）
- Given 合成 corpus（测试内），When pytest，Then 全过零失败且不依赖大文件
- Given EOT 频率实测，When 与 MIGRATION §3 判据表对照（1/1128≈8.87e-4），Then 量级一致

### 3.7 FR7 勘误三件

**描述：** 缺口分析发现的文档漂移修复。

**期望行为：**
- ①主题 6 锚点漂移：ai-infra/README.md:23「inference/generate.py（speculative 路径）」改为
  实际位置（model/modern_gpt.py `_draft_sample`/`_verify`/`_generate_speculative` +
  tools/experiments/run_inference_suite.py）；RUNNING_EXPERIMENTS.md B 组示例命令中
  `--speculative --draft_model` 等不存在参数同步勘误（与 E-F-4 修复同源方法：改为实际可执行
  的 run_inference_suite 调用）
- ②ADR-006：状态行补 superseded 注记（实现自 2026-10 起演进为 padded [E,cap,C] 静态缓冲
  方案，详见 attention-kernels.md §6；原 bmm+外层循环描述保留作历史记录）
- ③TECH_REPORT.md 容量规划节（≈L1019-1051）：bf16 口径（≈694 MB/50M）与 memory-flops
  §2.1 autocast 口径（16 B/param=0.95 GB）交叉注一句（两口径各自的适用条件）
- quantize_model 无生产调用方：仅在 tasks.md 登记待办（归 v0.9.0 量化总论），本 AR 不修

**验收标准：**
- Given 勘误后的主题 6 锚点，When 与 model/modern_gpt.py/run_inference_suite.py 实际位置核对，Then 一致
- Given 勘误后的 B 组命令，When 与 run_inference_suite.py argparse 核对，Then 参数全部存在（吸取 E-F-4 教训）
- Given ADR-006/TECH_REPORT 勘误，When 读改后文本，Then 注记清晰且不破坏原文档结构

### 3.8 FR8 索引与质量收口

**描述：** 主题地图扩容、README 挂载与 AR 级收口。

**期望行为：**
- ai-infra/README.md：主题地图 7→10（新增主题 8 数值精度与混精 / 9 scaling laws 与训练预算 /
  10 数据管道与 tokenization，各配本仓锚点+延伸阅读，状态「本期已交付」）；路线图代码块补
  v0.8.5（MoE 系统工程 + 长上下文）行并标注 v0.7.5 本期；阅读顺序衔接新手册
- README.md：文档索引挂载三手册；badge 测试数更新（以全量回归实测为准）
- EXPERIMENTS.md：新增记录节（三工具实测，含能说/不能说边界）
- 质量收口：全量回归零回归（基线 711 passed / 2 skipped）、三绿（black/ruff/mypy）、
  gitee slim-mirror 同步

**验收标准：**
- Given 索引更新后，When 链接存在性检查，Then 引用的仓内文件 100% 存在
- Given 全部任务完成，When 全量 pytest，Then 711+2 基线零回归（新增测试全过或合规 skip）
- Given AR 完成，When gitee 同步，Then 镜像链推送成功

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 三工具全量运行 | 各 ≤2 分钟（--quick ≤30 秒） |
| 可移植 | 纯 CPU | 三工具零 GPU/零重依赖（torch CPU + 标准库 + numpy）；测试不依赖 17GB train.bin |
| 兼容性 | 不触碰生产代码 | model/、training/、data/ 零改动；新增仅 tools/infra/、tests/、docs/ |
| 文档 | 中文 | 术语首现给英文；数值断言溯源 results/*.json 或 EXPERIMENTS/MIGRATION；诚实报告（拟合优度/样本点数/能说不能说） |

## 5. 约束与假设

**约束：**
- 全纯 CPU：三工具/三实验/三测试文件零 GPU 依赖（与 v0.7.0 triton 工具的 GPU 依赖形成互补）
- scaling 拟合只用已有数据（ablation-summary.json 7 组 + budget.json + EXPERIMENTS 曲线），
  不新跑训练——样本点少必须诚实标注
- 本机 train.bin（17GB）仅 data_probe.py 运行时抽样读取（memmap 零拷贝），测试一律合成数据
- gitee PNG 预算（~58KB）：手册数值呈现优先表格/JSON
- 云端正在按最新仓库补跑实验：本 AR 全部内容本机可完成，不与云端冲突
- Windows 纪律沿用（提交信息 python pathlib 写 UTF-8 临时文件 + git commit -F）

**假设：**
- torch CPU 路径下 bf16/fp16 表示行为与规范一致（torch 2.5.1 CPU 支持完整）
- ablation-summary.json/budget.json 数据格式稳定（v0.4.0 归档后未变）
- data/openwebtext/train.bin 本机可用（MIGRATION §1 五件套已就位；缺失时 data_probe 守卫降级）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| ulp | unit in the last place：给定浮点格式下相邻可表示数的间距，舍入误差的基本单位 |
| RNE | round-to-nearest-even：IEEE 754 默认舍入模式（银行家舍入，消除统计偏置） |
| autocast | AMP 的算子级精度切换上下文：白名单算子降精度计算、敏感算子（softmax/norm）保持 fp32 |
| GradScaler | fp16 训练的损失缩放器：放大 loss 防梯度下溢，unscale 后检 inf 跳过更新 |
| master weights | 混精训练中优化器持有的 fp32 权重副本（更新精度与表示精度分离） |
| compute-optimal | Chinchilla 口径下给定 FLOPs 预算使 loss 最小的 (N, D) 组合（D/N≈20 tokens/param） |
| B_crit | 临界 batch size：梯度噪声尺度决定的并行收益临界点（McCandlish et al., 2018） |
| WSD | warmup-stable-decay LR schedule：稳定段+短衰减段，退火快于 cosine |
| MinHash | 用最小哈希值集合近似 Jaccard 相似度的文档签名方法（去重第一级） |
| LSH | locality-sensitive hashing：把相似签名聚到同一桶的候选对检索技术 |
| doc_boundary | packing 后记录原文档边界的稀疏索引（跨文档 mask 的数据基础） |
| packing | 将多个短文档拼接进固定长度训练序列的 token 效率技术（配套跨文档 mask） |

## 附：需求澄清记录

- 2026-10-14 用户裁定（Round 1）：三本分立手册（主题地图 7→10 扩容）；三工具三实验全 CPU；
  scaling 拟合只用已有数据不新跑训练；勘误纳入锚点漂移+ADR-006+口径交叉三件，量化项仅登记。
- 2026-10-14 用户裁定（Round 2）：精度手册四精度+fp8 理论展望（无 fp8 实验）；数据管道手册
  全链路（获取→tokenize→去重→过滤→packing→shard/resume）；工具落 tools/infra/ 三件对齐
  run_roofline/triton_kernels 先例；交付形态全周期对齐 v0.7.0 节奏（JSON→EXPERIMENTS→索引→
  README→回归→gitee）。
- 2026-10-14 用户裁定（Round 3）：scaling 手册 Kaplan+Chinchilla 全脉络+本仓复盘；data_probe
  数据源用本机 train.bin memmap 抽样（测试合成守卫）；主题编号顺延 8/9/10；路线图补 v0.8.5 行。
- 2026-10-14 缺口分析（三路子代理）：数值精度/scaling laws/数据管道为未覆盖域中结合度与
  CPU 可完成度最高者；另发现 4 项文档漂移（3 项纳入本 AR 勘误，1 项登记）。
- 2026-10-14 用户裁定（Round 4，design 门控第 1 轮触发）：FR4 拟合式精化——ablation 7 组
  同预算训练 D 恒定，L(N,D) 联合式的 B/D^β 项不可辨识；主拟合改为 Kaplan 式
  L(N)=E+A/N^α，Chinchilla 联合式在合成数据上演示验证；预算计算器补训练时长输出
  （按本仓吞吐口径换算）。srs FR4 期望行为已同步更新。
