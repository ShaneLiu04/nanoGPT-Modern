# [AR-v0.6.0-ai-infra-foundations] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.6.0-ai-infra-foundations |
| AR 主题 | AI infra 学习内容第一階：显存/算力心算 + 学习路径索引 |
| 关联 SR | 无（学习向内容增强，非业务需求） |
| 日期 | 2026-10-13 |
| 状态 | Draft |

## 1. 背景与目标

nanoGPT-Modern 已具备完整的「现代 LLM 训练-推理-对齐」代码资产与大量真实实验数据（本地 8GB
Quadro RTX 5000 + 云端 RTX 6000D 实测），但从**自学习/教学**视角看，AI infra 领域的知识
散落在代码注释、ADR、TECH_REPORT 与实验记录中，缺乏系统性、公式化、可验证的学习材料。

经覆盖面深度分析（2026-10-13，见需求澄清记录），识别出七大 AI infra 学习缺口。为保证质量
拆分为 4 个 AR 交付：

| AR | 主题 | 状态 |
|----|------|------|
| **AR-v0.6.0（本 AR）** | 主题 1 显存/算力心算 + 学习路径索引骨架 | 本期 |
| AR-v0.7.0-ai-infra-kernels | 主题 3 Attention kernel 深潜 + 主题 5 torch.compile/Triton | 计划 |
| AR-v0.8.0-ai-infra-distributed | 主题 2 分布式训练系统 + 主题 7 弹性容错 | 计划 |
| AR-v0.9.0-ai-infra-serving-rl | 主题 4 推理服务引擎 + 主题 6 RLHF 系统工程 | 计划 |

本 AR 是整个系列的**地基**：显存/算力公式是后续所有主题（ZeRO 分片数学、chunked GRPO
显存、KV cache 物化、PD 分离带宽）的公共引用；学习路径索引是全系列的导航骨架。

**核心定位**：每个公式都要能被本仓**真实数据交叉验证**（区别于博客式转载）——
这是本仓库学习材料的核心竞争力。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- `docs/ai-infra/` 新目录与学习路径索引（curriculum：7 主题 → 本仓代码/实验/ADR 锚点 → 4-AR 路线图 → 延伸阅读）
- 显存/算力心算深度手册（`docs/ai-infra/memory-flops-arithmetic.md`）：参数/梯度/优化器状态/激活（含 gradient checkpointing 折减）/KV cache 逐 token 公式推导；6ND 训练 FLOPs、attention 二次项、prefill vs decode 算术强度；roofline 模型与 memory-bound vs compute-bound 判据
- 可验证计算器工具：模型配置 → 显存五项分解（GB）、FLOPs/step（MFU 定义入手册、实测 MFU 归 roofline/E-F-2，见 design D4）；公式被**单测锁定**
- 本机 roofline 微实验脚本（matmul 尺寸扫描 + decode memory-bound 验证），实测数据入手册
- RUNNING_EXPERIMENTS.md 新增 F 组实验 SOP（完整 GPU 实验文档化，待高算力机器执行）
- README 文档索引挂载 ai-infra 系列

**Out of Scope（本 AR 不做的）：**

- 主题 2/3/4/5/6/7 的正文内容（分别由 AR-v0.7.0/0.8.0/0.9.0 承接；本 AR 索引中留占位与锚点）
- 完整 GPU 实验**实际执行**（如多卡通信带宽、PD 分离测量——仅文档化 SOP，待高算力 GPU）
- 训练/推理代码路径的行为变更（计算器/微实验为**纯增量**，不触碰现有训练链路）

## 3. 功能需求

### 3.1 FR1 学习路径索引（curriculum 骨架）

**描述：** `docs/ai-infra/README.md` 作为 AI infra 学习系列的导航入口。

**触发条件：** 读者从 README 文档索引进入。

**期望行为：** 覆盖 7 大主题的学习地图：每主题给出（a）核心问题（b）本仓代码/实验/ADR
锚点（c）所属 AR 与交付状态（d）2-3 条延伸阅读（论文/源码/博客）；附推荐学习顺序
（显存心算 → kernel → 分布式 → serving/RL）。

**异常处理：** 未交付主题明确标注「AR-vX.Y 计划中」，不出现死链接。

**验收标准：**
- Given docs/ai-infra/README.md 已创建，When 检查 7 个主题条目，Then 每条均含本仓锚点（文件路径或实验编号）与所属 AR 标注
- Given 主题尚未交付，When 索引引用之，Then 显示「计划中」而非链接到不存在文件
- Given 任意已交付文档，When 从索引点击路径，Then 路径真实存在（索引内引用的仓内文件 100% 存在）

### 3.2 FR2 显存/算力心算手册

**描述：** `docs/ai-infra/memory-flops-arithmetic.md` 深度自学习参考手册。

**触发条件：** 读者学习显存/FLOPs 心算，或准备面试。

**期望行为：** 至少覆盖以下推导与验证，每节公式后接「本仓验证」小节：
1. **显存五项分解**：参数（×精度）、梯度、Adam 优化器状态 m/v（两种计账口径均讲：本仓 autocast——参数即 fp32 主权重，静态 4+4+8=16 B/param；bf16 纯权重流派 2+2+8=12 B/param；计算器默认前者）、激活（逐层推导 Transformer block 显存，含 attention scores [B,H,T,T]、SwiGLU 中间维、gradient checkpointing 只存层边界输入的折减）、KV cache（2·L·n_kv·d·T·B bytes/token 序列公式）
2. **FLOPs**：6ND 前反向启发式推导（乘加 2FLOP×前向矩阵乘×反向两倍）、attention 的 2·T²·d 项与 T²·H scores、MoE 稀疏 FLOPs
3. **算术强度与 roofline**：FLOPs/byte 定义、matmul/attention/GEMM 的强度曲线、memory-bound vs compute-bound 判据、decode（batch=1）为何必然 memory-bound（每 token 读全部权重 → 强度 ≈ 1）
4. **本仓验证锚点**（必须逐条数值对照）：moe 组 +78M 参数 ≈ +1.25GB 优化器状态（AR-v0.4.0 记录）、KV 物化实测斜率 12,335 B/token（24L 模型，**与层数无关**——逐层串行执行 + 分配器复用使峰值 ≈ 单层瞬时 12,288 B/token = 3×16heads×64d×4B；MIGRATION §8.2）、decode 68→64.5ms/step 与带宽反推、云端 E-A 组峰值显存（baseline 23.1 / modern-full 26.3 / moe 29.3 GB，RTX 6000D）——moe 与 modern-full 实测组间差 3.0GB 中静态部分（优化器态 +1.25GB 理论）须与计算器核对，剩余激活/碎片口径须在手册中显式拆解；本机 15.49GB/16GB WDDM 溢出案例（AR-v0.4.0 T007）作为「不干净锚点」反面教学案例

**异常处理：** 无（纯文档）。

**验收标准：**
- Given 手册中每个显存公式，When 与本仓实测数据对照，Then 偏差需给出解释（±10% 内或注明测量口径差异）
- Given 手册引用本仓实验数据，When 追溯 docs/results/*.json 或 EXPERIMENTS.md，Then 数值一致
- Given 读者只看手册，When 手算 50M 模型 bf16 训练显存，Then 与 FR3 计算器输出一致（互为验证）

### 3.3 FR3 显存/FLOPs 计算器（可验证工具）

**描述：** 新增 `utils/infra_math.py` 纯函数计算模块 + CLI，手册公式与工具互锁。

**触发条件：** `python -m utils.infra_math --n_layer 12 --n_embd 768 ...` 或被测试/微实验导入。

**期望行为：**
- `param_count(...)`：按 GPT 结构（embedding/attention/FFN，含 GQA n_kv_head、MoE num_experts、tie_weights）推参数量
- `training_memory_breakdown(...)`：返回参数/梯度/优化器状态/激活（支持 gradient_checkpointing 开关）/KV cache（训练 eval 用）五项 GB dict + 总计
- `inference_kv_bytes(...)`：给定 (n_layer, n_kv_head, head_dim, seq_len, batch, dtype) → KV 总字节与 B/token 斜率
- `flops_per_token(...)`/`flops_per_step(...)`：6ND 模型项 + attention 序列长度项
- `arithmetic_intensity(...)`：给定 op 类型（matmul/decode）→ FLOPs/byte
- CLI：`--precision/--grad_ckpt/--seq_len/--batch/--gqa/--moe`，输出 Markdown 表格

**异常处理：** 非法组合（moe 但 ffn_variant=swiglu 交互等）抛 ValueError；所有输入负数/零校验。

**验收标准：**
- Given 50M modern-full 配置（EXPERIMENTS 云端实测 59.561M 参数），When `param_count` 计算，Then 误差 <1%
- Given Adam+bf16 混合精度（**本仓 autocast 口径**：参数即 fp32 主权重，无独立 bf16 权重副本，见 trainer_base.py build_amp_context），When `training_memory_breakdown`，Then 总静态 = 16 B/param（参数 4 + 梯度 4 + optimizer m/v 8，即 optimizer 项 = 8 B/param）；另 Given `--precision bf16-pure`（bf16 纯权重流派），Then 总静态 = 12 B/param（参数 2 + 梯度 2 + optimizer m/v 8）——两种口径在 docstring 与手册均注明，计算器默认仓内 autocast 口径
- Given KV cache 配置 (L=12, n_kv=6, d=64, T, B=1)，When `inference_kv_bytes`，Then 与 manual 2·L·n_kv·d·T 公式逐值一致
- Given 本仓实测数据点（KV 物化斜率 12,335 B/token、moe 优化器态 +1.25GB），When 计算器复算，Then 数值吻合（测试内嵌这些回归断言）
- Given `python -m utils.infra_math --help`，When 执行，Then 退出码 0

### 3.4 FR4 roofline 本机微实验

**描述：** `tools/infra/run_roofline.py`（或 tools/profiling/ 下）本机可跑的微型实验：
GPU matmul 尺寸扫描（小→大，跨越 L2/HBM 工作集），测 GFLOPs/s 并与理论峰值对比，绘制
roofline 轨迹；单 batch decode 步测量（时间/步 → 反推有效带宽）验证 memory-bound。

**触发条件：** `python tools/infra/run_roofline.py --quick`（约 1-3 分钟）。

**期望行为：** 输出 JSON（docs/results/roofline.json）+ Markdown 摘要（尺寸/FLOPs/耗时/GFLOPs/s/带宽利用率/判定 bound 类型）；`--quick` 模式可并入 CI 冒烟级测试（CPU fallback：CUDA 不可用时跳过 GPU 部分，仅验证逻辑）。

**异常处理：** 无 CUDA → 打印跳过原因并以 CPU 小矩阵冒烟（保证脚本逻辑可测）；实验异常不产生部分 JSON。

**验收标准：**
- Given 本机 Quadro RTX 5000（sm_75），When 运行 `--quick`，Then 完成于 5 分钟内且 JSON 含 ≥10 个尺寸点
- Given 小尺寸点（工作集 < L2），When 与大尺寸点对比，Then 小尺寸 GFLOPs/s 显著低于峰值（缓存效应可见）或注明与预期差异
- Given decode 步实测时间，When 反推有效带宽 = 权重字节/时间，Then 与 GPU 标称带宽对比并输出利用率（分母以 `nvidia-smi` 实测卡型查规格表为准——本机 Quadro RTX 5000 桌面版 16GB/448GB/s，nvidia-smi 实测 16384 MiB）
- Given 测试套件（pytest，无 GPU 环境），When 导入模块函数（纯计算部分），Then 冒烟通过

### 3.5 FR5 F 组实验 SOP（延后执行的完整实验文档化）

**描述：** RUNNING_EXPERIMENTS.md 新增「F 组：AI infra 系统实验」，把需要高算力 GPU
的实验文档化为可执行 SOP（命令 + 预期产物 + 判读标准），本 AR 只交付文档不执行。

**触发条件：** 后续获得高算力 GPU（云端/多卡）。

**期望行为：** F 组至少含：E-F-1 通信带宽微基准（all-reduce Ring 实测 vs 2(N-1)/N·V/B 理论，为 AR-v0.8.0 主题 2 供数据）、E-F-2 多卡 roofline 大矩阵（MFU 测量）、E-F-3 KV cache 物化斜率 GPU 复测（对照本机 12,335 B/token 发现）；每项含环境要求、命令、产物路径、成功判据。

**异常处理：** 无（纯文档）。

**验收标准：**
- Given RUNNING_EXPERIMENTS.md F 组三项，When 逐项检查，Then 均含环境要求/命令/产物/判据四要素
- Given 状态跟踪表，When 查 F 组条目，Then 标注「待高算力 GPU（AR-v0.6.0 文档化）」
- Given 本仓现有命令风格（编排器/断点续跑），When F 组命令对照，Then 与 RUNNING_EXPERIMENTS 现有 A-D 组风格一致

### 3.6 FR6 README 索引挂载

**描述：** README.md 文档索引新增 AI infra 学习系列入口（含学习顺序建议一句话）。

**验收标准：**
- Given README 文档索引，When 检查，Then 含 docs/ai-infra/ 入口与 7 主题一行摘要
- Given README 三条阅读线（为什么读这个仓库），When AI infra 系列挂载，Then 不破坏现有阅读线结构（增量一行/一节）

### 3.7 FR7 质量收口

**验收标准：**
- Given 全部任务完成，When `pytest -m "not perf"`，Then 当前基线（2026-10-13 实测：640 passed / 2 skipped / 2 deselected，含 3f18682 新增 12 用例）全过零回归（新增测试全过）
- Given CI lint 范围（含本 AR 新增的 utils/infra_math.py 与 tools/infra/），When mypy/black/ruff，Then 三绿
- Given 本部分完成，When gitee 同步，Then 镜像链推送成功

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | 计算器 CLI | 单次调用 < 1s（纯 CPU 数学） |
| 可测性 | 公式锁定 | 手册每个显存公式至少 1 条单测断言对应 |
| 兼容性 | 不触碰训练链路 | 现有测试零回归（基线 640 passed，2026-10-13 实测）；新增仅 utils/tools/docs |
| 文档 | 中文 | 术语首现给英文；与 ADR/TECH_REPORT 深度一致 |

## 5. 约束与假设

**约束：**
- 遵循用户裁定：7 主题全做但分 4 个 AR；完整 GPU 实验仅 SOP 文档化（待高算力 GPU）
- Windows 本机无 NCCL（gloo only）、单卡 8GB——微实验必须在此环境可跑（CPU fallback 可测）
- docs/ai-infra/ 内容进 git（PNG 若 >95KB 会被 gitee 镜像压缩——微实验图优先表格/JSON 数据）

**假设：**
- 云端正在按最新仓库补跑实验（A-D 组），F 组 SOP 不与其冲突（F 组面向后续高算力窗口）
- AR-v0.4.0 的 T011-T013（图表/文档）后续由云端数据触发，与本 AR 文档改动无文件冲突（README 索引节为增量）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| MFU | Model FLOPs Utilization，实测 FLOPs/s ÷ 理论峰值 FLOPs/s |
| 算术强度 | FLOPs ÷ 读写字节数（FLOPs/byte） |
| roofline | 算术强度 vs 峰值性能的瓶颈分析模型 |
| ZeRO | DeepSpeed 的零冗余优化器分片（1: 优化器态 2: +梯度 3: +参数）——本 AR 仅在手册中引用，正文归 AR-v0.8.0 |
| KV 物化 | SDPA enable_gqa 在无原生 GQA 内核的后端展开 KV head 导致的显存放大（本仓 MIGRATION §8.2 实测 12,288·L B/token） |

## 附：需求澄清记录

- 2026-10-13 用户裁定：七主题全做、分多 AR 保质量；形态=文档+小工具+微实验，完整实验先 SOP 文档化待高算力 GPU；深度=自学习参考；语言=中文。
