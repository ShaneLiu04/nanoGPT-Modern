# [AR-v0.7.0-ai-infra-kernels] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.7.0-ai-infra-kernels |
| AR 主题 | AI infra 学习内容第二阶：Attention kernel 深潜 + torch.compile/Triton 编译栈 |
| 关联 SR | 无（学习向内容增强，非业务需求）；上游 AR-v0.6.0-ai-infra-foundations（已归档） |
| 日期 | 2026-10-13 |
| 状态 | Draft |

## 1. 背景与目标

AR-v0.6.0 交付了主题 1（显存/算力心算）三件套并归档。深度缺口分析（2026-10-13，三路
子代理盘点文档/代码/数据资产）确认：

- **代码在、教学缺**密度最高的区域是主题 2（attention kernel）与主题 5（compile 栈）：
  GQA 三种 kernel 路径（grouped-broadcast / enable_gqa / 手写 decode）、ring_attention
  分块参照实现、MoE padded grouped-GEMM、torch.compile 全链路（含 fullgraph decode 编译）
  代码全部就位，但无一篇手册进入 kernel/编译器内部；
- **约一半实测数据未被手册承接**：MoE grouped GEMM 5.8×、KV 物化斜率 12,335 全案、
  compile 1.59×（256 tok）与 64 tok 负结果、GQA −75% 等锚点数据仍只在 EXPERIMENTS.md；
- **Triton 零实现**是代码空白，但本机环境已验证可用（triton 3.8.0 + torch 2.5.1+cu121，
  手写 @triton.jit kernel 在 Quadro RTX 5000 冒烟通过）——具备「写真实 kernel + 本机实测」
  的条件。

本 AR 目标：交付主题 2 + 主题 5 的两本深度手册 + 三个 Triton 微实现 kernel（独立微实验，
不接入模型），把上述「代码在、教学缺」与「数据无承接」两类缺口同时闭合。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- `docs/ai-infra/attention-kernels.md`：SDPA backend 生态与探测逻辑；flash 分块 +
  online-softmax 完整数学推导（以本仓 ring_attention.py 纯 PyTorch 参照实现逐行对照）；
  GQA 三 kernel 路径深潜（含 KV 物化 bug 全案复盘）；MoE grouped GEMM static-shape 教学；
  未引用存量数据嵌入正文
- `docs/ai-infra/compile-stack.md`：torch.compile 三层架构（dynamo→inductor→cudagraphs）；
  本机 dynamo 层实测（recompile 计数）；inductor/Triton codegen 理论；triton 3.8 vs
  torch 2.5.1 断裂案例教学化；compile 存量数据（1.59×/负结果）嵌入
- `tools/infra/` Triton 微实验：fused SwiGLU / fused RMSNorm / GQA decode scores 三个
  `@triton.jit` kernel + PyTorch eager 参照 + 数值一致性（atol/rtol）+ A/B 计时 +
  算术强度标注 → `docs/results/triton_kernels.json`
- `tests/test_triton_kernels.py`：无 GPU/无 triton 环境可测（参照实现纯函数级单测 +
  importorskip 降级）
- RUNNING_EXPERIMENTS.md 新增 E-F-4（云端 inductor 实测 SOP）；EXPERIMENTS.md 新增
  Triton 微基准记录节
- docs/ai-infra/README.md 主题 2/3 状态转「已交付」；顺带修正「7 篇 ADR」笔误（实为 6 篇）
- README 文档索引挂载两本新手册；质量收口 + gitee 同步

**Out of Scope（本 AR 不做的）：**

- Triton kernel 接入 modern_gpt.py forward（用户裁定：独立微实验，不触碰训练链路）
- Triton flash attention 教学实现（sm75 无 TMA，价值/风险比不划算——手册推导 +
  ring_attention 参照已覆盖教学目标）
- 主题 4 分布式 / 主题 6 推理服务 / 主题 7 RLHF 工程（归 v0.8.0/v0.9.0）
- 本机 inductor 环境修复（降级 triton 3.1 会破坏手写 kernel 依赖；compile 实测限
  dynamo 层，inductor 层理论 + 云端 E-F-4 SOP）
- 量化理论（归 v0.9.0 serving AR）

## 3. 功能需求

### 3.1 FR1 手册《attention-kernels.md》

**描述：** 深度自学习参考手册，进入 attention kernel 内部。

**触发条件：** 读者学习 kernel/算子层，或准备面试。

**期望行为：** 至少覆盖以下推导与验证，每节公式后接「本仓验证」小节：
1. **SDPA backend 生态**：flash/mem-efficient/math 三后端能力矩阵与选择逻辑
   （attention_utils.py 探测路径逐行对照）；is_causal 折叠 vs 加性 mask 的代价
2. **flash 分块推导**：为什么 [B,H,T,T] 物化不可行 → tiling 分块 + online-softmax
   （running max/sum 状态递推）完整数学推导；以 ring_attention.py 纯 PyTorch 分块
   实现为参照逐行对照「推导↔代码」
3. **GQA 三 kernel 路径**：grouped-broadcast 零拷贝（reshape/singleton 广播语义 vs
   repeat_interleave 拷贝）；enable_gqa 原生路径（含内部 KV 头扩展物化陷阱——MIGRATION
   §8.2 全案：斜率 12,335 B/token 定位→修复→114× 显存削减）；手写 decode kernel
   （T==1 按 KV-head 原生分辨率算 scores）
4. **MoE grouped GEMM**：Python 逐专家循环的 launch 开销 → stable-sort 分桶/capacity
   截断/static-shape [E,cap,C] 批量 GEMM → 为什么静态形状利于 kernel；load-balance
   aux loss 的向量化
5. **本仓验证锚点**（存量数据嵌入正文）：MoE grouped GEMM 5.8×（16 专家）、KV 物化
   12,335→修复后平坦、GQA-2 KV −75%、本机 roofline L2 边界 1160×（cache 效应对
   kernel 设计的含义）

**异常处理：** 无（纯文档）。

**验收标准：**
- Given 手册中 flash/online-softmax 推导，When 与 ring_attention.py 实现对照，Then 每个推导步骤有对应代码行锚点
- Given 手册引用的存量数据（5.8×/12,335/−75%），When 追溯 moe_kernel.json/MIGRATION §8.2/ablation-summary.json，Then 数值一致
- Given GQA 三路径，When 与 modern_gpt.py:163-297 对照，Then 语义一致（何时走哪条路径的判定条件与代码一致）

### 3.2 FR2 手册《compile-stack.md》

**描述：** torch.compile 编译栈深度手册（分层实测+理论）。

**触发条件：** 读者理解「compile 为什么快/何时慢/何时重编译」。

**期望行为：** 至少覆盖：
1. **三层架构**：dynamo（FX 图捕获/guard/重编译语义）→ inductor（fusion 决策/Triton
   codegen/自动调优）→ cudagraphs（reduce-overhead 模式的图录制与显存语义）
2. **本机 dynamo 层实测**：recompile 计数实验（动态 shape/字典/属性突变各触发一次
   重编译的可复现演示，对齐 MIGRATION §8.1 方法）；fullgraph 约束与图断裂
3. **inductor 理论**：fusion 规则（逐元素/归一化/_reduction 合并）、codegen 流水线、
   自动调优（max-autotune）；与手写 Triton（FR3）的分工
4. **环境断裂案例教学化**：本机 triton 3.8 vs torch 2.5.1 inductor API 断裂的机理
   （inductor 绑定特定 triton 版本），及其对「工具链版本矩阵」的一般教训
5. **本仓验证锚点**：compile 256 tok 1.59× vs 64 tok 负结果的分层归因（编译成本摊平
   边界）；FSDP→compile 包裹顺序（ARCHITECTURE §2）；compile 与 EMA/checkpoint 的交互
6. **云端 SOP 延伸**：inductor 层实测（E-F-4，见 FR5）

**异常处理：** 无（纯文档）。

**验收标准：**
- Given dynamo 层 recompile 演示，When 本机复跑，Then 计数变化可复现（脚本化，MIGRATION §8.1 先例）
- Given compile 存量数据（1.59×/负结果），When 追溯 inference.json/EXPERIMENTS §6.4，Then 数值一致且归因口径一致（编译成本摊平边界）
- Given 三层架构描述，When 与 trainer_base.py/modern_gpt.py compile 调用点对照，Then 配置项（mode/fullgraph/dynamic）语义一致

### 3.3 FR3 Triton 微实验工具

**描述：** `tools/infra/triton_kernels.py`：三个教学级 Triton kernel + 基准。

**触发条件：** `python tools/infra/triton_kernels.py [--quick]`（本机 GPU）。

**期望行为：**
- 三个 `@triton.jit` kernel：**fused SwiGLU**（gate/up/silu·mul 三算子合一）、
  **fused RMSNorm**（逐行归一化，含 rsqrt 融合）、**GQA decode scores**（对标仓内
  手写 _decode_gqa_attention 的 [B,n_kv,rep,S] 路径）
- 每个 kernel 配 PyTorch eager 参照实现（亦为无 GPU 环境的纯函数单测对象）
- 数值一致性：atol/rtol 断言（fp32/bf16 两口径）
- A/B 计时（CUDA events，warmup+iters）+ 算术强度标注（复用 utils/infra_math 口径）
- 产物 `docs/results/triton_kernels.json`（schema：kernel 名/形状/精度/参照耗时/
  triton 耗时/加速比/强度/一致性容差）；stdout Markdown 摘要（表格替代绘图）
- 无 CUDA 或无 triton：打印原因并以 CPU 参照实现冒烟（保证逻辑可测），退出码语义化

**异常处理：** 无 CUDA/无 triton → CPU 冒烟路径；测量中断 → 不写部分 JSON、退出 1；
非法入参 → ValueError。

**验收标准：**
- Given 本机 Quadro RTX 5000 + triton 3.8.0，When 运行，Then 三个 kernel 全部通过数值一致性断言并产出 JSON（≤5 分钟）
- Given eager 参照与 Triton 输出，When fp32 口径比对，Then allclose（atol=1e-5/rtol=1e-4 量级，容差在 JSON 中记录）
- Given 测试套件（无 GPU 环境），When pytest，Then 参照实现纯函数单测通过、triton 用例 importorskip 跳过而非失败

### 3.4 FR4 测试

**描述：** `tests/test_triton_kernels.py`。

**期望行为：** 参照实现（eager SwiGLU/RMSNorm/GQA scores）纯函数级单测（无 GPU 依赖，
含与 model/modern_gpt.py 同类算子的锚点一致性，如 RMSNorm 与仓内实现对照）；
Triton kernel 用例 `pytest.importorskip("triton")` + GPU 可用性守卫；非法入参 ValueError。

**验收标准：**
- Given 无 GPU/无 triton 的 CI 环境，When pytest tests/test_triton_kernels.py，Then 全部通过或 skip，零失败
- Given 参照 RMSNorm 实现，When 与 model/modern_gpt.py RMSNorm 前向对照（fp32），Then 数值一致

### 3.5 FR5 实验记录与 SOP 更新

**描述：** EXPERIMENTS.md 新增 Triton 微基准记录节；RUNNING_EXPERIMENTS.md 新增 E-F-4。

**期望行为：**
- EXPERIMENTS.md：Triton 三 kernel 实测记录（表格 + 能说/不能说边界，如「教学级
  kernel 未做 autotune/向量化调优，加速比不设阈值诚实报告」）
- RUNNING_EXPERIMENTS.md：E-F-4 云端 inductor 实测 SOP（环境/命令/产物/判据四要素，
  对齐 F 组格式）；状态表登记；更新历史追加

**验收标准：**
- Given E-F-4 条目，When 逐项检查，Then 四要素齐备且命令风格与 F 组一致
- Given EXPERIMENTS 记录节，When 与 triton_kernels.json 逐值比对，Then 一致（舍入口径注明）

### 3.6 FR6 索引与 README 挂载

**描述：** 学习路径索引与主 README 更新。

**期望行为：** docs/ai-infra/README.md 主题 2（attention kernel）与主题 3（compile/
Triton）状态转「本期已交付」并挂手册链接；修正「7 篇 ADR」→「6 篇」笔误；README.md
文档索引挂两本新手册。

**验收标准：**
- Given 索引更新后，When 链接存在性检查，Then 引用的仓内文件 100% 存在
- Given README 文档索引，When 检查，Then 新增两行且不破坏既有结构

### 3.7 FR7 质量收口

**描述：** AR 级质量收口与交付同步。

**触发条件：** T001-T006 全部完成后。

**期望行为：** 全量回归零回归（不触碰训练链路，新增仅 tools/infra/、tests/、docs/）；CI lint 范围三绿（tools/infra/ 已在 v0.6.0 纳入）；gitee slim-mirror 同步推送成功。

**异常处理：** 若出现回归，定位并修复后重新收口（不强行交付）。

**验收标准：**
- Given 全部任务完成，When 全量 pytest，Then 基线（683 passed / 2 skipped，2026-10-13 实测）零回归（新增测试全过或合规 skip）
- Given CI lint 范围（tools/infra/ 已含），When mypy/black/ruff，Then 三绿
- Given AR 完成，When gitee 同步，Then 镜像链推送成功

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 性能 | Triton 微实验全量运行 | ≤5 分钟（--quick ≤1 分钟） |
| 可测性 | 公式锁定 | 手册关键数值断言至少 1 条单测对应（参照实现） |
| 兼容性 | 不触碰训练链路 | 现有测试零回归；新增仅 tools/infra/、tests/、docs/ |
| 文档 | 中文 | 术语首现给英文；数值断言溯源 results/*.json 或 MIGRATION/EXPERIMENTS |

## 5. 约束与假设

**约束：**
- 本机 inductor 断裂（triton 3.8.0 vs torch 2.5.1，MIGRATION §8.1 已录）：compile
  实测限 dynamo 层；不做环境降级修复（会破坏手写 kernel 的 triton 3.8 依赖）
- triton `@jit` 要求 kernel 定义在 .py 源文件（inspect.getsource）——工具天然满足
- sm75（Quadro RTX 5000）无 TMA/新特性：kernel 用基础语言特性（program_id/arange/
  load/store），不追求极致性能（教学定位）
- gitee PNG 预算（~58KB）：图表优先表格/JSON
- 云端正在补跑实验：本 AR 全部内容本机可完成，不与云端冲突；E-F-4 SOP 面向后续窗口

**假设：**
- triton 3.8.0 手写 kernel 在本机的可用性已冒烟验证（2026-10-13，add_kernel allclose OK）
- ring_attention.py/flash_attention.py/attention_utils.py 的现有实现稳定（本 AR 不改它们，
  只作锚点）

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| online-softmax | 分块计算 softmax 时维护 running max/sum 状态的数值稳定递推算法（flash attention 核心） |
| tiling | 把 Q/K/V/O 分块装入 SRAM/cache 逐块计算的 kernel 组织方式 |
| grouped-broadcast | GQA 中把 Q reshape 出 KV-head 维、利用 SDPA singleton 广播替代 KV repeat 拷贝的零拷贝技巧 |
| KV 物化 | SDPA enable_gqa 路径在无原生 GQA 内核的后端内部展开 KV head 导致的显存放大（本仓实测 12,288·L B/token） |
| dynamo | torch.compile 的 FX 图捕获层（guard/重编译语义） |
| inductor | torch.compile 的默认编译后端（fusion 决策 + Triton codegen） |
| cudagraphs | CUDA 图录制回放（reduce-overhead 模式，消除 launch 开销） |
| capacity（MoE） | 分桶后每专家的最大 token 容量，超容量丢弃（static-shape grouped GEMM 的代价） |

## 附：需求澄清记录

- 2026-10-13 用户裁定：Triton 独立微实验不接入模型；两本手册分立；compile 分层实测
  （dynamo 本机 + inductor 理论/云端 SOP）不修环境；MoE grouped GEMM 教学入本 AR；
  Triton kernel 选型三个（SwiGLU+RMSNorm+GQA decode）；验收口径 A/B+一致性+JSON。
- 2026-10-13 缺口分析（三路子代理）：「代码在、教学缺」密度最高区域 + 约半数实测数据
  无手册承接 → 本 AR 同时闭合两类缺口。
