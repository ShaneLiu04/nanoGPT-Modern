# [AR-v0.4.0-experiments] 需求设计说明书

| 字段 | 内容 |
|------|------|
| AR 编号 | AR-v0.4.0-experiments |
| AR 主题 | 实验证据化与出版级可视化平台（Evidence-Grade Experimentation & Visualization） |
| 关联 SR | 无（版本演进需求，源自 CHANGELOG P0「50M 参数完整训练验证」+ README 架构增益声称缺数据支撑） |
| 日期 | 2026-10-07 |
| 状态 | Approved（req 门控 PASS，2026-10-07） |

---

## 1. 背景与目标

v0.3.0 交付了覆盖预训练/对齐/推理/评估的全栈框架，但**实验证据存在系统性缺口**：

1. **实验停留快速验证档**：现有实验全部为 3.3M 参数 + 合成随机数据（val loss ≈ 10.84 仅为随机熵，无信息量；算术 accuracy=0），`out/` 目录为空，实验结果未沉淀进仓库。
2. **架构增益声称零数据支撑**：README 声称 GQA -75% KV cache、RoPE+NTK 外推、SwiGLU/MoE 增益，无任何真实数据图/表可引用。
3. **实验结果不可机器读取**：`trainer_base.py` 无结构化 metrics 输出（仅 console/TensorBoard），无法自动聚合成图。
4. **图片不可复现**：docs 仅 4 张静态 PNG，无生成代码；无任何实验数据图。
5. **监控未接入**：`utils/monitoring.py`（LossSpike/GradNorm）从未被 trainer 导入。
6. **消融粒度不足**：`run_ablations.py` 只支持 cache on/off。

**目标**：v0.4.0 将框架从「能跑」升级为「有证据」——补齐结构化实验记录、组件级消融、50M 完整规模真实实验与出版级图表流水线，使 README 的每一条架构声称都有可复现的数据与图表支撑。

## 2. 需求范围

**In Scope（本 AR 要做的）：**

- 框架：结构化实验记录（metrics JSONL + experiment manifest）
- 框架：monitoring 接入 BaseTrainer
- 框架：ModernGPT 消融架构开关（`rope=off` 回退可学习绝对 PE；`ffn_variant=gelu4x`）
- 框架：组件消融编排器（7 组矩阵、断点续跑、汇总）
- 实验：全量 OpenWebText 数据准备（~9B tokens + 全局 shuffle 索引）
- 实验：7 组 50M 完整规模（18k iters）预训练对照/消融
- 实验：全套推理实验（KV cache 显存/吞吐矩阵、投机解码、连续批处理、torch.compile）
- 实验：长上下文（NTK ppl-vs-长度曲线、Needle-in-Haystack heatmap）
- 实验：对齐全链路（SFT → GRPO G4/G8 → DPO 胜率）在 50M checkpoint 上
- 可视化：出版级图表生成器（中文标注、统一科技风、SVG+PNG）
- 文档：`docs/FIGURES.md` 图库、EXPERIMENTS.md 全量重写、README 嵌入真实图表

**Out of Scope（本 AR 不做的）：**

- 3D 并行（TP/PP）、QAT、剪枝、SSM/Mamba 混合（CHANGELOG P1-P3，需多卡 A100）
- 100-200M 模型扩展
- vLLM/TGI 推理后端、Web UI、边缘部署
- 现有 4 张 PNG 示意图（fig1-4）的重制
- 框架对外 API/协议变更

## 3. 功能需求

### 3.1 FR1 结构化实验记录

**描述：** Trainer 侧逐 step 记录训练指标为机器可读文件，并生成实验清单。

**触发条件：** 任意训练脚本（pretrain/SFT/GRPO/DPO/iterative GRPO）启动时。

**期望行为：**
- `out/experiments/<exp_id>/metrics.jsonl`：每行一个 JSON 事件（step、loss、val_loss、lr、grad_norm、tokens/s、显存峰值、监控事件如 loss_spike）
- `out/experiments/<exp_id>/manifest.json`：exp_id、完整超参、git commit、环境（GPU 型号/VRAM、PyTorch 版本、CUDA、sm 架构）、seed、时间戳、产物 checkpoint 指针
- TensorBoard/wandb 现有日志行为不变

**异常处理：** metrics 写入失败不中断训练（降级为 console 警告）；manifest 在训练结束与每次 checkpoint 时双写。

**验收标准：**
- Given 任一训练脚本跑 ≥50 步，When 训练完成，Then `metrics.jsonl` 存在且每行可被 `json.loads` 解析、含 ≥50 条记录，`manifest.json` 含环境与超参字段
- Given `metrics.jsonl`，When 用一行 pandas/纯 Python 读取，Then 可无损还原 loss 曲线与监控事件

### 3.2 FR2 监控接入 BaseTrainer

**描述：** 将 `utils/monitoring.py` 的 LossSpikeDetector / GradientNormMonitor 接入 `trainer_base.py` 训练循环。

**触发条件：** 训练循环每 step（grad_norm）/每 eval（loss spike 检测）。

**期望行为：** 梯度范数逐 step 进入 metrics 流；loss spike（>3σ）事件写入 metrics 并按现有策略触发 LR 自动降低；开关默认开启、可 CLI 关闭。

**异常处理：** 监控组件异常不中断训练。

**验收标准：**
- Given 固定种子训练，When 开启监控跑 100 步，Then metrics.jsonl 中出现 `grad_norm` 字段；构造 spike（如中途注入大 LR）时出现 `loss_spike` 事件
- Given 既有全部测试，When 运行回归，Then 全部通过（不破坏现有行为）

### 3.3 FR3 消融架构开关补齐

**描述：** ModernGPT 新增两组架构开关，使组件消融可行。

**触发条件：** CLI `--rope off` / `--ffn_variant gelu4x`（或 YAML 等价字段）。

**期望行为：**
- `rope=off`：位置编码回退为可学习绝对 embedding（GPT-2 风格），其余现代组件保持；参数量变化在 config 报告
- `ffn_variant=gelu4x`：SwiGLU(8d/3) 替换为 GELU(4×) FFN
- 两开关进 CLI / YAML / Hydra / Pydantic 校验（互斥与取值枚举）
- KV cache 路径与生成路径在 `rope=off` 下正确工作（绝对 PE 无 start_pos 偏移问题）

**异常处理：** 非法组合（如 `rope=off` 且 `rope_scaling=ntk`）被 Pydantic 校验拒绝并给出明确错误。

**验收标准：**
- Given `rope=off` 配置，When 前向/生成（含 KV cache），Then 输出形状与数值有限，cache/no-cache 输出一致
- Given `ffn_variant=gelu4x`，When 参数量统计，Then FFN 参数符合 GELU(4×) 公式
- Given 非法组合配置，When Pydantic 校验，Then 报错并含人可读提示

### 3.4 FR4 组件消融编排器

**描述：** `run_component_ablations.py` 编排 7 组消融矩阵并自动汇总。

**触发条件：** `python run_component_ablations.py --stage all|train|summarize --exp ...`。

**期望行为：** 7 组配置（baseline / modern-full / no-rope / no-swiglu / mha / moe / qk-norm）按序训练（每组 50M/18k iters）；已完成的组跳过（断点续跑）；全部完成后输出汇总表（final val loss、参数量、tokens/s、显存峰值、相对 baseline 变化%）为 JSON+Markdown。

**异常处理：** 单组训练崩溃记录错误并继续下一组（除非 `--fail_fast`）。

**验收标准：**
- Given 7 组中 2 组已有完整 metrics/manifest，When 重跑编排器，Then 仅执行缺失 5 组
- Given 全部完成，When 汇总，Then 生成含全部组、全部指标列的汇总表

### 3.5 FR5 全量 OpenWebText 数据准备

**描述：** 下载并准备全量 OpenWebText 训练/验证 bin + 全局 shuffle 索引。

**验收标准：**
- Given `data/openwebtext/train.bin|val.bin` + `.shuffle.idx`，When `data/validate.py`，Then 通过完整性检查
- Given 7 组消融，When 消耗 tokens（7 × 221M ≈ 1.55B < 数据量），Then epoch 重复率为 0；若实测数据量不足导致重复，Then 重复率 <5% 且写入各组 manifest 的 `epoch_repetition` 字段

### 3.6 FR6+E7 预训练主对照与组件消融（7 组 × 50M × 18k iters）

**描述：** 在真实 OpenWebText 上完成 7 组完整训练，构成项目核心证据。

**验收标准：**
- Given 7 组全部完成，When 读取各组 metrics.jsonl，Then 每组含 18k 步完整记录与 ≥18 个 val 点
- 汇总表可回答：「Modern-full 相对 Baseline 的 val loss 变化」「各组件的独立贡献」

### 3.7 FR8 全套推理实验

**描述：** 在 50M Modern-full checkpoint 上量化推理子系统。

**验收标准：**
- Given 50M modern-full checkpoint，When 运行 KV cache 基准（{无缓存, ring fp16, INT8, paged} × {256, 1024, 4096} tokens），Then 产出 12 组「显存占用 + prefill/decode 吞吐」记录的结构化 JSON
- Given 同一 checkpoint 与 draft 配置，When 运行投机解码基准（batch=1 与 batch>1 各 ≥3 组采样），Then JSON 含接受率与端到端加速比（vs 无投机基线）
- Given 并发 {1,2,4,8,16} 五档请求，When 运行连续批处理基准，Then JSON 含各档吞吐（tok/s）与平均延迟（ms）
- Given torch.compile on/off 各一次推理基准，Then JSON 含加速比记录

### 3.8 FR9 长上下文实验

**描述：** NTK 外推与 Needle-in-Haystack。

**验收标准：**
- Given modern-full checkpoint（训练长度 1024），When 分别以基础 RoPE 与 NTK 缩放测 {1024, 2048, 4096, 8192} 四档上下文 ppl，Then 产出两条 ppl-vs-长度曲线数据（JSON，各 ≥4 点）
- Given Needle 评估网格（深度 5 档 × 长度 4 档 = 20 格，每格 ≥3 trials），When 运行 NeedleEvaluator，Then 产出每格检索成功率矩阵（JSON）

### 3.9 FR10 对齐全链路实验

**描述：** SFT → GRPO(G4/G8) → DPO 在 50M checkpoint 上的完整对齐实验。

**验收标准（指标口径：easy/medium/hard 各 ≥100 样本；格式遵从率 = 含合法 `<answer>` 标签的比例；accuracy = 数值正确（相对误差 <1e-6）的比例）：**
- Given modern-full checkpoint，When SFT 训练完成并在三档算术集评估，Then metrics/JSON 含各档 format_rate 与 accuracy
- Given SFT checkpoint，When 分别以 G4/G8 运行 GRPO，Then 两组 metrics.jsonl 含逐 eval 的 mean_reward 曲线，且对齐评估产出 reward 提升 Δ 与 KL（vs SFT reference）
- Given GRPO-G8 checkpoint，When 运行 DPO + 胜率评估（≥100 对比样本），Then JSON 含 policy vs SFT reference 的胜率（%）与 DPO/IPO/KTO 损失曲线

### 3.10 FR11 出版级图表生成器

**描述：** `docs/figures/` 下统一风格、可复现的图表生成脚本集。

**期望行为：** 统一样式模块（配色/字体/中文标注/网格）；每图一个生成命令，输入为 `out/experiments/` 数据，输出 SVG+PNG 双格式到 `docs/figures/out/`；无数据时给出明确错误。

**验收标准：**
- Given 全部实验数据，When 逐一运行生成脚本，Then 每个脚本 0 退出码、产出同名 .svg 与 .png
- 图表覆盖：训练 loss 曲线（7 组）、消融汇总条形图、KV cache 显存阶梯/吞吐、投机解码接受率、连续批处理曲线、NTK ppl 衰减、Needle heatmap、GRPO reward 曲线、DPO 胜率、（可选）GradNorm/监控事件图

### 3.11 FR12 图库与文档嵌入

**描述：** `docs/FIGURES.md` 图库（每图配「看什么/怎么读/说明了什么」学习性讲解）；EXPERIMENTS.md 全量重写为 v0.4.0 真实数据版；README 重写实验段落嵌入真实图表。

**验收标准：**
- Given `out/experiments/` 全部实验数据，When 逐一运行图表生成脚本，Then 每个脚本 0 退出码、产出同名 .svg 与 .png
- Given 全部生成图，When 构建 `docs/FIGURES.md`，Then 每图收录且配讲解与生成命令
- Given README 与 EXPERIMENTS.md，When 重写完成，Then 实验章节引用真实数据与图表路径，且不残留「预期/目标」措辞表述已验证事实
- Given EXPERIMENTS.md，When 检查，Then 含环境表、复现命令、全部结果表

### 3.12 FR13 质量保障

**验收标准：**
- Given FR1-FR4 新增测试，When 运行 `pytest tests/ -q`，Then 新增测试全部通过
- Given 既有 230+ 测试与 `mypy`/`black`/`ruff`，When 全量运行，Then 全部通过/保持绿色

## 4. 非功能需求

| 类型 | 指标 | 要求 |
|------|------|------|
| 可靠性 | 长时实验 | 全流程断点续跑；单点失败不污染已完成实验数据 |
| 可复现性 | 实验 | manifest 记录 git commit + 种子 + 环境；图表可由数据一键重生成 |
| 资源 | GPU 显存 | 50M 模型 batch=12 × 1024 训练在 16GB 内完成（必要时梯度检查点） |
| 诚实性 | 报告 | 环境限制（sm_75 无 flash/bf16）如实标注；参数量差异列明 |

## 5. 约束与假设

**约束：**
- 单卡 Quadro RTX 5000 16GB（Turing sm_75）：SDPA 走 mem_efficient/math 后端、无 bf16 tensor core（AMP fp16）；实验环境如实标注
- 总 GPU 时长估 25-40h，跨多个会话执行，依赖 SDD 状态机断点恢复
- 磁盘：OpenWebText raw + bin 估 20-25GB
- 消融公平性：同 12L/8H/512D 骨架、同 seed=1337、同数据顺序；组件导致的参数量差异在报告中列明
- 新增依赖：matplotlib（seaborn 可选）入 `plot` extras；补装 tiktoken

**假设：**
- 本机网络可访问 HuggingFace datasets（OpenWebText 下载 2-5h）
- 50M 模型 18k iters 单组训练 2-4h 估算成立（若实测显著超时，降级方案：主对照 2 组保持 18k，5 组消融降至可完成时长并在文档说明——需用户确认后执行）
- CUDA 12.x + PyTorch 2.x 环境与现有 CI 兼容

## 6. 术语说明

| 术语 | 定义 |
|------|------|
| exp_id | 实验唯一标识（如 `ablation-no-rope`），对应 `out/experiments/<exp_id>/` |
| manifest | 实验清单：超参+环境+种子+产物指针的 JSON |
| metrics.jsonl | 逐事件 JSON Lines 训练记录 |
| 消融组 | 7 组架构对照配置之一 |
| NTK | RoPE 的 NTK-aware 长度外推缩放 |
| Needle-in-Haystack | 长上下文检索压力测试（深度×长度网格） |
