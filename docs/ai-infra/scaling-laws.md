# Scaling Laws 与训练预算深潜手册（Scaling Laws & Training Budget: Kaplan / Chinchilla / B_crit / WSD）

> 主题 9｜AR-v0.7.5 交付。配套微实验工具：`python tools/infra/fit_scaling.py`（纯 CPU，
> 产物 `docs/results/scaling_fit.json`）。本手册所有本仓数值均可溯源：budget.json /
> ablation-summary.json / EXPERIMENTS.md §6.1-6.2 / utils/lr_scheduler.py。

## 1. 问题形态：给定预算，怎么花

infra 岗 per-day 问题：**给我 X 的算力预算，模型多大、数据多少、训多久，loss 最低？**
拆开是三个子问题：

1. **规模问题**（本手册 §2-§3）：参数量 N 与数据量 D（tokens）怎么分预算？
2. **批次问题**（§4）：batch size 开多大还有收益？（B_crit）
3. **退火问题**（§5）：LR schedule 怎么设计，训练末期 loss 还能往下掉？

本仓是单卡教学预算（`docs/results/budget.json`），正好做**记账口径**的活教材——「你说的
数据量，到底指哪个数？」（§6 第一问）。

## 2. Kaplan 脉络（2020）：幂律与大模型样本效率

Kaplan et al.（Scaling Laws for Neural Language Models）的核心结论：

- **loss 对 N/D/C 各自呈幂律**：固定其他变量，L(N) = E + A·N^(-α)。E 是不可约损失
  （数据分布本身的熵），A/N^α 是可优化部分；实测 α ≈ 0.07-0.1（N 以参数计）；
- **大模型样本高效**：固定训练 token 预算，把参数量开大、每个参数见更少数据，loss 更低
  ——推论是「优先训大、不用训完」（compute-optimal 的 Kaplan 版：N 增长快于 D）；
- **无数据瓶颈假设**：Kaplan 拟合里 D 的约束松（部分实验固定 D 扫 N），隐含「数据管够」。

方法论要点：**单变量扫描 + 幂律拟合**。固定 D 扫 N 得 L(N)，固定 N 扫 D 得 L(D)，再
拼联合式。这个方法论的盲区（联合尺度行为没被约束住）正是 Chinchilla 反转的伏笔。

## 3. Chinchilla 脉络（2022）：compute-optimal 与 D/N≈20

Hoffmann et al.（Training Compute-Optimal LLMs）改用**联合视角**：给定算力 C ≈ 6ND
（每 token 训练前向+反向约 6N FLOPs，口径同 utils/infra_math.py），在 (N, D) 平面上
最小化 L(N,D) = E + A·N^(-α_N) + B·D^(-α_D)。三种估计方法（固定 C 扫 (N,D) 网格 /
 IsoFLOP 等算力曲线 / 拟合联合式）交叉验证，结论：

- **compute-optimal 条件：D/N ≈ 20 tokens/param**（实测 15-25 区间，规范值 20）；
- 参数与数据**同比例增长**（N ∝ C^0.5, D ∝ C^0.5），与 Kaplan 的「N 增长快于 D」相反；
- **反转的根因**：Kaplan 的扫参里 D 相对不足（早期实验模型小、数据量固定偏低），拟合
  出的 A/N^α 占比过强；Chinchilla 在每个 C 档位都给了 D 足够的伸展空间，才发现「小模型
  没喂饱数据」才是当时的常态。一句话：**结论差异是实验设计差异，不是谁的拟合错**。

实务含义（面试高频）：给预算 C，N* = sqrt(C/(6×20×k))… 直接用公式族
N* ∝ C^0.5、D* = 20N*；每 token 成本 6N；训完时长 = D / 吞吐。`fit_scaling.py` 的
预算计算器实现这套（文献默认 α_N=0.34/α_D=0.28 口径来自 Hoffmann 拟合值），并附本仓
拟合参数对照。

## 4. 临界 batch size（McCandlish 2018）：并行的免费午餐边界

**B_crit（critical batch size）**：由梯度噪声尺度决定——batch 开到 B_crit 之前，
吞吐提升几乎不伤样本效率（「免费并行」）；超过 B_crit，同样的算力花在「更多步」比
「更大 batch」更划算（大 batch 边际递减）。

- 数学直觉：梯度 ∇L 的 minibatch 估计带噪声；SGD 的进展同时取决于方向（噪声）和步数
  （大 batch 步数少）。噪声尺度 σ 定义了 B_crit，实测随训练推进**变大**（loss 降、
  梯度更一致），也随模型变大而变大；
- 经验量级：小模型（~50M）B_crit 通常在 10^5-10^6 tokens 区间，大模型更高；
- **本仓对照**（budget.json）：batch_size=32 × block_size=1024 = **32,768 tokens/iter**，
  对 50M 级模型大概率在 B_crit 以下的「免费并行」区——所以梯度累积（GradAccumulator）
  等工具把显存不够的批次拆开累积，语义上等价 batch 32，不失样本效率；
- 引申：为什么大模型训练流行 mega-batch（百万 token 级）+ 高 LR——因为 B_crit 大了，
  而不是「batch 越大越好」的普适律。

## 5. LR schedule：cosine 与 WSD 的退火语义

**共同结构**：warmup（线性升 LR，避免初期不稳）→ 主段 → 衰减段（末期精细化）。本仓
实现（`utils/lr_scheduler.py:14`，`LRScheduler`）：支持 `cosine` / `linear` / `wsd` /
`constant` 四式，warmup 与 decay 长度独立配置，`__call__(step)` 按 step 返回当前 LR。

| 式 | 语义 | 特点 |
| --- | --- | --- |
| cosine | 从 lr 余弦衰减到 min_lr（L81） | 平滑，末期降速快，最常用 |
| WSD（warmup-stable-decay） | 稳定段 LR 不动 + **短而陡**的线性衰减段（L20/L29：decay_iters 指「衰减段长度」） | 可中途暂停/续训（stable 段无进度概念），衰减段贡献「退火收益」——近年主流（MiniCPM/Llama 3 系实测同算力下与 cosine 打平甚至更优） |
| linear / constant | 线性降 / 不降 | 基线与消融用 |

**退火损失的直觉**：LR 高时参数在 loss 面上「跳跃搜索」，LR 降到很小才能落进精细极小值
——所以 loss 曲线的大幅下降集中在衰减段尾端（「annealing bump」）。这解释了两个实践
现象：①为什么 cosine/WSD 的最终 loss 对「最后 10% 训练」极度敏感；②WSD 为什么敢用
很短的衰减段——退火收益主要来自「LR 变小」本身而非衰减路径形状。

**本仓配置锚点**（budget.json）：`warmup_iters=2000 / max_iters=18000`（warmup 占
11%）、cosine 到 min_lr、seed=1337。

## 6. 本仓拟合复盘：一个 50M 模型的预算解剖

### 6.1 记账口径三件套——「数据量」指哪个数？

budget.json 的三个 token 数，**回答不同问题**：

| 口径 | 数值 | 算式 | 回答的问题 |
| --- | --- | --- | --- |
| 语料全量 corpus | 1,244,464,437 | 数据集总量（MIGRATION §3 判据表同源） | 语料库里有多少 token |
| 训练 split 一纪元 | ≈1,182,599,168 | one_epoch_iters 36,079 × 32,768 | 一个 epoch 吃多少（≈corpus×95% train split） |
| **实际消耗 consumed** | 589,824,000 | max_iters 18,000 × 32,768 | 这次训练真的喂了多少（**半个纪元**） |

教学点：谈 scaling 必须先钉口径。Chinchilla 的 D 指 consumed（实际训练 token 数），
不是语料量。

### 6.2 compute-optimal 对照：意外地贴线

用 consumed 口径算 tokens/param（现代全配 modern-full，params_m=59.561）：

- corpus 口径：1,244,464,437 / 59.56M ≈ **20.9 t/p** —— 与 Chinchilla 20 惊人地贴线
  （名义 50M 口径则是 24.9）；
- consumed 口径：589,824,000 / 59.56M ≈ **9.9 t/p** —— 实际训练只喂了 compute-optimal
  的一半数据。

结论（诚实版）：本仓的**语料配比**是按 compute-optimal 准的；**单卡算力**只够跑半个
纪元（max_iters < one_epoch_iters），属于「数据配好了、算力没花完」的教学预算——
所以消融对照实验之间的可比性建立在「所有组同预算」上（这正是 §6.3 拟合的约束，
也是它的软肋）。

### 6.3 七点 Kaplan 拟合：教学拟合，不是 scaling study

`docs/results/ablation-summary.json` 7 组（同 budget、不同架构消融）：

| 组 | params_m | best_val_loss |
| --- | --- | --- |
| baseline | 64.086 | 3.6118 |
| modern-full | 59.561 | 3.5730 |
| no-rope | 60.085 | 3.6258 |
| no-swiglu | 58.775 | 3.5593 |
| mha | 64.280 | 3.5397 |
| moe | 137.442（total）/ 59.561（active） | 3.5824 |
| qk-norm | 59.562 | 3.5558 |

`fit_scaling.py` 对这 7 点拟合 L(N)=E+A·N^(-α)，**三个必须诚实标注的局限**：

1. **D 恒定**：7 组同预算训练，L(N,D) 联合式的 B·D^(-β) 项不可辨识（吸收进常数）——
   只能拟合 Kaplan 式 L(N)，Chinchilla 联合式在工具里用合成数据演示；
2. **N 与架构混淆**：组间差异主要来自架构消融（rope/swiglu/mha/moe），params_m 的差异
   是消融的副作用而非受控扫参——拟合出的 α 不具备跨架构外推力；
3. **7 个点**：任何幂律用 7 个点拟合，参数不确定度都很大——工具报告协方差传播的
   参数标准差（fit_quality.param_uncertainty），教学价值在「拟合方法与诚实报告」，
   不在参数值本身。

### 6.4 与工具互证

本节数字与 `docs/results/scaling_fit.json` 一一对应：`repo_budget_review.bookkeeping`
= §6.1 三件套；`tokens_per_param` = §6.2 两口径；`kaplan_fit.params_m/val_loss/fitted`
= §6.3 表；`fit_quality.limitations` = §6.3 三条。外推预测（predictions）与预算计算器
（budget_calculator，含按本仓吞吐口径换算的训练时长）的算例解读见 EXPERIMENTS 记录节。

## 7. 能说 / 不能说

**能说**：
- Kaplan vs Chinchilla 的结论差异及根因（固定变量扫参 vs 等算力联合扫参）；
- compute-optimal 的操作化：N ∝ C^0.5、D ≈ 20N、每 token 6N FLOPs；
- B_crit 的语义（免费并行边界）与本仓 batch 32k tokens 的位置判断（大概率低于 B_crit）；
- WSD/cosine 的退火收益机理与本仓 lr_scheduler 四式实现；
- 本仓预算的诚实解剖：语料配比贴 compute-optimal（20.9 t/p），算力只够半个纪元。

**不能说**：
- 不能把 7 点消融拟合出的 α 当作本仓架构的 scaling 预测参数（架构混淆+点太少）；
- 不能说 D/N≈20 是普适律——它是 Hoffmann 数据集/架构/精度的条件最优，混合精度、
  MoE、多轮 epoch（数据重复）都会移动这个比值（后续 Muennighoff 的 data-constrained
  scaling 给出重复 token 的边际衰减）；
- 不能从「corpus 口径 20.9 t/p 贴线」推出本仓训练是 compute-optimal 的——实际消耗
  只有 9.9 t/p，单卡预算是欠训练的。

## 8. 自测要点

1. Kaplan 和 Chinchilla 对「预算怎么分」给出的答案差在哪？根因是什么？
2. 6ND 口径：6 哪来的？（答：前向 2N + 反向 4N FLOPs/token）
3. 本仓 batch 32×1024=32k tokens，B_crit 之下意味着什么？梯度累积为什么不亏？
4. WSD 的 D 指什么？为什么敢比 cosine 的衰减段短？
5. budget.json 三个 token 数（1.24B/1.18B/0.59B）各回答什么问题？Chinchilla 的 D 是哪个？
6. 运行 `python tools/infra/fit_scaling.py`，对照 scaling_fit.json：7 点拟合的 α 与
   参数不确定度是多少？敢用它外推 10×模型吗？（答：不敢，说出三条局限）
