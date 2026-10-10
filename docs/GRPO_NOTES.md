# GRPO_NOTES.md — RLHF 家族对比笔记

> AR-v0.5.0-hpc-readiness FR14 交付物。三份训练代码（`train_grpo.py` / `iterative_grpo.py` /
> `train_dpo.py`）的算法差异心智模型，按三个维度组织：**优势计算**、**参考模型更新**、**KL 处理**。
> 对照代码阅读（FR14-AC3：描述准确）。
>
> 相关：[ARCHITECTURE.md](ARCHITECTURE.md)（训练基座）· [adr/adr-002-grpo-vs-ppo.md](adr/adr-002-grpo-vs-ppo.md)（GRPO vs PPO 决策）

## 1. 家族全景与继承关系

```
BaseTrainer                       (trainer_base.py，模板方法 + 钩子)
  ├─ GRPOTrainer                  (train_grpo.py，在线 RL + 规则奖励)
  │    └─ IterativeGRPOTrainer    (iterative_grpo.py，ref 更新 + 拒绝采样 SFT)
  └─ DPOTrainer                   (train_dpo.py，离线偏好优化)
```

入口 shim：`train_grpo_hydra.py` / `iterative_grpo_hydra.py` 仅是 `@hydra.main` 薄壳，
import 同一训练器类，不含算法逻辑。

## 2. GRPOTrainer：在线组相对策略优化

**数据流**（`train()` 每步）：
`sample_group`（G 个响应/prompt，`policy.eval()+no_grad` 记录 old_logprobs，
ref 模型同步记 ref_logprobs）→ `compute_grpo_loss`（train 模式一次性重算 new_logprobs）。

**关键不变量**（类 docstring 原文语义）：old_logprobs 在 `eval()+no_grad`（dropout=0）下记录、
new_logprobs 在 `train()` 下重算；PPO ratio 无偏要求 **模型 dropout=0.0**
（`maybe_warn_dropout` 守护；mask 差异会直接偏置 ratio）。

### 维度一：优势计算（`compute_advantages`，group/batch/none 三模式）

| 模式 | 公式 | 语义 |
|---|---|---|
| `group`（默认，原版 GRPO） | `(r - mean_G) / std_G`，逐 prompt | 组内相对：每个 prompt 的 G 个响应互为基线 |
| `batch` | 全 `[G,B]` 批归一 | 小 group_size 时更稳 |
| `none` | `r - mean_G`（只中心化） | 不除 std |

可选 `adv_clip` 对优势做对称截断。优势进入 PPO 式 clipped surrogate：
`-min(ratio·A, clamp(ratio,1±ε)·A)`。

### 维度二：参考模型更新

**GRPOTrainer：参考模型冻结**。`_build_model` 同时构建 policy 与 ref（同构），
ref 恒 `eval()` + `requires_grad=False`，整个训练期间不更新。

### 维度三：KL 处理

损失中加 **反向 KL(ref‖policy) = ref_logp - new_logp**（数值稳定形式，`utils/rl_utils.py`
的 `compute_kl_divergence`），系数 `--beta`，token 级 mask 后按 token 平均。
KL 既是正则项也产出监控指标 `kl_loss`。

## 3. IterativeGRPOTrainer：迭代式 GRPO

继承 GRPOTrainer 全部算法，通过两个钩子注入迭代行为：

### 维度一：优势计算

与父类完全一致（复用 `compute_advantages` / `compute_grpo_loss`）。

### 维度二：参考模型更新（核心差异）

`update_reference_model(ratio)`（`on_rollout_end` 钩子周期触发）：

- **hard 更新**（ratio=1.0）：`ref ← policy` 直接覆盖；
- **soft/EMA 更新**（0<ratio<1）：`ref ← ratio·policy + (1-ratio)·ref`。

参考模型从「冻结锚点」变成「慢速移动的信任域」，允许策略持续偏离初始模型。

### 维度三：KL 处理

损失形式同父类，但 KL 的参照点随 ref 更新而移动（相对当前 ref 的信任域），
而非相对初始 SFT 模型的固定距离。

**额外的拒绝采样 SFT**（`rejection_sampling_sft`，`on_eval_end` 钩子）：
对每个 prompt 采 N 个响应 → 规则奖励排序 → 仅保留 top-K 且 reward>0 →
构造 `ArithmeticDataset` 做几步 SFT（低 LR）。这是「RL 探索 + 监督精炼」的混合策略，
与 `best_policy_state`（评估最优策略快照）配合防止策略退化。

## 4. DPOTrainer：离线偏好优化

不走在线 rollout：数据是 `(chosen, rejected)` 偏好对（`--preference_source=synthetic` 或
`grpo`——后者从 GRPO checkpoint 的高/低奖励响应自动构造，见 ARCHITECTURE.md §6）。

### 维度一：优势计算（隐式）

没有显式优势项。偏好信号编码在损失结构里：
`-logsigmoid(β·[(π_c-π_r) - (ref_c-ref_r)])`（chosen 与 rejected 的
**相对 logprob 提升**相对参考模型做差）。β（`--beta`，默认 0.1）是温度。
支持三种变体（`--preference_loss`）：`dpo`（label smoothing 可选）/ `ipo` / `kto`。

### 维度二：参考模型更新

**冻结**（同 GRPOTrainer）。ref_model 仅提供 logprob 基线，
DPO 的理论保证正建立在「固定参考」上——这是它与 Iterative GRPO 的本质分歧：
DPO 用 Bradley-Terry 推导消掉了显式 RL，代价是必须锁死参考。

### 维度三：KL 处理（隐式）

**没有显式 KL 项**。DPO 的推导（BT 模型 + KL 约束优化的闭式解）把
「贴近参考模型」折叠进损失的 `(ref_c-ref_r)` 差分里：偏离参考不会单独受罚，
只有在伤害偏好排序时才受罚。β 越小越接近纯偏好、越大越保守。

## 5. 一页对比

| | GRPOTrainer | IterativeGRPOTrainer | DPOTrainer |
|---|---|---|---|
| 范式 | 在线 RL（规则奖励） | 在线 RL + 迭代 | 离线偏好学习 |
| advantage | 组内相对（group/batch/none） | 同 GRPO | 隐式（chosen-rejected 差分） |
| 参考更新 | 冻结 | **周期 hard/EMA 更新** | 冻结（理论要求） |
| KL | 显式反向 KL(ref‖policy)，β 加权 | 同左（参照点随 ref 移动） | 隐式（折叠进损失差分，β 温控） |
| 采样成本 | 每步 G 个 rollout | 同左 + 拒绝采样批次 | 无 rollout（数据先行） |
| dropout 不变量 | 必须 0.0 | 必须 0.0 | 不涉及 ratio，无此约束 |
| 何时选它 | 无标注偏好、有规则奖励 | 漂移信任域 + 防退化 | 已有偏好对（或 GRPO 产物） |

**组合用法**（ADR-002）：GRPO 探索 → 其 checkpoint 经 `_build_preference_dataset_from_grpo`
转偏好对 → DPO 精炼。三种算法共享 BaseTrainer 的 AMP/分布式/检查点/监控设施
（见 [ARCHITECTURE.md](ARCHITECTURE.md)）。
