# ADR-002: 采用 GRPO 而非 PPO 作为 RL 对齐算法

**状态**: Accepted  
**日期**: 2026-05  
**作者**: nanoGPT-Modern Team

---

## 背景

PPO（Proximal Policy Optimization）需要单独的 Value/Critic 网络，在小模型（50M 参数）场景下：
1. Critic 网络占用额外显存和参数量
2. Critic 训练不稳定（value estimation 偏差大）
3. 实现复杂度显著高于策略梯度方法

## 决策

采用 GRPO（Group Relative Policy Optimization）：通过组内相对奖励计算优势函数，完全去除 Critic 网络。

## 考虑方案

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| PPO + Critic | 成熟，社区资料丰富 | 需要额外模型，显存开销大，小模型 Critic 不稳定 | 否决 |
| REINFORCE + baseline | 极简，无 Critic | 方差大，收敛慢 | 否决 |
| **GRPO** | 无 Critic，组内归一化降低方差，实现简洁 | 组大小影响方差/偏差 trade-off | **采纳** |
| DPO (offline) | 无需在线采样，稳定 | 依赖偏好数据质量，探索能力弱 | SFT 后补充 |

## 后果

- **正向**: GRPO 在 50M 参数模型上稳定收敛，显存节省约 30%（无需 Critic）。三层批量化优化后 GPU 利用率从 5-20% 提升至 60-80%。
- **负向**: 组大小（group_size）需要调参：过小方差大，过大偏差大（均值偏离真实优势）。
- **补充**: 提供 DPO/IPO/KTO 作为离线偏好学习补充，形成 "GRPO 探索 + DPO 精炼" 的组合策略。

## 参考

- `training/train_grpo.py`
- `training/train_dpo.py`
- DeepSeekMath: Advancing Mathematical Reasoning of LLMs via RL (Shao et al., 2024)

---

*最后更新: 2026-06-30*
