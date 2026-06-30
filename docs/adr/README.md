# 架构决策记录 (ADR) 索引

> 记录 nanoGPT-Modern 的关键架构决策、权衡分析与选择理由，帮助新成员理解设计背景，避免重复讨论已决议事项。

---

## 记录列表

| ADR | 标题 | 状态 | 日期 | 影响范围 |
|-----|------|------|------|----------|
| [ADR-001](adr-001-gqa-grouped-broadcast.md) | 采用 Grouped-Broadcast 零拷贝实现 GQA | Accepted | 2026-05 | `model/modern_gpt.py`, `tests/test_gqa_broadcast.py` |
| [ADR-002](adr-002-grpo-vs-ppo.md) | 采用 GRPO 而非 PPO 作为 RL 对齐算法 | Accepted | 2026-05 | `training/train_grpo.py`, `training/train_dpo.py` |
| [ADR-003](adr-003-pytorch-vs-jax.md) | 采用 PyTorch 而非 JAX 作为深度学习框架 | Accepted | 2026-05 | 全部 |
| [ADR-004](adr-004-kv-cache-quantization.md) | 采用 KV Cache INT8/FP8 量化 | Accepted | 2026-06 | `model/kv_cache_utils.py` |
| [ADR-005](adr-005-config-system.md) | 采用 YAML + argparse + 可选 Pydantic 的配置系统 | Accepted | 2026-06 | `utils/config.py`, `utils/config_models.py` |
| [ADR-006](adr-006-moe-grouped-gemm.md) | 采用 MoE grouped GEMM 优化替代 Python 循环 | Accepted | 2026-06 | `model/modern_gpt.py` |

---

## ADR 模板

新建 ADR 时，请复制以下模板并填写：

```markdown
# ADR-XXX: <标题>

**状态**: Proposed / Accepted / Deprecated / Superseded by ADR-YYY  
**日期**: YYYY-MM  
**作者**: <作者>

---

## 背景

<描述触发此决策的问题或上下文>

## 决策

<明确的决策陈述>

## 考虑方案

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| | | | |

## 后果

- **正向**: 
- **负向**: 
- **风险**: 

## 参考

- <相关文件/论文/链接>

---

*最后更新: YYYY-MM-DD*
```

---

*索引最后更新: 2026-06-30*
