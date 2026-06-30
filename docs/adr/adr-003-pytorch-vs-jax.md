# ADR-003: 采用 PyTorch 而非 JAX 作为深度学习框架

**状态**: Accepted  
**日期**: 2026-05  
**作者**: nanoGPT-Modern Team

---

## 背景

项目需要选择一个深度学习框架作为基础。候选包括 PyTorch、JAX 和 TensorFlow。

## 决策

采用 PyTorch 2.0+ 作为唯一支持的深度学习框架。

## 考虑方案

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| **PyTorch** | 动态图调试友好，SDPA/torch.compile 原生支持，社区最大，部署路径清晰 | JIT 优化不如 JAX 彻底 | **采纳** |
| JAX | 函数式编程，pmap 并行简洁，XLA 编译优化强 | 调试门槛高，学习曲线陡，工业生态较弱 | 否决 |
| TensorFlow | 生产部署成熟 | 2.x API 混乱，社区流失严重 | 否决 |

## 后果

- **正向**: PyTorch 2.0+ 的 SDPA 自动选择最优融合 kernel（FlashAttention/Memory-Efficient/Math），`torch.compile` 支持生成阶段加速。团队无需学习新框架。
- **负向**: 在超大规模（>100B 参数）场景，PyTorch 的分布式通信效率可能不如 JAX 的 pmap/pjit。
- **缓解**: 预留了 JAX 实验分支的接口抽象（`model/` 的模块设计尽量 framework-agnostic），未来若需要可迁移核心算子。

## 参考

- `docs/TECH_REPORT.md` 第 4.3.1 节
- `model/attention_utils.py` SDPA 后端切换逻辑

---

*最后更新: 2026-06-30*
