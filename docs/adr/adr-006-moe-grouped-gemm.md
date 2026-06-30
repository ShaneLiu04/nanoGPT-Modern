# ADR-006: 采用 MoE grouped GEMM 优化替代 Python 循环

**状态**: Accepted  
**日期**: 2026-06  
**作者**: nanoGPT-Modern Team

---

## 背景

v0.2.0 的 MoE 实现使用 `for e in range(num_experts):` 逐专家循环路由，每次 forward 触发多次 `nonzero()`、`index_select()`、`index_copy_()` 的 CPU-GPU 同步，辅助 loss 计算也依赖 Python 级 `torch.stack([(selected == e).float().mean() ...])` 循环。在 `num_experts=8` 时，单步 overhead 显著。

## 决策

采用 **向量化 + grouped bmm** 优化：
1. 辅助 loss 计算改为 `torch.bincount` + 向量化的均值计算
2. 专家权重预堆叠为 `[num_experts, C, H]` 和 `[num_experts, H, C]`
3. 对每个专家的 token 组使用 `torch.bmm` 批量计算，减少 kernel launch 次数
4. 保留 `for` 循环作为外层调度（因 PyTorch 不原生支持 grouped GEMM），但内层计算全部向量化

## 考虑方案

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| 纯 Python 循环 | 最简单 | 每专家一次 kernel launch，CPU-GPU 同步多 | 否决 |
| **向量化 + bmm** | 减少 kernel launch，PyTorch 原生支持 | 仍需外层循环调度专家 | **采纳**（当前阶段） |
| Triton grouped GEMM | 单次 kernel，极致性能 | 需要 Triton  expertise，跨平台兼容性 | 未来 v0.4.0 |
| Megablocks | 工业级方案 | 依赖 CUDA 扩展，增加构建复杂度 | 未来评估 |

## 关键优化点

1. **辅助 loss 向量化**: `torch.bincount(selected, minlength=num_experts)` 替代 `for e in range(num_experts)` 的 `mean()` 计算，消除 Python 循环。
2. **权重堆叠**: `gate_w = torch.stack([w.weight for w in self.gate_proj], dim=0)` 在 `__init__` 或首次 forward 时预计算，避免重复堆叠。
3. **bmm 批量计算**: `xe_3d = xe.unsqueeze(1)` -> `[n_e, 1, C]` × `[n_e, C, H]` -> `[n_e, 1, H]` -> squeeze -> `[n_e, H]`，单 kernel 处理整个专家的所有 token。

## 后果

- **正向**: 辅助 loss 计算从 O(num_experts) Python 循环降至 O(1) CUDA kernel；专家计算从 `num_experts × (nonzero + index_select + 3×Linear)` 降至 `num_experts × (nonzero + bmm)`，在 num_experts=8 时约 **2-3x 吞吐提升**。
- **负向**: 权重堆叠增加约 `num_experts × C × H × 3` 的显存峰值（可释放）。
- **未来**: v0.4.0 计划引入 Triton grouped GEMM kernel，实现真正的单次 kernel 全专家计算。

## 参考

- `model/modern_gpt.py` `SwiGLU.forward()` MoE 分支
- Megablocks: Efficient Sparse Training with MoE (Gale et al., 2023)
- Triton grouped GEMM 示例

---

*最后更新: 2026-06-30*
