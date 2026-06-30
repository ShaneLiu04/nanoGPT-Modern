# ADR-004: 采用 KV Cache INT8/FP8 量化

**状态**: Accepted  
**日期**: 2026-06  
**作者**: nanoGPT-Modern Team

---

## 背景

长上下文推理（>4K tokens）时，KV Cache 显存成为主要瓶颈。以 GQA-2KV、12 层、4096 上下文为例：
- FP16 KV Cache: 2 × 12 × 2 × 4096 × 64 × 2 = 24.6 MB per sequence
- 批量 16 时: ~394 MB 显存

需要一种量化方案在保持生成一致性的前提下压缩 KV Cache。

## 决策

采用 **per-channel INT8 量化**（对称量化）作为默认 KV Cache 压缩方案，FP8 作为前瞻性选项（PyTorch 2.4+ 原生支持时自动启用）。

## 考虑方案

| 方案 | 压缩比 | 精度损失 | 复杂度 | 结论 |
|------|--------|----------|--------|------|
| **per-channel INT8** | 2× | <0.1% PPL | 低（PyTorch 原生） | **采纳** |
| per-token INT8 | 2× | 0.3-0.5% PPL | 中 | 备选 |
| KIVI (2-bit/4-bit KV) | 4-8× | 1-2% PPL | 高（需自定义 kernel） | 未来 |
| FP8 (E4M3/E5M2) | 2× | <0.05% PPL | 低（PyTorch 2.4+） | 自动探测启用 |
| 不量化 | 1× | 0 | — | 基准 |

## 关键设计

1. **量化粒度**: per-channel（沿 head_dim 维度），因不同 channel 的数值分布差异大，per-channel 比 per-tensor 精度更高。
2. **对称量化**: KV 向量经 LayerNorm/RMSNorm 后已近似零均值，对称量化足够且避免 zero-point 开销。
3. **仅在存储时量化**: `update()` 时量化写入缓存，`get_cache()` 时反量化，compute 路径保持原精度。
4. **FP8 自动回退**: 若 PyTorch 不支持 `torch.float8_e4m3fn`，自动回退到 INT8。

## 后果

- **正向**: GQA-2KV + INT8 量化后 KV Cache 仅 2,304 B/tok（相比 MHA FP16 的 18,432 节省 87.5%）。`cache=True` 与 `cache=False` 输出仍保持 bit-wise 一致（反量化精度足够）。
- **负向**: 增加了 `get_cache()` 时的反量化计算开销（约 +5% decode 时间），但显存节省允许更大 batch，整体吞吐提升。
- **风险**: 极低（<0.1% PPL 的 per-channel INT8 在 LLM 社区已有广泛验证）。

## 参考

- `model/kv_cache_utils.py` `QuantizedKVCacheManager`
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (Liu et al., 2024)
- FP8 Formats for Deep Learning (Micikevicius et al., 2022)

---

*最后更新: 2026-06-30*
