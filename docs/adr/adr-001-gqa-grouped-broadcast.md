# ADR-001: 采用 Grouped-Broadcast 零拷贝实现 GQA

**状态**: Accepted  
**日期**: 2026-05  
**作者**: nanoGPT-Modern Team

---

## 背景

GQA（Grouped Query Attention）通过让多个 Query head 共享同一组 Key/Value head 来减少 KV Cache 显存。原始实现使用 `repeat_interleave` 将 KV head 显式复制到 Q head 数量，但这导致 KV 张量临时膨胀，**抵消了 GQA 的显存节省收益**。

## 决策

采用 grouped-broadcast 方式：将 Q reshape 为 `[B, n_kv_head, n_rep, T, head_dim]`，KV unsqueeze 为 `[B, n_kv_head, 1, S, head_dim]`，利用 PyTorch SDPA 的广播语义实现零 KV 拷贝。

## 考虑方案

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| repeat_interleave | 简单直观，所有后端兼容 | KV 显存临时膨胀 n_head/n_kv 倍 | 否决 |
| raw（不展开） | 零拷贝，显存最优 | FlashAttention 等后端不支持 | 仅 fallback |
| **grouped-broadcast** | 零拷贝，所有 SDPA 后端兼容 | 需要 reshape 操作 | **采纳** |

## 后果

- **正向**: GQA-2KV 显存从 18,432 B/tok 降至 4,608 B/tok（节省 75%），且生成输出与 repeat_interleave 方案 bit-wise 一致。
- **负向**: 增加了运行时探测逻辑（`probe_gqa_sdpa_support`），首次 forward 有微秒级开销。
- **风险**: 不同 PyTorch 版本的 SDPA 广播行为可能不一致，已通过 16 个测试用例覆盖。

## 参考

- `tests/test_gqa_broadcast.py`
- `model/modern_gpt.py` lines ~450-510

---

*最后更新: 2026-06-30*
