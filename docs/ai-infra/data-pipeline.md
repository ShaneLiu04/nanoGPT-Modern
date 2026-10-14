# 数据管道与 Tokenization 全链路深潜手册（Data Pipeline: Download → Tokenize → Dedup → Filter → Pack → Shard/Resume）

> 主题 10｜AR-v0.7.5 交付。配套微实验工具：`python tools/infra/data_probe.py`（纯 CPU，
> 产物 `docs/results/data_probe.json`）。本手册逐环节接 `data/` 代码锚点，判据数字溯源
> MIGRATION.md §3。

## 1. 全旅程鸟瞰

token 从语料到 loss 曲线的全旅程：

```text
OpenWebText parquet（云端）
  → curl range 并行分片下载 / 流式 tokenize（§2）
  → BPE encode（§3）              ── 文本 → uint16 token 流
  → MinHash/LSH 去重（§4）        ── 文档级近似去重
  → 组合过滤链（§5）              ── 长度/重复度/质量
  → 贪心 packing（§6）            ── 短文档拼进 block_size，doc_boundary 记边界
  → memmap shard/resume（§7）     ── 零拷贝读取、rank 跨步分片、断点续训
  → DataLoader → loss 曲线
```

本仓这套管道的完整实测：**8,926,903,359 tokens / 7,913,597 docs / 平均 1,128 tokens/doc**
（MIGRATION §3 判据表；EOT 频率 1/1128 ≈ 8.87e-4）。本机 `data/openwebtext/train.bin`
= 17,853,806,718 B（uint16），`data_probe.py` 对它做 memmap 抽样复测（§8）。

## 2. 获取与流式：分片下载与 1M 子集陷阱

**分片下载**（`tools/data/prepare_openwebtext_shards.py`）：parquet 文件按 HTTP range
分片并行拉取——大数据集的常规工程化（单连接吞吐有上限，分片才吃满带宽）。

**流式 tokenize**：分片边下载边 encode，token 流式追加落盘，**不在内存攒整个数据集**
——「流式」的本质是内存常量化的管道。

**1M 默认子集陷阱**（`data/prepare.py:31` `resolve_max_docs`）：默认只取 1M 文档的
prefix 子集。这是刻意的设计（教学预算 + 复现速度），但也是个陷阱——**你以为在训
OpenWebText 全量，实际是它的 1M prefix**。任何「全量」主张必须回查 resolve_max_docs
与实际落盘 doc 数（MIGRATION §3 的 7,913,597 docs 就是实测落盘口径）。

## 3. Tokenize：BPE 与 dtype 的两个工程决策

**编码**：GPT-2 BPE（tiktoken `encode_ordinary_batch`）。vocab=50,257 的构成：
256 byte 基元 + 50,000 merge + 1 个 `<|endoftext|>`（EOT=50256，同时是文档分隔符与
packing 的 padding 值）。

**vocab 大小的权衡**（50k vs 100k+）：

| 维度 | vocab 小（50k） | vocab 大（100k+） |
| --- | --- | --- |
| embedding 参数 | 少（50k×d） | 多（100k×d，d=768 时多 38M 参数） |
| 序列效率 | 低（英文 ≈4 char/token） | 高（常见词整词成 token，同文本 token 数少） |
| 适用 | 小模型（参数预算敏感） | 大模型（序列效率换取的算力节省超过 embedding 成本） |

经验律： vocab 与模型规模同涨（GPT-2 50k → Llama 3 128k）；小模型用大 vocab 是把
参数预算浪费在低频 token 上。

**dtype 自动选择**（`data/prepare.py:307` `_select_dtype`）：token 上限 ≤65,535 →
uint16（2 B/token，本仓 train.bin 17GB 的由来）；超过 → uint32（4 B/token）。
判断式很简单：`vocab_size < 65536`。读取侧 `_detect_dtype_from_index` 从 .idx 元数据
恢复 dtype，memmap 零拷贝直读（§7）。

## 4. 去重：MinHash → LSH → Jaccard 三级流水

精确两两比对的成本是 O(n²)——790 万文档不可行。三级近似流水（`data/dedup.py`）：

**第一级 MinHash 签名（L31）**：把文档变成 char 5-gram 集合，用 128 个独立哈希函数各取
最小值，得到 128 维签名。数学根底：**P(minhash(a) = minhash(b)) = Jaccard(a,b)**——
两个文档的 n-gram 集合越像，签名任意一位相等的概率越大。签名相等位数 ≈ Jaccard 的
无偏估计，文档→128 个 int，比价成本骤降。

**第二级 LSH 桶（L69）**：把 128 维签名切成 **8 band × 16 row**，每个 band 内 16 位
全部相同才落同桶。概率魔术：Jaccard=s 的文档对至少一 band 碰撞的概率是
1-(1-s^16)^8——s=0.8 时 ≈0.92，s=0.3 时 ≈1e-5，**高相似对进候选、低相似对滤掉**，
候选对数量从 n² 降到近线性。`query()` 返回候选集，只对候选做第三级。

**第三级 Jaccard 判定（L117）**：候选对回到原始 n-gram 集合算精确 Jaccard
（`_jaccard`：交集/并集；双空集约定 1.0），过阈值判重。

`data_probe.py` 的 `minhash_dedup_curve` 实验直接调用这套生产实现扫重复率：构造
不同重复率的合成 corpus，验证检出率曲线单调上升、独立文档误杀率≈0。

## 5. 过滤：组合链与统计

`data/filter.py` 的 `CompositeFilter`（`stats` L202 汇总各过滤器通过率）：长度过滤
（过短文档多为噪声）、重复度过滤（文档内 n-gram 重复率，广告/模板页特征）、正则
规则、fastText 分类器（可选，质量分）。组合链的设计要点是**每个过滤器独立可关**、
stats 逐项落盘——过滤是有损操作，哪一级杀了多少文档必须可审计（MIGRATION §3 判据表
就是这条审计链的产物）。

`data/quality_filter.py`：熵分（低熵文本=模板/列表，高熵=自然语言）与分层采样——
质量分 vs 数量的权衡是数据侧的核心 tradeoff：质量阈值抬高，单文档平均信息量上升但
总量缩水；Chinchilla 视角下「20 t/p 的数据配额」配什么质量的数据，直接决定同样的
N/D 换多少 loss（数据质量是 scaling 拟合里的隐藏变量）。

## 6. Packing：短文档拼接与 doc_boundary

OpenWebText 平均 1,128 tokens/doc，block_size=1024——**多数文档塞不满一个训练序列**。
朴素解法每文档一个 block + padding：浪费 = 1 - 1128/1024 区间的碎片。packing 的解法
（`data/openwebtext.py:326` `PackingDataset`）：

```python
# _build_samples (L349)：贪心打包
doc_starts = [0]                      # 扫描 EOT 定文档边界（1M token 分块扫，防大 memmap 慢）
# ...
seq_tokens.append(...)                # 文档依次入当前序列，装不下就 flush
def flush_seq():                      # (L388)
    pad = block_size - L
    x = seq_tokens + [eot] * pad      # 尾块用 EOT padding
    y = x[1:] + [eot]
    doc_ids = seq_doc_ids + [-1] * pad
```

三个关键语义（对照 `tests/test_packing.py` 6 用例）：

1. **返回 (x, y, document_ids) 三元组**：`document_ids` 逐 token 标记所属文档——attention
   用它构建跨文档 mask（文档 A 的 token 看不见文档 B 的 token），「packing 效率」与
   「跨文档污染」的矛盾由此解掉；
2. **y=-1 掩码**（docstring L334）：新文档首 token 的预测位 y 置 -1——该位置没有
   in-document 上下文，loss 不应计算。这是「文档边界处不求 loss」的实现；
3. **贪心的取舍**：贪心装包不是最优 bin-packing（NP-hard），但对长尾分布的网页文本
   填充率轻松 >95%（`data_probe.py` 的 `greedy_packing` 实验实测合成分布的填充率与
   padding 浪费）。

## 7. Shard 与 resume：memmap 零拷贝、跨步分片、rank-local 偏移

**MemmapDataset（L110）**：`np.memmap(bin_path, mode="r")` 直读 17GB 文件——虚拟内存
按页加载，不整读进 RAM；block 切片 `data[i*(B+1):(i+1)*(B+1)]` 零拷贝。样本长度
`len(data) // (block_size+1)`：+1 是 x/y 错位的重叠位（Karpathy 式切片）。

**rank 跨步分片（L159 `_rank_positions`）**：多卡训练时 rank r 消费全局块位置
r, r+W, r+2W, ...（W=world size）——**互斥且全域覆盖**（无 padding 无丢弃，docstring
L116-117 的契约）。除法分片（r 拿连续一段）会因各 rank 末尾长度不齐引入同步空转；
跨步分片把不齐摊到每个 rank。

**resume_offset 是 rank-local（L118）**：断点续训的偏移计数「本 rank 消费了多少样本」，
不是全局位置——各 rank 快慢不同，全局偏移无法对齐恢复。checkpoint 里存
`{"resume_offset": n}`（state_dict L165），恢复时各 rank 自己跳过前 n 个。

**GlobalShuffledDataset（L485）**：全局置换表（可落盘复用的 shuffle_idx，MemmapDataset
L145-157 加载时校验长度防漂移）——为什么不用 shuffle_buffer：buffer 只保证局部乱序，
全局置换保证 epoch 级均匀；置换表落盘则**断点续训后顺序可复现**（seed 一致时同序列）。
`get_dataloader`（L544）把上述组装成 worker 分片安全的 DataLoader（worker_info 分割
迭代域，防多 worker 重复读同一段）。

## 8. 本仓实测锚点：判据表与 data_probe 复测

MIGRATION §3 判据表（管道端到端的验收数字）：

| 判据 | 数值 |
| --- | --- |
| 总 tokens | 8,926,903,359 |
| 总 docs | 7,913,597 |
| 平均 doc 长度 | 1,128 tokens |
| EOT 频率 | 1/1128 ≈ **8.87e-4** |
| train.bin（本机） | 17,853,806,718 B = 8,926,903,359 × 2 B（uint16 零开销核对） |
| dtype/词表范围 | uint16，值域 [0, 50256] |

`data_probe.py` 的复测实验（真实剖析 source=real）：等步长跳跃抽样 train.bin
（memmap 零拷贝，不吃满页缓存）→ EOT 频率应 ≈8.87e-4 量级、top-token 直方图（GPT-2
BPE 下高频 token 是 the/and/, 等）、doc 长度分布 p50/p99。**同一统计量两种口径**
（全量判据表 vs 抽样估计）量级一致即验收——抽样估计有置信区间，不要求逐位相等。

## 9. 数据质量边界：污染检测与质量-数量权衡

**污染检测**：eval 基准的文本混进训练集会让 benchmark 虚高。工程做法：把基准样本的
n-gram 当 needle，在训练语料里检索命中（`data_probe.py` 的 `contamination_demo` 用
合成 needle 注入演示：注入的序列 100% 检出、未注入的 0% 误报——真实语料上则按
「命中 n-gram 数 > 阈值」判定）。严格版用 suffix array / exact-dedup 索引，教学版
理解「检测=检索+阈值」即可。

**质量 vs 数量**：过滤阈值每收紧一档，语料缩水一截，但 scaling 的 D 是「喂进去的
token 数」——过滤后要不要补数据补到 20 t/p？这是数据策略的核心循环：
**过滤 → 统计 → 按预算回填 → 复测**。本仓的答案在教学预算下从简（组合链 +
stats 审计），生产管道里质量-数量曲线本身是要做实验拟合的对象。

## 10. 能说 / 不能说

**能说**：
- MinHash 的数学根底（P(最小哈希相等)=Jaccard）与 LSH 的概率门控（band 碰撞率对 s 的
  阶跃形状）；
- packing 三件套：贪心拼接、doc_boundary 跨文档 mask、y=-1 边界掩码——效率与污染
  的一体两面；
- 跨步分片为什么互斥全覆盖、resume_offset 为什么必须 rank-local；
- 本仓管道端到端的验收数字（8.93B tokens/7.91M docs/EOT 8.87e-4）与抽样复测口径。

**不能说**：
- 不能把抽样统计（data_probe 抽 20 万 token）说成全量判据——量级一致是验收，逐位
  相等不是（抽样有方差）；
- 不能说贪心 packing 填充率 100%——尾块永远有 EOT padding（flush_seq 的 pad）；
- MinHash/LSH 是**近似**去重：高相似但 band 全不碰的文档对会漏检（概率小非零），
  阈值选择是 recall/precision 的旋钮，没有免费的确定性问题。

## 11. 自测要点

1. MinHash 签名第 i 位为什么能估计 Jaccard？8 band×16 row 对 s=0.5 的碰撞概率是多少？
   （≈1-(1-0.5^16)^8≈1.2e-4）
2. PackingDataset 返回三元组各是什么？y=-1 出现在哪些位置、为什么？
3. 跨步分片（stride W）与除法分片（连续段）比，优点是什么？resume_offset 为什么不能
   是全局位置？
4. uint16 的 token 上限是多少？什么 vocab 规模必须升 uint32？
5. 运行 `python tools/infra/data_probe.py`：实测 EOT 频率与 8.87e-4 差多少？这个差在
   抽样方差内吗？
6. 判据表的 8,926,903,359 tokens 与 train.bin 的 17,853,806,718 B 怎么互相核对？
