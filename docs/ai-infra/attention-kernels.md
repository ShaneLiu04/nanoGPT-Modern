# Attention Kernel 深潜手册（Attention Kernels: SDPA / GQA / Flash / MoE）

> 目标：合上手册前，你能讲清楚 attention 从「N² 显存爆炸」到「分块 + 在线 softmax + 零拷贝广播」的完整演化链，并能把每个推导步骤对到本仓的具体代码行。
> 方法：每节「问题 → 推导 → 本仓验证」三段式；代码锚点全部可点开对照；实测数字溯源 [EXPERIMENTS.md](../EXPERIMENTS.md) / [MIGRATION.md](../MIGRATION.md) / `docs/results/*.json`。
> 姊妹篇：[显存与算力心算手册](memory-flops-arithmetic.md)（roofline 与算术强度）、[torch.compile 编译栈手册](compile-stack.md)（编译层）。

本仓标准模型：12 层 / 8 头 / 512 维 / GQA-2（n_kv_head=2，head_dim=64）。下文未注明处均指该配置。

---

## 1. 为什么 attention 是 kernel 战场

训练时一层注意力的朴素实现要物化 **[B, H, T, T]** 分数矩阵：block=1024 时单个分数矩阵 = 1×8×1024×1024×4 B = 32 MiB，反向还要存它——T 翻倍，显存×4。这就是「计算量 O(T²) 不是问题、显存 O(T²) 才是问题」的经典场景，也是 FlashAttention 的出发点（§4）。

decode（T=1 增量解码）则是另一个极端：每步只有一个 query，计算量极小，瓶颈完全在**把 KV cache 从 HBM 搬进计算单元的带宽**。本仓实测（`docs/results/roofline.json`）：

| 场景 | 实测 | 含义 |
|------|------|------|
| 大 GEMM（compute-bound） | 8.47 → 9,825.7 GFLOPs/s，**1160×** 提升 | 算力墙离屋顶还很远时，优化空间在并行度 |
| GEMV（memory-bound） | 349.2 GB/s，达规格带宽 77.9% | 每搬 1 字节能做多少 FLOPs 决定上限 |
| decode 单步 | 4.81 GB/s 有效带宽 | 大量时间花在 kernel 启动与小张量调度上 |

**教学点**：训练态 attention 优化「省显存」（分块），decode 态优化「省搬运」（零拷贝 + 减少隐式复制）。§3-§5 分别对应这两条战线。

---

## 2. SDPA 后端分层与 mask 的代价

### 2.1 四个后端

`F.scaled_dot_product_attention`（SDPA）是 PyTorch 的 attention 分发器，内部按输入特征选后端：

| 后端 | 特征 | 何时用 |
|------|------|--------|
| flash | 不物化 [B,H,T,T]，分块 + 在线 softmax | fp16/bf16、sm80+、无复杂 mask |
| memory-efficient (xFormers) | 分块，但允许更多 mask/dtype 组合 | fp32、老架构（本机 sm75 的实际主力） |
| math | 朴素全物化实现 | 兜底，任何输入都能算 |
| cuDNN | 黑盒融合实现 | sm80+ 部分版本 |

**本仓验证**：`model/attention_utils.py:104` 的 `sdpa_supports_enable_gqa()` 用**签名检查**（`inspect.signature`，L114-118）判定 torch≥2.5 的 `enable_gqa` 参数是否存在——刻意做成静态探测而非首次前向试探，因为**惰性探测会打断 torch.compile 图捕获**（L109-112 注释）。`probe_gqa_sdpa_support()`（L127-234）则进一步在真实 device/dtype 上探测两种零拷贝形态，结果按 `(device, dtype)` 缓存（L154-156）且永不抛异常——探测失败返回 `(False, False)` 让调用方安全回退（L227-230）。

### 2.2 `is_causal` 折叠 vs 加性 mask 的代价

- `is_causal=True` 是一个**布尔**：后端知道「下三角可见」，可以把它折进 kernel 的分块逻辑里，零额外显存——flash/memory-efficient 都能吃。
- 加性浮点 mask（`attn_mask=[B,1,T,T]`）必须**物化成真实张量**传进去，且复杂 mask 会让 flash 后端直接拒收、退到 memory-efficient 甚至 math。

**本仓验证**（`model/modern_gpt.py:659-682`）：`CausalSelfAttention.forward` 的选择逻辑是——

```python
# L661：有 cache 时 q 位置恒 >= k 位置，因果性自动满足，连 is_causal 都不用传
is_causal = past_kv is None and T > 1 and document_ids is None and not window_active
# L664 起：只有滑动窗口 / 文档边界 / padding 这类"非规则"约束才构建加性 mask
```

三层降级：① 干净训练 → `is_causal=True`（最便宜）；② decode + cache → 什么都不传（L661 折叠掉）；③ 滑动窗口/跨文档 → 物化加性 mask（L665-682，`[B,1,T,T]` 或 `[1,1,T,T]`）。对应测试：`tests/test_attention_mask_causal_fold.py`。

---

## 3. GQA：KV 膨胀的两种解法

**问题**：GQA 让 n_kv_head（如 2）< n_head（如 8），KV cache 缩小 4 倍。但 SDPA 的数学定义要求 Q/K/V 头数一致——最直接的补齐是 `k.repeat_interleave(n_rep, dim=1)`，**它把省下的显存又复制了回去**（EXPERIMENTS §3.2：GQA-2 的 KV 是 4,608 B/tok，比 MHA 的 18,432 省 75%；repeat_interleave 会当场翻回 18,432）。

本仓有两条零拷贝路径，对应两个时机：

### 3.1 训练/整段前向：grouped-broadcast（`_gqa_grouped_sdpa`，modern_gpt.py:177-242）

思路：不改 KV，改 **Q 的形状**，让广播发生在真实维度上：

```
Q: [B, n_head, T, D] --view--> [B, n_kv, rep, T, D]   # L222，纯视图
K: [B, n_kv, S, D]   --unsqueeze(2)--> [B, n_kv, 1, S, D]  # L223-224，零拷贝
SDPA 对 singleton 维 rep 原生广播 → 输出 reshape 回 [B, n_head, T, D]  # L241-242
```

为什么它比 `repeat_interleave` 好：`expand` 视图不分配内存，且「头数不匹配」变成了「正常的 5 维广播」，**每一个 SDPA 后端都接受**（L199-201 注释：flash / memory-efficient / math / cuDNN 全测过）。mask 也要跟着升维：传入 `[B,1,X,Y]` 时补 `unsqueeze(1)` 成 `[B,1,1,X,Y]` 保持可广播（L229-231）。

数值上与 `repeat_interleave` **逐位相同**（`tests/test_gqa_broadcast.py`）。探测函数 `probe_gqa_sdpa_support()` 返回的第二个布尔就是这个形态可用性；`_probe_sdpa_gqa_support`（L163-174）是弃用的旧探测，保留只为兼容。

### 3.2 单 token 解码：原生分辨率 scores（`_decode_gqa_attention`，modern_gpt.py:245-297）

T==1 时连 SDPA 都不必用——手动两步批量矩阵乘，scores 直接在 **KV 头原生分辨率**上算：

```
q: [B, n_head, 1, D] --view--> [B, n_kv, rep, D]        # L290
att = softmax((q_g @ kᵀ) * scale)   # [B, n_kv, rep, S]  # L293-294
y = att @ v                         # [B, n_kv, rep, D]  # L296
```

关键收益：scores 矩阵是 `[B, n_kv, rep, S]` 而非展开后的 `[B, n_head, S]`——**比任何后端回退路径都小 2×rep 倍**，且完全不依赖后端行为（§5 会看到这个「不依赖」救了一条 decode 链路）。`enable_gqa=True` 的原生路径（modern_gpt.py:389/770 处按探测结果启用）在后端没有真 GQA 内核时会**内部物化**头扩展——这正是 §5 的 bug 来源。

**本仓验证**：`tests/test_decode_gqa_attention.py`（8 用例）：与 SDPA allclose、与 repeat_interleave 参照逐位一致、cache 视图与 value-form 输入位相同、T>1 拒绝等。

---

## 4. FlashAttention：在线 softmax 推导 ↔ 代码逐行对照

**朴素 softmax 的问题**：`softmax(x)_i = exp(x_i)/Σexp(x_j)`。分块计算时块内 `Σexp` 只是全和的一部分，直接除会算错；且不平移最大值时 `exp` 会溢出。FlashAttention 的答案是**维护三个 running 状态**，把整个序列的分块扫完再归一：

### 4.1 三变量递推（数学）

设第 j 块的原始分数 `s_ij ∈ R^{T_q × T_k}`（已乘 scale），running 状态为行向量 `m`（至今最大值）、`l`（至今 exp 和）、`acc`（至今加权值和）：

```
m_new   = max(m, rowmax(s_ij))                       # 1. 更新最大值
p_ij    = exp(s_ij − m_new)                          # 2. 用新最大值平移（永远 ≤ 1，不溢出）
scale_c = exp(m − m_new)                             # 3. 旧累加的修正因子（≤ 1）
l_new   = l · scale_c + rowsum(p_ij)                 # 4. 旧和缩放 + 新块贡献
acc_new = acc · scale_c + p_ij @ v_j                 # 5. 同样的缩放作用于值累加
m       = m_new
最终输出 = acc / l                                    # 6. 全序列扫完后才做除法
```

**为什么正确**：朴素 softmax 等价于对 `exp(s − m_global)` 归一；递推版只是把 `m_global` 换成逐步逼近的 `m`，每一步都对旧累加补乘 `exp(m_old − m_new)`，扫完全序列时 `m = m_global`、`l = Σexp(s−m_global)`，与朴素定义严格相等。**为什么省显存**：任何时刻只驻留一块 `s_ij`，[B,H,T,T] 从不存在。

### 4.2 ↔ `model/ring_attention.py` 逐行对照

本仓 `blockwise_attention` 是这一算法的纯 PyTorch 参照（无需 flash-attn/Triton），推导六步与代码逐行对应：

| 推导步骤 | 代码（ring_attention.py） | 说明 |
|----------|--------------------------|------|
| 分块循环 | L101-119 双层 for：外层 query 块 i，内层 kv 块 j | `m/lse/acc` 在 L106-117 按块初始化 |
| s_ij 计算 | L128 `torch.matmul(q_i, k_jᵀ) * softmax_scale` | 含因果/padding 掩码 L132-143 |
| 1. m_new | L145-146 `m_ij = max(s_ij, -1)`; `m_new = maximum(m, m_ij)` | 行最大，keepdim |
| 2/3. 平移与修正 | L150-152 `m_new_safe = where(isneginf(m_new), 0, m_new)`; `exp_scale = exp(m − m_new_safe)`; `exp_ij = exp(s_ij − m_new_safe)` | **L150 安全处理**：整行被掩成 −inf 时 `exp(−inf−(−inf))=NaN`，替 0 防染毒 |
| 4. l 更新 | L154 `lse = lse * exp_scale + exp_ij.sum(-1, keepdim=True)` | 与推导 4 完全一致 |
| 5. acc 更新 | L155 `acc = acc * exp_scale + exp_ij @ v_j` | 与推导 5 完全一致 |
| 6. 最终除法 | L161 `out[...] = acc / lse` | **L160 安全处理**：全被掩的 padding 行 l=0 会除零，替 1 让输出保持 0 |
| 序列尾裁剪 | L163-164 | padding 的 query 行直接切掉 |

掩码两处：因果 `k_pos <= q_pos`（L132-136）；**KV padding 位置即使非因果也绝不可见**（L139-140 `k_pos < Tkv`）——分块把序列补齐到块大小整数倍，补位必须显式排除，这是逐位一致测试能过的关键细节。GQA 在此参照中按块局部 `repeat_interleave`（L124-126）——参照实现优先可读性，生产路径用 §3 的零拷贝。

**本仓验证**：`tests/test_ring_attention.py`——与 SDPA 全序列对照 allclose、因果/非因果、GQA 头数、padding 边界。`model/flash_attention.py:31-274` 则是可选 `flash-attn` 包的缓存感知包装（`flash_attention_with_cache`，见 modern_gpt.py:641-657 的调用与 `None` 回退）。

### 4.3 能说 / 不能说

- 能说：在线 softmax 的**数学**与任何实现（flash-attn / Triton / 纯 PyTorch）一致，本仓参照与 SDPA 逐位对齐。
- 不能说：性能。参照实现有 Python 层循环开销，不是 FlashAttention 的性能等价物；sm75 上 fp32 也进不了 flash 后端（§2.1）。

---

## 5. KV cache 与 decode 显存：一次 12,335 B/token 的完整复盘

### 5.1 零拷贝 append-cache 设计

本仓 decode 的 KV cache 设计目标：**cache buffer 预分配一次，每步 decode 零新分配**。`CausalSelfAttention.forward`（modern_gpt.py:455-487）识别 `(KVCacheManager, layer_idx)` 形态的 `past_kv`，条件齐备时走快速路径：

```python
# L473-483：T==1、无外部 mask/document/positions、无滑动窗口、未驱逐、layout=="append"
kv_view = cache_obj.get_layer_view(cache_layer_idx, extra=1)   # 零拷贝视图，预留 1 个新位置
```

条件不满足则回退该层的 value-form（L487），行为不变。这是「零拷贝视图」一半；另一半是消费端不复制——§3.2 的 `_decode_gqa_attention` 直接吃视图。

### 5.2 bug：SDPA 后端把零拷贝视图又复制了

**测量**（MIGRATION §8.2，24L/16H/8KV/1024d fp32，prompt=8，decode 至 512/1024/2048/4096）：预分配 buffer 之上仍测到峰值显存**斜率 12,335 B/token**（理论 KV 写入量 98,304 的 1/8），R²=0.99999 完美线性——每步都在产生随上下文线性增长的新分配。

**定位**（`torch.cuda.memory._record_memory_history` 分配栈追踪）：栈帧直指 `F.scaled_dot_product_attention(..., enable_gqa=True)`——sm75 没有 flash GQA 内核，**memory-efficient 后端在内部物化 8→16 的 KV 头扩展**，每层每步 ~`[B,16,S,64]` fp32 ×3 份。12,288 B/token ≈ 3×4,096×24 精确吻合。设计好的零拷贝视图被 SDPA 内部复制击败；grouped 5D 广播同样物化（~2×kv-native）。

**修复**：T==1 的 decode 改走 §3.2 手写内核（`_decode_gqa_attention`），绕开后端。

**效果**：斜率 12,335 → **−1,090 B/token（平坦）**；单步瞬时分配 57.7 MB → 0.5 MB（L=4096，**114×**）；GPU 吞吐 68 → 64.5 ms/step（+5%——物化在 GPU 上只是带宽拷贝，收益主要在**显存容量**：长上下文/大 batch 不再 OOM）。

### 5.3 教学点

1. **零拷贝是全链路性质**：cache 给了视图，kernel 又复制一遍，优化等于没做。profilers（`_record_memory_history`）是唯一可靠裁判。
2. **`enable_gqa` 不是性能开关**：它只声明「头数不同」，后端有没有原生 GQA 内核决定它是零拷贝还是隐形物化——探测（§2.1）+ 显式 kernel（§3.2）才是完整答案。
3. **修复的验收要包含负样本**：`tests/test_decode_gqa_attention.py` 的 8 用例里有「cache 视图 vs value-form 输入位相同」——保证修复本身没引入新复制。

---

## 6. MoE grouped GEMM：从逐专家循环到静态形状

**问题**：E 个专家的 FFN，朴素写法是 Python 循环逐专家 `F.silu(x_e @ w_gateᵀ) * (x_e @ w_upᵀ) @ w_downᵀ`——E 次 kernel launch + 动态形状（每个专家分到的 token 数不同），launch 开销随 E 线性放大。

本仓实现（`model/modern_gpt.py:988-1046`）四步：

| 步骤 | 代码 | 教学点 |
|------|------|--------|
| ① 路由 | L994-999 `router → softmax → topk(1)` | top-1 稀疏：每 token 只走 1 个专家 |
| ② 负载均衡 aux loss | L1004-1007 `f_i` = 实际分到专家 i 的 token 占比（`bincount`），`P_i` = router 平均概率，`aux = α·E·Σ(f_i·P_i)` | 均衡时 f=P=1/E，loss=α；全偏斜时 f·P 集中到被偏爱的专家，loss 变大——可微的「反偏爱」信号，向量化无 Python 循环 |
| ③ 向量化 dispatch | L1018-1022 `sort(selected, stable=True)` + `bincount`/`cumsum` 得 `seg_starts`，`slots = arange − seg_starts[sorted]`，`keep = slots < capacity` | stable sort 让同专家 token 保持原序；`capacity = int(cf·total/E)`（L1012）截断超载 token（Switch Transformer 风格，丢弃即输出 0） |
| ④ padded grouped GEMM | L1031-1036 静态缓冲 `padded [E, capacity, C]`，scatter 写入（L1032），三个批量 GEMM 一次算完 | **静态形状** `[E, cap, C]` 与实际 token 分布无关——kernel 数从 3E 降到 3，形状固定利于编译器与调度（见 compile-stack.md §3） |
| ⑤ gather 回原序 | L1040-1042 `out_flat[t_idx] = de[e_idx, s_idx] * router_probs[t_idx, e_idx]` | 按路由概率加权，**梯度流向 router**（直通 top-1 的不可微选择，概率乘法补回可微性）；padding 行恒零，scatter 天然跳过 |

**本仓验证**（`docs/results/moe_kernel.json`，4096 token，2/4/8/16 专家）：

| 专家数 | loop（逐专家） | grouped（padded） | 加速 |
|--------|---------------|------------------|------|
| 2 | 2.038 ms | 1.979 ms | ≈1.0× |
| 4 | 3.432 ms | 1.889 ms | 1.8× |
| 8 | 6.424 ms | 1.862 ms | 3.4× |
| 16 | **11.391 ms** | **1.978 ms** | **5.8×** |

loop 随 E 线性恶化（每次 launch ~0.6-0.7 ms 的固定成本×3 算子），grouped 几乎平坦（E 只改变静态缓冲的 E 维，GEMM 总量不变）。**E=2 时两者打平**——grouped 的收益来自消灭 launch 次数，专家太少时无 launch 可省，反而多付 padding 计算。对应测试：`tests/test_moe_*.py`。

---

## 7. Ring Attention 的真实形态与边界

**本仓是什么**：`ring_attention.py` 是**单卡** blockwise 参照——把长序列切块串行算，用 §4 的在线 softmax 保证与全序列数学一致。它省的是 [B,H,T,T] 显存，不省时间（甚至更慢）。

**真 Ring Attention 是什么**：多卡场景下，序列沿卡切分，每张卡只持有自己那段的 Q/K/V；K/V 块沿卡间环形拓扑**轮流传递**（P2P copy 与本卡计算重叠），每传一轮算一次块间 online-softmax 更新。序列长度上限从「单卡显存」变成「卡数×单卡显存」，代价是 E 轮通信。

**共同内核**：两者共享 §4 的三变量递推——这正是本仓参照的教学价值：把递推写对，单卡版和 256 卡版只是「块从哪来」的区别。

**不能说**：本仓没有多卡 Ring 数据（单卡 DDP/FSDP 在 MIGRATION §8.3-8.4 验证过，但那是数据并行不是序列并行）；多卡曲线、通信重叠效率等结论一律不外推。

---

## 8. 自测要点

对应测试（全绿是手册正确性的机器证明）：

| 测试 | 覆盖 |
|------|------|
| `tests/test_gqa_broadcast.py` | §3.1 grouped-broadcast 与 repeat_interleave 逐位一致、掩码升维 |
| `tests/test_decode_gqa_attention.py` | §3.2 与 SDPA allclose、cache 视图零拷贝性质、T>1 拒绝（8 用例） |
| `tests/test_attention_utils.py` | §2.1 探测逻辑、缓存、永不抛异常 |
| `tests/test_attention_mask_causal_fold.py` | §2.2 is_causal 折叠与加性 mask 等价性 |
| `tests/test_ring_attention.py` | §4 在线 softmax 与 SDPA 逐位对照、掩码/padding/GQA |

三个思考题（答案都在正文锚点里）：

1. `ring_attention.py:150` 为什么用 `m_new_safe = where(isneginf(m_new), 0, m_new)` 而不是直接 `exp(m − m_new)`？——提示：全掩行的 `−inf − (−inf)`。
2. `_gqa_grouped_sdpa` 改的是 Q 的形状而不是 KV 的形状，为什么方向不能反过来？——提示：KV 是要省的对象（§3），repeat 任何一侧都会物化。
3. MoE 的 `capacity` 截断丢弃了 token，为什么训练还能收敛？——提示：L1041 的加权 gather 与 ② 的 aux loss 各自补回了什么。

---

*维护约定：代码行号锚点以当前 main 为准；若重构移动了行号，请同步更新本手册的对照表（`grep -n` 可定位）。数据引用请保持溯源到 `docs/results/*.json` 或 MIGRATION/EXPERIMENTS 原文。*
