# MoE 系统工程手册（Mixture of Experts: 路由 / 容量 / 负载 / 专家并行）

> 目标：合上手册前，你能把 MoE 拆成三笔账——**容量账**（总参数 2.3 倍于激活参数，买到什么）、**系统账**（路由/溢出/负载的工程代价）、**通信账**（专家并行的 all-to-all）——并对本仓 E-A-5「未赢」的实测给出四因子归因，而不是一句「MoE 没用」。
> 方法：每节「问题 → 机制 → 本仓验证」；理论节（专家并行）无代码无实验，明标「理论档」；实测溯源 [EXPERIMENTS.md](../EXPERIMENTS.md) §6.2/§6.6/§6.11 与 `docs/results/{ablation-summary,moe_kernel,moe_probe}.json`。
> 姊妹篇：[Attention Kernel 手册](attention-kernels.md) §6（grouped GEMM kernel 视角）、[显存与算力心算手册](memory-flops-arithmetic.md)（参数量分解）、[长上下文手册](long-context.md)（另一条稀疏化路线）。

本仓标准 MoE 配置：E-A-5 消融为 **top-1、E=4**（总参 137.4M / 激活 59.6M）；容量/负载探针（E-C-6）用 **E=8、合成路由器**（探针原生环境）。下文注明处均指该配置。

---

## 0. 一页全景：MoE 用「参数-计算解耦」换什么

稠密模型：参数量 = 计算量，一个 token 过全部参数。MoE：路由器为每个 token 选 1 个专家（top-1），**总参数 ∝ E，每 token 计算量 ≈ 稠密单份**。本仓 E-A-5 的实测口径：

| 账 | 数值 | 来源 |
|----|------|------|
| 容量账（总参数） | 137.4M（稠密 modern 59.6M 的 2.3 倍） | `ablation-summary.json` |
| 计算账（激活参数） | 59.6M（与 modern-full 相同） | 同上 |
| 结果账 | val loss 3.5824 vs 稠密 3.5593（最优）/ 3.5730（modern-full）——**没有赢** | EXPERIMENTS §6.2 |
| 系统账 | 吞吐 146k→115k tok/s（−21%），显存 26.3→29.3 GB | 同上 |

这张表是本手册的「问题」：容量翻倍没有兑现成更低的 loss，还付了系统代价。§1 讲机制，§2 用微实验把溢出/负载量化，§4 给四因子归因。

---

## 1. 机制深化：top-1 路由 + aux 负载均衡损失 + capacity（Switch Transformer 语义）

全部锚点在 `model/modern_gpt.py` `SwiGLU.forward` 的 MoE 分支（L1040-1098）。

### 1.1 top-1 路由（L1046-1051）

`router_logits = self.router(x_flat)`（E 维线性层）→ softmax → `topk(1)`。每个 token 精确路由到一个专家。Switch Transformer 选 top-1 的理由：路由简单（无组合爆炸）、容量分析容易（每专家负载 = 选中数）、训练稳定。

### 1.2 负载均衡 aux 损失（L1053-1059）

```python
f = bincount(selected) / (B*T)      # 硬分布：每专家实际分到的 token 比例
P = router_probs.mean(dim=0)        # 软分布：平均路由概率
aux = α · E · (f * P).sum()         # α = moe_aux_loss_factor = 0.01
```

**乘积形式是关键**：只有「实际分到很多 token 且路由概率也高」才会贡献大罚项——某专家被硬路由喂饱但概率低（f 高 P 低）或反之，罚项都不大。手算两个极端：

- 完全均匀：f_i = P_i = 1/E → `E·Σ(1/E²) = 1`，aux = **α**（这是 aux 的下限，均匀即免罚）；
- 一半专家被打满（f = [2/E,...,0]，P 同步）：`E·Σf·P = E·(E/2)·(4/E²) = 2`，aux = **2α**——偏斜直接抬罚项。

这就是 §2 偏斜注入实验压的那条曲线。`test_moe_load_balance.py` 5 用例锁定：稠密 SwiGLU aux 恒零 / MoE 返回 aux / **aux 可微**（梯度流向 router）/ capacity 丢 token / config 序列化往返。

### 1.3 capacity 溢出（L1061-1094）

`capacity = int(cf · B·T / E)`（L1064，**int 截断**与 Switch 一致）。超容量的 token 被丢弃——输出行保持零（L1083 静态形状 `[E, capacity, C]` 的 padded 缓冲只写入 keep 的 token，L1092-1094 gather back）。

**keep 的语义**（L1070-1074）：`torch.sort(selected, stable=True)` 按专家分组且组内保持 token 原序，slot < capacity **保留每专家的前 capacity 个 token**。工程含义：溢出不是随机丢，是「排在后面的丢」——对 decode（T=1 逐 token）几乎无影响，对 prefill 是确定性的前缀偏好。

### 1.4 checkpoint 兼容（L989-1025）

专家权重堆叠成单参数（`gate_w/up_w: [E, H, C]`、`down_w: [E, C, H]`），但 state_dict 走**legacy 逐专家键格式**（`gate_proj.{e}.weight`），load/save 双向翻译钩子保证 T012 重写前后 checkpoint 字节兼容——重构不破坏已训模型的工程纪律。

---

## 2. 容量/负载微实验解读（E-C-6，`tools/infra/moe_probe.py`）

探针在纯 CPU 上实例化真实 `SwiGLU` 前向（E=8、T_total=256、合成路由器），`dispatch_stats` 镜像函数与 forward 的 keep 集合**逐 token 恒等**（契约测试锁定，三档 cf 全部 `dropped_rows_zero` 交叉验证通过）。数据（`docs/results/moe_probe.json`，EXPERIMENTS §6.11）：

| 实验 | 实测 | 解读 |
|------|------|------|
| cf=0.5 | dropped 128 / 256（50%） | 名义容量减半：均匀负载下也丢一半——capacity 是硬资源 |
| **cf=1.0** | dropped 26（**10.2%**） | **名义「刚好」下仍有 10.2% 溢出**：均匀-ish（熵 2.053 / 上限 ln8=2.079）也做不到精确等分，capacity headroom 的存在理由 |
| cf=1.25 | dropped 0 | 冗余清零——**本仓 forward 默认 cf=1.25 的设计依据** |
| 负载分布 | max/min = 2.0 | 未训练路由器接近均匀但仍有 2 倍最值差 |
| 偏斜注入 k=1/2/4 | overflow 10.2% → 12.1% → 16.0% | 路由偏斜**单调**放大溢出；§1.2 的 aux loss 就是压这条曲线的机制 |

**诚实框**：合成（随机初始化）路由器不是训练后的路由器——E-A-5 真实训练的专家分布见 §4；探针结论是**机制语义**（溢出率如何随 cf/偏斜变化），不是任何真实模型的负载断言。

---

## 3. 专家并行理论节（理论档：无代码 / 无实验）

> 本仓无多卡 MoE 实验（E-C-4 只测了单卡 FSDP）。本节是文献口径的手算，明标理论档。

### 3.1 EP（Expert Parallelism）布局与 all-to-all

专家分片到卡：E=8 专家、2 卡 → 每卡 4 个专家；数据同时 2 路切分。问题：token 的路由目标专家**不在本卡**时，hidden state 必须搬过去算完再搬回来——每层 MoE 两次 **all-to-all**：

```
dispatch:  token x [d] ──→ 拥有其专家的卡     （去程）
combine :  专家输出 [d] ──→ token 原卡的对应位置（回程）
```

### 3.2 通信量公式 + 手算

每 GPU 每层每方向的通信量 ≈ `tokens_on_gpu × P(专家远端) × d × dtype_bytes`。手算（假设：E=8、2 卡、B·T=32,768 token/step、d=512、bf16）：

- 每 card 持有 16,384 token；4/8 专家本地 → P(远端) = 1/2 → 8,192 token 出、8,192 进（回程对称）；
- 每方向每层：8,192 × 512 × 2 B = **8.4 MB**；dispatch+combine 双向 ≈ 16.8 MB/层；
- 12 层 MoE → **≈200 MB/step/GPU**。PCIe（~25 GB/s）下约 8 ms/步；NVLink（300 GB/s）下可忽略——对 0.22 s/步（146k tok/s ÷ 32k token）分别是 ~4% 与 ~0%。

### 3.3 EP vs DP+replica 取舍

| 维度 | EP（专家分片） | DP+replica（专家全量复制） |
|------|---------------|---------------------------|
| 专家显存 | 每卡 E/卡数 份 | 每卡全量 E 份（2 卡 = 2 倍专家显存） |
| 激活通信 | 每层 2 次 all-to-all（∝ tokens×d） | 无 |
| 梯度同步 | 专家梯度**本卡局部**（不 all-reduce） | 全部参数 all-reduce：137M×4 B ≈ 550 MB/step |
| 同步点 | 12 层 × 2 = 24 次/step（延迟敏感） | 1 次/step（可与计算重叠） |
| 负载敏感 | **straggler**：偏斜时慢卡拖全场 | 无（各卡独立算自己的 batch） |

取舍一句话：**小 E / 少卡，replica 更简单；E 大到显存放不下副本、或专家梯度太大，EP 用激活 all-to-all 换显存**。EP 的 straggler 问题正是 §2 偏斜实验的并行版——负载不均在不均匀的等待时间里变成吞吐损失，生产系统用 capacity + drop（同 §1.3）或 expert choice 路由控制它。

---

## 4. E-A-5 未赢归因（四因子分解）

事实（`ablation-summary.json` / EXPERIMENTS §6.2，同数据同种子同日程，只动架构开关）：

| 组 | val loss | 对比 |
|----|----------|------|
| moe（top-1，E=4） | **3.5824** | — |
| modern-full（稠密同激活） | 3.5730 | moe **+0.0094** |
| no-swiglu（全场最优稠密） | 3.5593 | moe **+0.0231** |

四因子归因（**能说/不能说**边界明确）：

1. **训练预算不足**：18k iters、半个 epoch。路由器从随机初始化开始学「谁擅长什么」，专家各自只见约 1/4 的 token——等效每专家训练量只有稠密的四分之一。文献里 MoE 增益都在更长训练、更大数据下兑现。能说：本预算下路由器大概率还没学好；不能说：18000 iters 一定不够（没有扫更长日程）。
2. **等激活 ≠ 等容量利用**：137.4M 总参是 59.6M 激活的 2.3 倍，但每个 token 只「间歇」激活其中一份——容量红利的前提是路由能把它用起来（因子 1 没成立时它就是死重量）。能说：总参数没有兑现为有效容量；不能说：137M 参数本身没用。
3. **capacity 丢 token**：cf=1.25 下 §2 显示均匀负载零溢出——**在 E-A-5 配置里这条不是主因**，但训练早期路由不均时溢出真实存在（丢 token = 丢梯度信号）。能说：机制上存在、本配置余量充足；不能说：完全无影响。
4. **吞吐税**：146k→115k tok/s（−21%）、墙钟 +18 min。路由 + dispatch + aux 反传是真实的每步开销。这条**无条件成立**，是 MoE 的进场费。

综合：本仓能说的完整句子是——「在 50M 激活、18k iters、半个 epoch 的预算下，top-1 E=4 的 MoE 没有买到更低的 loss，还付了 21% 吞吐和 3 GB 显存」。**不能说**「MoE 不行」——那是把「本预算下没赢」外推成方法否定。

---

## 5. grouped GEMM 5.8× 复盘（ADR-006 + E-C-3）

### 5.1 演化线（三步）

1. **v0.2 Python 循环**：逐专家 `nonzero()/index_select()`，每专家多次 CPU-GPU 同步——ADR-006 的「背景」一节记录的痛点；
2. **ADR-006（2026-06）**：bincount 向量化 aux + 权重预堆叠 + bmm，**保留外层 for 循环**（「PyTorch 不原生支持 grouped GEMM」）；
3. **T012（现行）**：padded grouped-GEMM **全向量化**——stable sort 分派 + 静态形状 `[E, capacity, C]` 缓冲 + 一次批量 matmul，外层循环消失。

**ADR-006 的 superseded 注记**（2026-10 勘误）是文档纪律的教学点：决策第 4 条已被取代，但**不改 ADR 原文**——加注记指向现行实现与实测，历史决策链完整可溯。决策 1-3（bincount aux、权重堆叠、向量化）仍有效并由 `tests/test_moe_vectorized.py` 契约锁定。

### 5.2 E-C-3 实测（4096 token，bf16，`moe_kernel.json` / EXPERIMENTS §6.6）

| E | grouped GEMM | 逐专家循环 | 倍数 |
|---|--------------|-----------|------|
| 2 | 1.98 ms | 2.04 ms | 1.0× |
| 4 | 1.89 ms | 3.43 ms | 1.8× |
| 8 | 1.86 ms | 6.42 ms | 3.5× |
| 16 | 1.98 ms | 11.39 ms | **5.8×** |

grouped **平坦**（1.86-1.98 ms 与 E 无关）：kernel 数量由「每专家一组」变成「一次批量 matmul」，E 涨只改缓冲形状；循环**线性涨**：每专家固定的 launch + 同步开销 × E。这解释了实现选择，**不改变 §4 里 MoE 训练损失没有赢过稠密模型的事实**——kernel 快是让税从更重变轻，不是免税。

---

## 6. 一页速查

| 问题 | 答案 | 锚点 |
|------|------|------|
| top-1 路由在哪 | router → softmax → topk(1) | modern_gpt.py:1046-1051 |
| aux loss 均匀时是多少 | α（0.01）——乘积形式的下限 | L1053-1059 |
| capacity 公式 | `int(cf·B·T/E)`，int 截断 | L1061-1064 |
| 溢出怎么丢 | stable sort 组内保前 capacity 个，输出行置零 | L1070-1094 |
| cf 默认为什么 1.25 | cf=1.0 仍溢出 10.2%，1.25 清零 | §2 / moe_probe.json |
| checkpoint 兼容怎么做 | legacy 逐专家键双向翻译钩子 | L989-1025 |
| EP 通信公式 | tokens × P(远端) × d × bytes × 2（dispatch+combine） | §3.2 |
| E=8/2 卡手算 | ≈200 MB/step/GPU（bf16，12 层） | §3.2 |
| E-A-5 为什么没赢 | 四因子：预算/容量未利用/capacity 非主因/吞吐税 | §4 |
| grouped GEMM 为什么快 | kernel 数与 E 解耦：1.86-1.98 ms 平坦 | §5.2 |
