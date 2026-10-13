# 显存与算力心算手册（Memory & FLOPs Arithmetic）

> 目标：合上手册前，你能**手算**一个 50M 级 GPT 的训练显存、推理 KV、FLOPs 和 roofline 判据，并知道每个数字的可信边界。
> 方法：每节「公式 → 手算 → 工具复算 → 本仓实测对照」四段式。工具：`python -m utils.infra_math --help`。
> 所有实测数字引用自 [EXPERIMENTS.md](../EXPERIMENTS.md) 与 [MIGRATION.md](../MIGRATION.md)，可追溯到 `docs/results/*.json`。

本仓标准模型（下文「50M 模型」）配置：`config/pretrain_cloud.yaml`——12 层 / 8 头 / 512 维 / GQA-2（n_kv_head=2，head_dim=64）/ SwiGLU（中间维 1408）/ RoPE / RMSNorm / vocab 50257 / block 1024 / lm_head 与 embedding 绑定。

---

## 0. 单位底账

| 单位 | 字节 | 记法 |
|------|------|------|
| 1 KiB | 2^10 = 1,024 | 显存行业习惯用 2 的幂 |
| 1 MiB | 2^20 = 1,048,576 | `nvidia-smi` 的 MiB |
| 1 GiB | 2^30 = 1,073,741,824 | 表格里的 "GB" 通常指它 |

**快速锚点**（fp32）：一个参数 4 B；一百万参数 = 4 MiB；一亿参数 = 400 MiB 纯权重。
**换算心法**：看到 "M 参数"，×4 得 MiB（fp32 权重），×16 得训练全账（下文 §3）。

---

## 1. 第一项：参数量 N

### 1.1 公式（逐项对齐 model/modern_gpt.py）

| 部件 | 参数量 | 代码对齐 |
|------|--------|----------|
| embedding（wte，绑定时 lm_head 不重复计） | vocab·d | L1623-1624 张量共享，L1905 按 id 去重 |
| 每层注意力 | q d² + k d·n_kv·hd + v d·n_kv·hd + o d²（全 bias=False） | L335-338 |
| 每层 FFN（SwiGLU） | 3·d·I，I = align₁₂₈(8d/3) | L911-913, L1282 |
| 每层 FFN（GELU） | 2·d·4d = 8d²（I=4d 不对齐） | L1061-1062 |
| 每层 norm | RMSNorm：d（仅 gamma）；LayerNorm：2d | L37 |
| 最终 norm | 同上 ×1 | L1616 |
| QK-Norm（可选） | 2·head_dim/层（RMSNorm(head_dim) 跨头共享 gamma） | L346-347 |
| MoE（E 个专家） | (E−1)·FFN + router d·E | L905-932 |

**手算近似**：非 embedding 参数 ≈ 12·L·d²（注意力 4d² + FFN 8d²）。50M 模型：12×12×512² = 37.7M，实际 33.8M（GQA 收窄 k/v + SwiGLU 对齐 −10%）——近似用于数量级，精确用计算器。

### 1.2 本仓精确账（公式分毫不差版）

**modern-full = 59,560,960**（报告值 59.561M，EXPERIMENTS §6.1）：

| 部件 | 数值 |
|------|------|
| embedding（tied） | 50257×512 = 25,731,584 |
| 每层：attn 655,360 + norms 1,024 + SwiGLU 3×512×1408 = 2,162,688 | 2,819,072 |
| ×12 层 | 33,828,864 |
| 最终 norm | 512 |
| **合计** | **59,560,960** |

**baseline（legacy nanoGPT 块）= 64,085,504**（报告值 64.086M）＝ wte 25,731,584 + wpe 1024×512 = 524,288 + 12×(attn 1,048,576 + GELU FFN 2,097,152 + LayerNorm 2×1,024 + **bias 4,608**) + ln_f 1,024。

三个教学点：
1. **legacy 带 bias（55,296 参数），现代架构全 bias=False**——「为什么 LLaMA 们去掉 bias」的直接对照；
2. **GQA-2 把 k/v 投影从 2×512×512 收到 2×512×128**——modern 比 baseline 少 4.5M 记录参数的主要来源；
3. **去掉 wpe（−524,288）**：RoPE 不占参数表。

**单开关增量**（公式 vs 实测报告，全闭合）：
- MoE（E=4）：+12×(3×2,162,688 + 512×4) = **+77,881,344**（报告 137.442M − 59.561M = 77.881M ✓）
- QK-Norm：+2×64×12 = **+1,536**（报告 qk-norm 组 59.562M ✓）

> 工具复算：`python -m utils.infra_math --n_layer 12 --n_embd 512 --n_head 8 --n_kv_head 2 --vocab_size 50257` → 59,560,960。

---

## 2. 第二项+第三项：梯度与优化器状态

### 2.1 公式与三口径计账表

梯度 = 参数同 dtype 一份；Adam 状态 = m + v 两份。**口径决定一切**：

| 口径 | 参数 | 梯度 | Adam m+v | 静态合计/param | 说明 |
|------|------|------|----------|---------------|------|
| fp32 纯 | 4 B | 4 B | 8 B | **16 B** | `--precision fp32` |
| **AMP autocast（本仓默认）** | 4 B（参数即 fp32 主权重） | 4 B | 8 B | **16 B** | 见下 |
| bf16 纯权重 | 2 B | 2 B | 8 B（状态 fp32） | **12 B** | `--precision bf16-pure`，业界另一派 |

**核心教学点：autocast 不省静态显存，省的是激活。**
本仓 `training/trainer_base.py` `build_amp_context`（L208）用的是 `torch.amp.autocast`：**模型参数本身就是 fp32 主权重**，没有独立的 bf16 权重 + fp32 主权重副本。bf16 只发生在前向的算子内部（激活、matmul 中间量）。所以：
- 50M 模型训练静态账 = 59.56M × 16 B = **953 MiB ≈ 0.95 GB**（与精度开关无关！）；
- 「bf16 权重 + 优化器内 fp32 主权重」是另一流派（部分框架/全 bf16 训练方案），那才是 4+4+4+4 的「经典混合精度 16B」分解——两种约定本手册都讲，**计算器默认仓内口径**，面试时先问清对方用哪种。

### 2.2 本仓验证：moe +1.25 GB 锚点

MoE 比 modern-full 多 77.88M 参数 → 静态增量 = 77,881,344 × 16 B = 1,246,101,504 B = **1.16 GiB ≈ 1.25 GB**（十进制）。
对照 AR-v0.4.0 记录的「优化器态 +1.25 GB」：**精确命中**。这是「口径表 × 参数差」公式的实测闭环。
（E-A 表 moe 组峰值 29.265 GB − modern 26.261 GB = 3.00 GB，其中静态 1.25 GB，其余 ≈1.75 GB 是专家激活/分组 GEMM 工作区——见 §4.4。）

---

## 3. 第四项：激活显存

### 3.1 逐张量表（per layer，bytes/token，fp32；bf16 除以 2）

按本仓 `Block` 前向图逐张量列（autograd 需要为反向保留的量）：

| 张量 | 形状 | bytes/token |
|------|------|-------------|
| 层边界 hidden（grad-ckpt 只存它） | [B,T,d] | d = 512 |
| ln_1 输入/输出 + rstd | [B,T,d]×2~3 | 2d~3d |
| q / k / v 投影输出 | [B,T,d] / [B,T,n_kv·hd]×2 | d + 2·n_kv·hd |
| RoPE 旋转后的 q,k 副本 | 同上 | ≈同上 |
| SDPA 输出 | [B,T,d] | d |
| o_proj 输出 + 残差 | [B,T,d]×2 | 2d |
| FFN（SwiGLU）：gate, up, silu(gate), 乘积 | [B,T,I]×3~4 | 3I~4I（I=1408） |
| FFN（GELU）：h, gelu(h) | [B,T,4d]×2 | 8d |
| MoE 额外：router logits | [B,T,E] | E |

**量级心算**（50M 模型，SwiGLU 路径）：≈ (12~17)·d + 3·1408 ≈ 6~10 KB/token/层 → ×12 层 × 32,768 token/步（batch 32×1024）≈ **2.4~4.0 GB**。
再加**大 vocab logits**：[32,1024,50257] bf16 = 3.29 GB（fp32 上_cast 翻倍）——小模型上 logits 常常比整个 block 激活还大，是新手最常漏的一项。

### 3.2 gradient checkpointing

只存每层输入 [B,T,d]（512 B/token/层 ×12 = 6 KB/token）+ 反向时**重算单层峰值**（≈一层全量激活）。
心算：ckpt 后激活 ≈ L·d·B·T + 单层全量 ≈ 0.2 GB + 单层 0.3 GB，对 50M 模型省下大部分激活，代价是约 +33% 前向 FLOPs（一次额外前向重算）。

### 3.3 本仓验证：单开关组间差的公式分解（E-A，batch 32×1024，bf16 训练）

| 开关（相对 modern-full 26.26 GB） | 实测 Δ | 公式能解释的部分 | 残差 |
|------|--------|------------------|------|
| +SwiGLU（vs no-swiglu 24.74） | **+1.52 GB** | SwiGLU 存 3~4×I=1408 vs GELU 存 2×2048：+0.5~6 KB/token/层 → +0.2~2.4 GB，实测落在区间内 | 0 |
| +MHA（vs mha 26.88） | **−0.62 GB**（GQA 省） | k/v 投影输出 2×512 vs 2×128：−3,072 B/token/层 → **−1.21 GB** | 实测只省一半：SDPA `enable_gqa` 路径在内核内部把 KV 头扩展物化（§5.2 同机理），吃掉了一半收益 |
| +QK-Norm（vs qk-norm 28.15） | **+1.89 GB** | 归一化后的 q[B,8,T,64]+k[B,2,T,64] 副本：2,560 B/token/层 → **−1.01 GB 下界** + rstd/中间量 | ≈0.9 |
| RoPE（vs no-rope 26.25） | **±0** | 旋转是逐头小量，learned PE 表也不进激活 | 0 |
| +MoE（29.27） | **+3.00 GB** | 静态 +1.25 GB（§2.2）+ 专家 FFN 激活近似持平（top-1） | ≈1.75 GB 分组 GEMM 工作区/路由索引 |

**modern-full 比 baseline 多 3.21 GB**（26.26 vs 23.05）的诚实分解：SwiGLU +1.52、GQA −0.62、RoPE/RMSNorm ≈0 → 公式合计 +0.9 GB，**残差 ≈ +2.3 GB 属于块结构差异**（legacy 的 fused c_attn Conv1d 一发算出 qkv vs modern 分立投影 + rope 应用图的张量数更多）——**心算有边界，闭合要 profiler**：这正是 F 组实验 E-F-3 的立项理由（见 [RUNNING_EXPERIMENTS.md](../RUNNING_EXPERIMENTS.md)）。

### 3.4 反面案例：WDDM 共享显存「15.49 GB」噪声

本机（Quadro RTX 5000 16 GB，Windows WDDM 模式）曾测得峰值 15.49/16 GB——**不是干净锚点**：WDDM 会把溢出换页到系统内存，`torch.cuda.max_memory_allocated` 语义与 Linux TCC/default 模式不同。心算手册只采信 Linux 独占模式的 E-A 云端数字（23.1/26.3/29.3 GB）。教训：**报显存必附（GPU 型号 / 驱动模式 / 统计口径）三元组**。

---

## 4. 第五项：KV cache（推理）

### 4.1 公式

```
KV bytes = 2 · L · n_kv_head · head_dim · seq_len · batch · dtype_bytes
每 token 斜率 = 2 · L · n_kv_head · head_dim · batch · dtype_bytes
```
（因子 2 = K 和 V 两个张量；GQA 用 n_kv_head 不是 n_head——这是省的本质。）

50M 模型数字：fp32 12,288 B/token；bf16 6,144 B/token；MHA（n_kv=8）对照 49,152/24,576——**GQA-2 省 75%**（EXPERIMENTS §6.2 记录的 18,432→4,608 为另一组 head_dim=96 的参数集，比值 75% 是重点；绝对值必须核对自己模型的 hd/L/n_kv——**看数字先问参数集**，这本身是面试题）。

### 4.2 本仓案例研究：KV 物化 bug（MIGRATION §8.2）

24 层 / 16 头 / 8 KV / 1024 维 fp32 模型，decode 递增到 4096 token 时实测显存斜率 **12,335 B/token**（R²=0.99999），只有「全量 KV」98,304 的 1/8，且**与层数无关**——为什么？

- 逐层是**串行**执行的：第 ℓ 层的瞬时物化在第 ℓ+1 层时已被释放，分配器复用同一块；
- 所以峰值 ≈ **单层瞬时** 12,288 B/token = 3 × [16 头 × 64 维 × 4 B]——根因是 sm75 无 flash 内核时，`F.scaled_dot_product_attention(..., enable_gqa=True)` 内部把 8 个 KV 头扩展成 16 份再算，每步物化 ~[B,16,S,64]×3；
- 修复（grouped 5D 广播）后：斜率 12,335 → −1,090（平滑），L=4096 时瞬时峰值 57.7 MB → 0.5 MB（114×），decode 68 → **64.5 ms/step**。

教学点：①「层数无关」的内存语义（串行 + 复用）；② **kernel 路径决定显存**——同一个数学，mem-efficient 的 GQA 扩展实现让你多付 8 倍瞬时；③ 修显存顺便提速 5%。

### 4.3 E-B decode 实测对照

- 无 cache 整段重算：141 tok/s @896、1,415 MB；append KV：181 tok/s、1,241 MB——50M 级 append 已全面占优；
- int8 KV：1232 MB（省字节在短上下文几乎看不见，吞吐为 append 的 73%）；
- paged KV：单请求 35 tok/s（块表开销，收益在多请求共享，见 ADR-004）。

---

## 5. FLOPs 心算

### 5.1 公式

**训练 FLOPs/token ≈ 6N + 12·L·T·d**
- 6 = 每个 matmul 权重 MAC 计 2 FLOP，前向 1 次 + 反向 2 次 = 3 次通过 ×2；
- N = 非 embedding 参数（embedding 查表不是 matmul；**tied lm_head 的 vocab 投影照样发生 matmul**，计 2·d·vocab/次）；
- 12·L·T·d = attention 项：每 query token 的 scores（2·T·d）+ 加权求和（2·T·d），×3 次通过，×L 层。

50M 模型 @T=1024：6×33.83M + 12×12×1024×512 = 203M + 75.5M ≈ **278 MFLOPs/token**（lm_head 项 2×512×50257≈51.5M，视口径计入）。

### 5.2 MoE 稀疏 FLOPs

top-1/E：每 token 只走 1 个专家的 FFN → 激活 FLOPs ≈ 稠密等价，但**全部 E 个专家的权重都要驻显存**（FLOPs 稀疏、显存不稀疏——与 §2.2 的 +1.25 GB 静态账互为表里）。
E-C 微基准（4096 token，bf16）：分组 GEMM 1.86–1.98 ms 几乎不随 E（2/4/8/16）变化；逐专家循环 2.04→11.39 ms（16 专家慢 5.8×）——**实现方式决定的 5.8×，与数学无关**。

### 5.3 本仓验证：吞吐 → 利用率

E-C：bf16 147.7k tok/s vs fp32 70.2k（**2.10×**）；FSDP 单卡 148.0k（包装开销可忽略）；梯度累积 157.1k（吞吐口径可比、收敛曲线不可比——迭代计数语义差异，EXPERIMENTS §6.6 有注）。
反算：147.7k tok/s × 278 MFLOPs/token ≈ **41 TFLOP/s**。是否「高」取决于卡型峰值——RTX 6000D 的规格表留给读者按 §6 方法查（这便是 MFU；完整多卡 MFU 核算归 F 组 E-F-2）。

---

## 6. 算术强度与 roofline

### 6.1 定义与判据

```
算术强度 I = FLOPs / 访存字节数 （FLOP/byte）
机器平衡点（ridge）= 峰值 FLOPs/s ÷ 峰值带宽
I > ridge → compute-bound；I < ridge → memory-bound
```

| 负载 | 强度（心算） | 判定 |
|------|--------------|------|
| 大 GEMM [M,N,K] | ≈ K（权重复用 M 次） | K 大时 compute-bound |
| GEMV（M=1） | ≈ 1 | 必然 memory-bound |
| **decode batch=1** | ≈ 1（每 token 读全部权重一次，2N FLOPs / 2N·dtype bytes） | **必然 memory-bound** |
| prefill | ≈ T（每权重吃 T 个 token） | 趋 compute-bound |

**decode 必然 memory-bound 的直觉**：batch=1 时每个权重参数只参与一次 MAC，2 FLOPs 换 dtype_bytes 字节访存——强度恒 ≈ 1/2（fp32）~ 1（bf16），远低于任何卡的 ridge（约 100+ FLOP/byte）。
**但注意方向 ≠ 饱和**：E-B append 178 tok/s @512，权重 bf16 119 MB/step → 有效带宽 ≈ 21 GB/s，远低于卡型标称——50M 小模型 batch=1 时decode 受 kernel 启动/延迟支配，带宽远未吃满。roofline 说的是**瓶颈方向**，不是利用率数值。

### 6.2 本机微实验

`python tools/infra/run_roofline.py --quick`：三段式——① GEMM 尺寸扫描（跨 L2 4 MB 工作集，小尺寸 cache-bound 可见）② GEMV 对照 ③ tiny 模型单 decode 步反推有效带宽。产物 `docs/results/roofline.json`。
本卡规格（Quadro RTX 5000 桌面版）：16 GB GDDR6 / **448 GB/s** / L2 4 MB / sm75（约 11 TFLOPS fp32，3072 核 ×2×~1.8 GHz）。

---

## 7. 五项分解总装（一页心算卡）

50M 模型（batch 32×1024，bf16 autocast，无 ckpt）：

| 项 | 心算 | 结果 |
|----|------|------|
| 参数 | 59.56M × 4 B | 0.24 GB |
| 梯度 | ×4 B | 0.24 GB |
| Adam m+v | ×8 B | 0.47 GB |
| 激活 | §3.1 表 ×12 层 ×32,768 tok + logits 3.29 GB | ≈ 2.4~4.0 + 3.3 GB |
| 静态合计 | 59.56M × 16 B | **0.95 GB** |
| 心算总量 | 静态 + 激活 | **≈ 7~9 GB** |
| 云端实测 | E-A modern-full | **26.26 GB** |

差值 ≈ 18 GB：块结构张量（§3.3 残差）、SDPA/分 GEMM 工作区、allocator 缓存与碎片、cuBLAS workspace——**总量只能精确到量级，结构比例靠单开关消融、精确归因靠 profiler（E-F-3）**。这正是本手册的方法论边界，也是它诚实的地方。

---

## 8. 口径词汇表

| 术语 | 本仓口径 |
|------|----------|
| 参数量 N | `get_num_params`（含 embedding；tied lm_head 按 id 去重）。报告「激活参数」= 稠密等价（MoE 组 59.561M） |
| 训练静态显存 | 参数+梯度+优化器 = 16 B/param（autocast 口径，参数即 fp32 主权重） |
| 峰值显存 | `torch.cuda.max_memory_allocated` + （GPU 型号/驱动模式/口径）三元组 |
| MFU | 实测 FLOPs/s ÷ 卡型峰值 FLOPs/s（完整核算见 E-F-2） |
| token 吞吐 | 处理过的 token 数 / 墙钟秒（梯度累积下迭代计数另注） |
| KV 斜率 | 每 token 的 cache 字节增量；区分「持久 KV」与「SDPA 内部瞬时物化」（§4.2） |
