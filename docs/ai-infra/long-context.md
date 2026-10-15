# 长上下文手册（Long Context: RoPE 外推 / 滑窗驱逐 / 检索 / MLA）

> 目标：合上手册前，你能把「上下文从 1K 扩到 8K」拆成三堵独立的墙——**位置外推**（RoPE 频率被用到训练没见过的区间）、**显存**（KV cache 随长度线性涨）、**检索**（装得下 ≠ 找得到）——并对每一堵墙说出本仓的机制、代码行和实测数字。
> 方法：每节「问题 → 推导 → 本仓验证」三段式；理论节（MLA）无代码无实验，明标「理论档」；实测数字溯源 [EXPERIMENTS.md](../EXPERIMENTS.md) §6.3/§6.11 与 `docs/results/*.json`。
> 姊妹篇：[Attention Kernel 手册](attention-kernels.md)（GQA/Flash/decode 带宽）、[显存与算力心算手册](memory-flops-arithmetic.md)（§4 KV 公式与物化 bug）、[compile-stack.md](compile-stack.md)（§0 GPU 执行模型）。

本仓标准模型：12 层 / 8 头 / 512 维 / GQA-2（n_kv_head=2，head_dim=64）/ block 1024。下文未注明处均指该配置。

---

## 0. 全景：三堵墙与机制地图

| 墙 | 症状 | 本手册章节 | 本仓机制 |
|----|------|-----------|----------|
| 位置外推 | PPL 在超训练长度后爆炸 | §1 NTK / §2 PI+YaRN / §3 retune | `rope_scaling` 四档 + `rope_retune.py` |
| 显存 | KV cache 线性增长 | §4 滑窗驱逐 / §7 KV 复习 | GQA-2 + `sliding_window_size` + int8 |
| 检索 | 能装下但找不到 needle | §5 needle 方法论 | `needle_in_haystack.py`（E-A-9） |
| （综合） | 大头数下 KV 太大 | §6 MLA（理论档） | 无代码，明标 |

三堵墙**正交**：外推改频率分布、显存改存储结构、检索改注意力行为。选型时分别决策，不要混为一谈（§7 末尾的决策树）。

---

## 1. NTK 复盘：既有正结果的机制读法

### 1.1 RoPE 一页

RoPE 对 query/key 的每对维度 (2i, 2i+1) 施加旋转，旋转角 = 位置 m × 频率 `inv_freq_i = 1/θ^(2i/dim)`。**i 小 → 高频（短波长，编码局部位置差）；i 大 → 低频（长波长，编码长程位形）**。注意力分数只依赖相对旋转 (m−n)·inv_freq——这是所有长度外推方法只动 `inv_freq`/base、不动注意力代码的原因。

### 1.2 NTK-aware 公式

NTK 不改公式，改 base：`θ' = θ · s^(dim/(dim−2))`（s = 目标长度/训练长度）。效果是**波长谱整体拉伸 s 倍**：训练时最大波长 `2π·θ` 的维度现在覆盖 `2π·θ'`，而高频维度几乎不动（`θ^(2i/dim)` 在 i 小时对 θ 不敏感）。直觉：让「ruler 的刻度间距放大 s 倍」而不是「把位置编号除以 s」。

**本仓验证**：`model/modern_gpt.py:119-125`——`RotaryEmbedding.__init__` 的 ntk 分支，factor≠1 时 `theta = base * factor**(dim/(dim-2))`；`tools/experiments/run_ntk_eval.py` 是 E-A-7 入口（本 AR 不改它，命令字节稳定，见 §3.3）。

### 1.3 E-A-7 实测（modern-full，train block 1024）

| 长度 | 朴素 PPL | NTK PPL | 因子 |
|------|---------|---------|------|
| 512 / 1024 | 38.50 / 40.46 | 同左 | 1 |
| 2048 | 82.67 | **39.09** | 2 |
| 4096 | 213.80 | **52.50** | 4 |

2x 长度 NTK 把 PPL 拉回训练长度附近；4x 仍高于训练长度（外推有代价）但退化小一个数量级。本轮最干净的正结果：**改的是位置频率，不是再训练**。溯源：`docs/results/ntk_eval.json`、EXPERIMENTS §6.3。

factor=1 时 NTK 与朴素逐位相同（同一公式 theta 不变）——这个恒等性在 E-A-8 的四档对照里被用作实现健全性检查（§2.4）。

---

## 2. PI 与 YaRN：机制推导 + 本仓实现对照

### 2.1 PI（Position Interpolation）：线性压缩位置

PI 把位置 m 映射为 m/s：`m·(inv_freq/s) == (m/s)·inv_freq`——**数学上等价于把位置轴压缩 s 倍再套用训练好的 RoPE 表**。所有维度（含高频）一起被压缩。

代价：高频维度的分辨率同步损失——训练时 1 个 token 的相位差变成 1/s 个。**零样本评测下这是净损失**（模型从没见过非整数的有效位置差），PI 原论文的做法就是在扩展长度上**继续微调**，让模型适应压缩后的相位。零样本劣势不是实现 bug，是方法本身的性质（§2.4 实测佐证）。

### 2.2 YaRN：NTK-by-parts 分维度处理 + 温度

YaRN 的观察：外推失败主要是低频维（长波长）被用到没见过的相位区间；高频维本来就只编码局部差，外推无害。于是**按维度分治**：

- 对频率对索引 i（0..dim/2−1），定义 `d_corr(r) = dim·ln(train_len/(2πr)) / (2·ln θ)`——波长在 train_len 内完成 r 圈旋转的维度索引；
- ramp 区间 `[d_corr(32), d_corr(1)]`（β_fast=32, β_slow=1，论文默认）：**i 小（高频）→ 外推不动；i 大（低频）→ 插值（inv/factor）；区间内线性混合**：`inv' = (1−t)·inv + t·(inv/factor)`，`t = clamp((i − lo)/(hi − lo), 0, 1)`；
- 注意力温度：`t_attn = 1 + 0.1·ln s`，乘进 attention logits（`attn_scale = t/√d`）——长上下文里 key 数量增多、相似度质量摊薄，t>1 让 softmax 更尖锐（降熵），缓解注意力过度分散；

**本仓验证（逐行对照，BS7）**：

| 机制 | 代码锚点 | 语义 |
|------|----------|------|
| ramp 索引域 | `model/modern_gpt.py:75-83` `_yarn_scaled_inv_freq` | `d_corr` 在**频率对索引空间**（0..dim/2−1），`idx = arange(dim//2)`（L80）；早期版本误用 `arange(0,dim,2)` 混淆维度索引与频率对索引——勘误：i 小=高频维→外推不动，i 大=低频维→inv/factor |
| clamp 混合 | L82-83 | `(1-t)·inv + t·(inv/f)`，t∈[0,1] |
| PI 分支 | L129-131 | `inv_freq /= factor`（等价 m/s） |
| ntk 分支 | L119-125 | §1.2 公式 |
| type 校验 | L115-117 | 非 ntk/pi/yarn → ValueError 列合法域 |
| 温度复用 | `model/modern_gpt.py:636-638` | `attn_scale = attn_temperature / sqrt(head_dim)`——**不加新注意力路径**，YaRN 温度写进既有 `attn_temperature`（L398/L1386 挂载），naive 复位 1.0 |
| config 校验 | `utils/config_models.py` | field_validator：type ∈ {ntk,pi,yarn}、factor ≥ 1.0、train_len > 0；learned 位置编码 + rope_scaling 组合仍拒绝（消息含 learned） |
| 内省 | L144-147 `rope_type` property | None / pi / yarn |

### 2.3 训练侧入口

`training/train_pretrain.py --rope_scaling_type {none,ntk,pi,yarn} --rope_factor s`：type ≠ none 组装 `rope_scaling={"type": t, "factor": s}`；factor < 1.0 → argparse 报错退出 2；`effective_block_size`（`model/modern_gpt.py:1470`）在 pi/yarn 下把 Block 构造的注意力窗口扩到 block×s（NTK 不扩——它改的是频率不是位置域）。

### 2.4 E-A-8 四档实测（本地低算力档，云端 50M 档待回填）

tiny checkpoint（4L/4H/256D、block 256、600 iters、val 6.02），`tools/experiments/run_context_eval.py`，factor = max(1, L/train_block)：

| L | factor | naive | ntk | pi | yarn |
|---|--------|-------|-----|-----|------|
| ≤256 | 1 | 四档全等（恒等健全性） | | | |
| 512 | 2 | 428.10 | 422.51 | **481.93** | **421.28** |
| 1024 | 4 | 510.01 | 497.98 | **620.98** | **488.13** |

- 排序：**yarn < ntk < naive < pi**。YaRN 兼得 NTK 的高频保持与 PI 的低频压缩，最优；PI 零样本 −21.7%（vs naive），§2.1 机制预言的分辨率损失；
- **诚实框**：600-iter 弱模型，排序方向与文献一致但幅度不可比；同 L 内比较才有意义（同 windows 同 seed，跨 L 窗口 token 数不同）；50M 全长度档（modern-full，512→4096）待云端回填。溯源：`docs/results/context_eval_tiny_local.json`、EXPERIMENTS §6.11。

---

## 3. retune 工程面：不重训的动态外推

### 3.1 四档分发（`tools/experiments/rope_retune.py`）

`retune_rope(model, method, factor, max_len, train_block)` 逐 `RotaryEmbedding`/`CausalSelfAttention` 模块**就地改写**（inv_freq 重建、attn_temperature 改写、rope_scaling/config 同步），`LEGAL_METHODS = (naive, ntk, pi, yarn)`（L22），非法值 ValueError 列合法域（L61-62）：

- `naive`：复位 base=10000、temperature=1.0、rope_scaling=None——**复位是显式语义**，保证四档对照里 naive 是干净基线而不是「上一个 method 的残留」；
- `ntk/pi/yarn`：`ntk_theta`/`pi_inv_freq`/`yarn_inv_freq`（L25-41）与 `RotaryEmbedding` 构造路径同公式。

### 3.2 两种使用方式

| 方式 | 入口 | 语义 |
|------|------|------|
| 训练时带 scaling | `train_pretrain --rope_scaling_type` | 模型**在压缩相位上学习**，PI/YaRN 的正规用法（PI 需要这个） |
| 评测时 retune | `rope_retune.retune_rope`（E-A-8/9 调用） | checkpoint 不动，**零训练成本换外推**；NTK/YaRN 零样本可用，PI 零样本有代价 |

### 3.3 为什么 run_ntk_eval.py 不改（D4）

E-A-7 是已归档实验，`run_ntk_eval.py` 的命令行字节稳定是复现性的硬约束。新入口 `run_context_eval.py` 复用共享 `rope_retune`，NTK 分支两处实现的公式一致性由 `test_rope_retune.py`（UT14）断言 inv_freq allclose 锁定——**同公式两处实现，用测试锁一致性而不是合并**（E-A-7 命令字节稳定优先）。

---

## 4. 滑窗注意力 + cache 驱逐：显存墙的有界解

### 4.1 机制

注意力只看最近 W 个 token：训练代价 O(T·W)（W≪T 时显著省），decode 的 KV cache 上界 = W。Mistral 7B 的口径是 W=4096——「窗口内做全注意力，窗口外不存在」。**与外推正交**：窗口内相对位置 ≤ W，始终在训练分布内，所以滑窗模型不需要长度外推；代价是**主动放弃远距检索**（§5 的 needle 会直接暴露这一点）。

### 4.2 本仓验证

| 机制 | 锚点 | 语义 |
|------|------|------|
| 配置挂载 | `model/modern_gpt.py:455-456` | `sliding_window_size`（None=禁用） |
| 窗口激活条件 | L640-646 | `past_kv is None and T > 1`——prefill/整段前向走窗口 mask；decode 单步（T=1）不需要 mask，靠 cache 驱逐保证窗口 |
| 窗口 mask | `model/kv_cache_utils.py:632` `build_sliding_window_mask` | 加性 bool mask，与 causal/padding/document mask 合并成单一张量喂 SDPA |
| **cache 驱逐** | L595-615 | list（分块）形态：**逆序**遍历块裁到累计 W，整块保留/尾部切片；value 形态：`past_k[:, :, -W:, :]` 尾部切片。KV 显存有界于 W |

测试锚（`tests/test_sliding_window.py`，5 用例）：mask 形状 / 前向可跑 / **滑窗改变输出**（不是恒等装饰）/ **cache 有界**（decode 步数≫W 后 past 长度 == W）/ config 序列化往返。第 4 条是本节机制的直接断言。

### 4.3 工程权衡

滑窗是三堵墙方案里唯一**主动丢信息**的：外推和 cache 优化都不改变「理论上能看到所有历史」，滑窗是硬截断。选它意味着接受「超过 W 的依赖不成立」——对话摘要、长程指代会断。所以顺序上：先 GQA（无质量代价）→ 再 int8（可测退化）→ 滑窗放最后（见 §7 决策树）。

---

## 5. needle 检索方法论（E-A-9）

### 5.1 为什么 PPL 不够

PPL 衡量平均预测质量，一个只靠 n-gram 平滑的模型可以有不错 PPL 而**完全不具备检索能力**。needle 测试强制制造一个「不看全文就答不出」的任务：把一个随机事实（needle）插进长文档（haystack）中间，问模型这个事实——能答对才算「上下文里的信息真的可用」。

### 5.2 网格语义

- **depth**：needle 插入位置比例。0.0=文档开头，1.0=紧挨问题——同一上下文长度下扫 depth，把「位置编码崩」和「注意力稀释」区分开：**位置编码崩先在深 needle（远离问题的位置）出现**；注意力稀释则全 depth 均匀劣化；
- **context length**：haystack 长度扫描，与 §1/§2 的外推长度对齐——超训练长度后的崩塌形态与 RoPE retune 档交叉（E-A-9 支持 `--rope_method` retune 后再评，就是做这个交叉）；
- **trials**：每格重复采样（needle 内容随机），抑制单样本偶然性。

### 5.3 本仓验证

- `evaluation/needle_in_haystack.py`：`evaluate` 返回 `{scores[], summary{overall_accuracy, by_depth, by_length}, raw}`——by_depth 聚合就是 5.2 的崩塌形态判读入口；
- `tools/experiments/run_needle_eval.py`（E-A-9 入口）：tiktoken `gpt2` 分词（50257 词表与本仓 ckpt 对齐，懒 import + 缺依赖 exit 1 明确报错）；`--rope_method/--rope_factor` 走 §3 retune；默认网格 4 长度 × 5 深度 × 3 trials，`--quick` 低算力档。

**本地档实测（诚实框）**：tiny 弱模型（val 6.02）[128,256]×3 depths×1 trial，accuracy **0.0**——弱模型无检索能力，预期内；本档结论仅「管道贯通」（网格遍历/聚合/落盘语义正确），检索能力结论等 50M 档。溯源：`docs/results/needle_eval_tiny_local.json`、EXPERIMENTS §6.11。

---

## 6. MLA 理论节（理论档：无代码 / 无实验）

> 本节全部是文献推导，本仓无实现无实测；对比表中 MHA/GQA/MQA 行可用 `utils/infra_math.py` 口径复算（KV = 2 × L × n_kv × head_dim × dtype 字节），MLA 行是理论推导。

### 6.1 动机：共享头 vs 压缩表示

MQA/GQA 靠**共享 KV 头**省 cache：n_kv 从 8 降到 1（MQA）省 8 倍，但所有 query 头共用一套 K/V 表征，共享过头质量掉得明显。MLA（DeepSeek-V2）换轴：不共享头，**压缩 token 的 KV 表示**——

- 下投影：`c = W_DKV · h`（latent 维 d_c ≪ 2·n_h·d_h）；
- 上投影：`K = W_UK · c`，`V = W_UV · c`——每个 query 头仍有自己的 K/V 视图；
- **吸收**：attention 里 `qᵀ·K = qᵀ·W_UK·c = (W_UKᵀq)ᵀ·c`——把上投影**吸收进 query 权重**，cache 只需存 c，decode 时在 query 侧升维。

### 6.2 解耦 RoPE

吸收与 RoPE 不相容：`RoPE(q)ᵀ·RoPE(K)` 里的旋转矩阵夹在 `W_UK` 中间提不出来。DeepSeek-V2 的解法是**解耦**：位置部分单独存一小段 `k_R = RoPE(W_R·h)`（d_r 维），内容部分走 latent c。cache = `c + k_R`，两者都能吸收/直接参与。

### 6.3 每 token 每 层 cache 对比（本仓口径）

元素数 = cache 张量每层元素；本仓 12 层、head_dim=64、fp16（2 B/元素）：

| 方案 | 元素/token/层 | 12 层 fp16 | 注 |
|------|--------------|-----------|----|
| MHA（n_kv=8） | 2×8×64 = 1024 | 24 KB | 基线 |
| **GQA-2（本仓）** | 2×2×64 = 256 | 6 KB | E-A-4：val loss 代价 0.033 |
| MQA（n_kv=1） | 2×1×64 = 128 | 3 KB | 共享过头，文献质量损失明显 |
| MLA（理论：d_c=512 + d_r=64） | 576 | ≈13.5 KB | **与头数无关** |

**诚实框**：在本仓 8 头小模型上，理论 MLA 的 cache（576 元素/层）反而**大于** GQA-2（256）——MLA 的收益随头数规模增长（DeepSeek 口径 32 头×128 维：MHA 8192 vs MLA 576，约 7%），小模型 GQA 是正确选择。关键不变量：**MHA/GQA/MQA 的 cache ∝ 头数（或组数），MLA 的 cache 只 ∝ latent 维度**——头数上去之后差距才拉开。

---

## 7. 长上下文显存压力复习（KV 一页）

公式：`KV bytes = 2 × n_layer × n_kv × head_dim × dtype_bytes × tokens`。本仓 GQA-2 fp32 = 12,288 B/token（[memory-flops-arithmetic.md §4.1](memory-flops-arithmetic.md) 口径），4096 ctx 全 cache 约 50 MB——50M 模型上不是问题。规模上去之后（128 头×128 维 fp16 × 80 层 ≈ 5.2 MB/token）：1M ctx ≈ 5.2 TB——**长上下文在大模型上是显存问题第一、算力问题第二**，这就是 GQA/MLA/量化/驱逐存在的理由。

- 物化 bug 案例：SDPA 后端隐式拷贝导致 KV 显存不降反升，[memory-flops §4.2](memory-flops-arithmetic.md)；decode 带宽视角（KV 搬运是 decode 瓶颈）[attention-kernels §5](attention-kernels.md)；
- int8 KV 量化：E-B-2 实测（显存减半、吞吐损失 <10%、PPL 退化可接受，EXPERIMENTS §6.4）。

**决策树**（给定目标 ctx 与显存预算，从上往下，代价从小到大）：

1. 先 GQA（无质量代价，本仓默认已开）；
2. 不够再 int8 KV（可测退化，先跑 E-B-2 类基准）；
3. 还不够：滑窗 W=目标 ctx（硬截断，接受 §4.3 的信息损失）或 MLA（工程复杂度高，头数大才值得）；
4. 位置外推**单独走** retune（§3：不动 cache，只动频率）——显存方案和它正交，可叠加。

---

## 8. 一页速查

| 问题 | 答案 | 锚点 |
|------|------|------|
| NTK 改什么 | base：`θ·s^(d/(d−2))`，波长谱整体拉伸 | modern_gpt.py:119-125 |
| PI 为什么零样本差 | 高频分辨率同步损失，需微调 | §2.1 / E-A-8 实测 |
| YaRN 核心一句话 | 高频外推、低频插值、ramp 线性混合 + 温度 | _yarn_scaled_inv_freq L61-83 |
| YaRN 温度怎么进来的 | 复用 `attn_temperature`，不加新路径 | modern_gpt.py:636-638 |
| 四档 retune 入口 | `rope_retune.retune_rope`（naive 是显式复位） | rope_retune.py:22,43 |
| 滑窗 decode 为什么不用 mask | T=1 单步，cache 驱逐已保证窗口 | L640-646 + L595-615 |
| 装得下=找得到？ | 否，needle 网格才是检索证据 | §5 / E-A-9 |
| MLA 什么时候值得 | 头数大：cache 与头数无关 | §6.3 诚实框 |
| 长上下文显存第一公式 | 2·L·n_kv·hd·dtype·tokens | memory-flops §4 |
