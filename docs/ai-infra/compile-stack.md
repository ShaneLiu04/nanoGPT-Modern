# torch.compile 编译栈手册（Dynamo / Inductor / CUDA Graphs）

> 目标：合上手册前，你能画出 torch.compile 的三层结构、说出「什么时候快、什么时候慢、什么时候重编译」，并知道本仓每处 `torch.compile` 调用为什么那样写。
> 方法：每节「原理 → 本仓验证」两段式；dynamo 层给出**本机可复跑**的内联实验；inductor 层给理论 + 云端 SOP（本机断裂即教学案例）。
> 姊妹篇：[Attention Kernel 深潜手册](attention-kernels.md)（§6 MoE 静态形状是 compile 的好朋友）、[显存与算力心算手册](memory-flops-arithmetic.md)。

本机环境：Windows / Quadro RTX 5000（sm75）/ torch 2.5.1+cu121 / triton 3.8.0。**本机 inductor 不可用**（§5 教学案例），dynamo 层完整可用。

---

## 0. 前置章：GPU 执行模型（读懂 BLOCK/num_warps 的地基）

> 本手册 §2-§5 讲 compile 栈「做什么」，但 Inductor/Triton 的每个决策（tile 多大、几个 warp、哪里用共享内存）都以下面的硬件模型为地基。不先懂这一章，§5 的 fusion 决策和 triton_kernels.py 里的参数就只是「别人的魔法数字」。本章可独立先读。

### 0.1 SM 与 warp：调度的两级单位

GPU 的最小**调度**单位是 warp（32 个线程，锁步执行同一条指令）；最小**执行**单位是 SIMT 通道。本机 Quadro RTX 5000（Turing，sm75）的物理结构：

| 层级 | 数量 | 说明 |
|------|------|------|
| SM（Streaming Multiprocessor） | 48 个 | 独立的「小 CPU」，含自己的寄存器堆/共享内存/调度器 |
| 每 SM FP32 核 | 64 个 | 48 × 64 = 3072 个 FP32 核 |
| 每 SM 驻留线程上限 | 1024（=32 warps） | occupancy 100% 的含义就是 32/32 |
| boost 时钟 | ~1815 MHz | 理论 FP32 峰值 = 3072 × 2 FLOP/clk × 1.815 GHz ≈ **11.15 TFLOPS** |

一个 warp 里 32 个线程的地址最好落在同一 DRAM/L2 事务内（合并访问，coalescing）——这就是为什么本仓 `tools/infra/triton_kernels.py` 里 tile 的最内维度总是连续的 `tl.arange(0, BLOCK)`：一个 warp 拿一段连续内存，一次搬完。

### 0.2 内存层级：越近越快、越小、越贵

| 层级 | 本机容量/带宽（数量级） | 备注 |
|------|------------------------|------|
| 寄存器 | 64K × 32bit/SM（=256KB），聚合带宽 TB/s 级 | Triton 里 `tl.zeros((BLOCK_M, BLOCK_N))` 的累加器就活在这里（triton_kernels.py:143-144 的 `acc_g/acc_u`） |
| 共享内存/L1 | 96KB 统一（可划 64KB 做 SMEM），~几十 TB/s 聚合 | tile 循环里反复使用的 A/B 分块先搬进来 |
| L2 | 4MB，~1-2 TB/s | 48 个 SM 共享 |
| HBM（GDDR6） | 16GB，**448 GB/s**（roofline.json `spec_bandwidth_gbs`） | 一切数据的源头，最慢的一环 |

数量级记忆法：寄存器:SMEM:L2:HBM ≈ **快 4 个台阶，每台阶约 4-10×**。fusion（§5.2）的全部意义就是别让中间结果掉到 HBM 再捞回来。

### 0.3 occupancy 与延迟隐藏

occupancy = 驻留 warp 数 / 32。它不直接等于性能，而是**延迟隐藏的弹药量**：一个 warp 等 HBM 数据（几百 ns）时，SM 切到另一个 warp 继续算。三资源决定 occupancy：寄存器（每线程占用越少、驻留 warp 越多）、SMEM（每 block 占用越少、驻留 block 越多）、线程槽（1024/SM）。

这直接解释 roofline.json 的两极：
- 大 GEMM（8192³）：tile 足够多，SM 满载，延迟被并行 warp 隐藏 → 9.83 TFLOPS（接近 §0.6 理论峰值的 88%）；
- decode 单步（`effective_gbps: 4.81`）：batch=1 时每个 kernel 只有一点点线程，SM 大部分时间空转等 launch——**不是带宽不行，是根本没喂满**（§1 三段式的第一二段主导）。

### 0.4 异步拷贝的进化线（本机明标：sm75 无 cp.async/TMA）

global → SMEM 怎么搬，三代硬件三种答案：

| 架构 | 机制 | 说明 |
|------|------|------|
| Turing（sm75，**本机**） | LDG → 寄存器 → STS 两跳 | global 只能先读到寄存器再写 SMEM，寄存器被搬运占用 |
| Ampere（sm80+） | `cp.async` | global 直达 SMEM，搬运不占寄存器、可与计算重叠 |
| Hopper（sm90） | TMA | 按**描述符**整块搬运，单条指令驱动，配合 warp 特化（producer/consumer） |

**本机 sm75 两者皆无（明标：无代码可验证，理论档）**——Triton 在本机生成的 tile 循环就是「两跳」形态。这解释了为什么 Triton 的 tile 循环在 Ampere 上收益更大：sm80+ 才有真正的 load/compute 重叠。

### 0.5 BLOCK/num_warps 怎么选：与本仓对照

Triton 把 tile 大小（BLOCK_*）和每 block 的 warp 数（num_warps）交给作者。选择逻辑全是 §0.1-§0.3 的推论：

- **tile 不能太小**：BLOCK=16×16 的 tile 只有 256 元素，一个 block 里 warp 太少，喂不满 SM；太大又让 occupancy 掉下去（SMEM/寄存器超限）或浪费在 padding 上。
- **本仓 `triton_kernels.py:323-325` 的 swiglu tile 用 BLOCK_M=16/BLOCK_N=16/BLOCK_K=16**：MoE 的 padded buffer 是 `[E, capacity, C]`，capacity 行往往不满——大 tile 会把算力浪费在 padding 行上，小 tile 保住「算的每一行都是真 token」。这是「形状特征优先于吞吐率教条」的实例。
- **`BLOCK_N=triton.next_power_of_2(N)`（triton_kernels.py:353）**：fused rmsnorm/softmax 一行（或一列）一个 block，BLOCK 覆盖整行避免跨 block 归约；next_power_of_2 是 Triton `tl.arange` 的硬约束（2 的幂）。
- **num_warps 默认 4**：一个 block 4 warps × 32 线程 = 128 线程，处理 256 元素 tile 时每 warp 拿 64 元素（2 轮×32），是「tile 元素数 / 32」向上取的常用起点；Inductor 自动调优（§2.3）本质上就是帮你扫这些组合。

### 0.6 roofline 算术闭合（本手册的验收口径）

理论峰值与实测的闭合比值定义为**理论/实测 ∈ [0.5, 2]**，超界必须解释：

| 项 | 理论 | 实测（roofline.json） | 比值 | 判定 |
|----|------|----------------------|------|------|
| FP32 峰值 | 11.15 TFLOPS（§0.1 推导） | 9.83 TFLOPS（gemm 8192³） | 1.13 | ✓ 闭合（缺的 12% 是 DRAM 供数跟不上，即 §5.2 的带宽墙） |
| 带宽峰值 | 448 GB/s（GDDR6 规格） | 349.21 GB/s（gemv n=k=4096） | 1.28 | ✓ 闭合（gemv 只读不写、L2 命中不足） |
| decode 带宽 | 448 GB/s | 4.81 GB/s | 93× | ✗ **超界且已解释**：非带宽瓶颈，launch/Python 开销主导（§1），这正是 compile 三板斧的靶子 |

第三行是本手册存在的理由：当 gap 不是 1.1× 而是 93×，优化对象就不是 kernel 内部而是调度开销——§1 从这里展开。

---

## 1. 为什么需要 compile

一个朴素 eager 步骤的耗时由三段叠加：

```
总耗时 = Python 解释开销（调度、dispatch、对象创建）
       + kernel 启动开销（每次 op 一次 launch，~微秒级）
       + kernel 内计算（真正的 FLOPs）
```

小模型 / 小 batch 时前两段占比极高（本仓 decode 单步有效带宽仅 4.81 GB/s，roofline.json——大量时间不在计算）。torch.compile 的三板斧分别打这三段：

| 层 | 打什么 | 手段 |
|----|--------|------|
| Dynamo | Python 开销 | 把 Python 前向**捕获成 FX 图**，之后按图执行，不再逐 op 走解释器 |
| Inductor | kernel 启动 + 计算效率 | 把图**融合（fusion）**后生成 Triton kernel：逐元素/归约合并成一个 kernel，中间结果不落显存 |
| CUDA Graphs | kernel 启动开销 | 把整段 kernel 序列**录制一次、整体回放**，launch 开销从 N 次变 1 次 |

---

## 2. 三层架构

### 2.1 Dynamo（图捕获 + guard + 重编译）

Dynamo 在首次执行时把 Python 字节码跟踪（trace）成 FX 图，并为输入生成 **guard**（守卫条件：形状、dtype、python 标量等指纹）。之后每次调用先验 guard：**通过则直接跑已编译的图；不通过则重编译**（recompile）并登记新 guard。

触发重编译的典型来源：

| 来源 | 例子 | 本仓对应 |
|------|------|----------|
| 形状变化 | batch/seq 维变化（未开 dynamic） | §3 两条路径都开 `dynamic=True` 规避 |
| Python 标量/属性变化 | `if x.shape[1] > 1024:` 分支两侧 | fullgraph 路径刻意排除数据依赖分支（§3） |
| 字典/自定义对象结构变化 | 往 dict 里塞新键 | MoE 向量化消除了逐专家 dict（§4.2） |

`fullgraph=True` 是**约束不是优化**：要求整个前向无图断裂（graph break），断则抛错——用于保证「整个 decode step 是一张图」，配合 CUDA Graphs 整体录制。

### 2.2 AOTAutograd（前反向分解）

图捕获之后、后端之前，AOTAutograd 把前向图扩展出反向图（联合分解），让 inductor 一次生成前反向代码。训练态 compile 的关键中间层；本仓训练链路的 `torch.compile(model)` 隐式经过它。

### 2.3 Inductor（fusion 决策 + Triton codegen + 自动调优）

Inductor 把 FX 图的算子做**融合**：逐元素链（mul/add/激活）和归约（sum/mean）合并成尽量少的 Triton kernel，中间激活不再写回显存（省带宽也省显存）。`max-autotune` 模式还会对候选 kernel 做基准择优。**手写 Triton 与 inductor 的分工**：inductor 自动生成通用融合，手写 kernel 负责它覆盖不到的结构性优化（本仓 Triton 微实验见 `tools/infra/triton_kernels.py`——教学级三件套）。

### 2.4 CUDA Graphs（reduce-overhead 模式）

`mode="reduce-overhead"` 在 inductor 产物之上再包一层 CUDA Graph 录制：整段 kernel 序列一次录制、逐步回放。代价是**静态显存语义**——输入/输出张量地址被固化在图里，dynamo 用静态 buffer + 拷入拷出模拟原地更新；多进程/多线程共享 buffer 时有已知坑。适合**形状固定的 decode 循环**，不适合形状多变的训练。

---

## 3. 本仓调用面：三个位置，三种写法

### 3.1 训练：FSDP 外层 compile（trainer_base.py:755-758）

```python
# trainer_base.py:743-758（FR9b）：先 FSDP 包裹，再 compile 最外层
model = FSDP(model, ..., use_orig_params=True)
self._fsdp_module = model
self.raw_model = model.module
if compile_model:
    model = torch.compile(model)          # L758：compile 在 FSDP 之外
```

**顺序为什么是 FSDP → compile**：官方组合序。compile 包在最外层看到的已是 FSDP 展平后的参数视图，避免 dynamo 跟踪进 FSDP 的 all-gather 内部逻辑造成图断裂。DDP 分支同理在 `DDP(...)` 之后再 compile（L769-778），且 DDP 用 `static_graph=True` + `gradient_as_bucket_view`（L772-773）——固定图结构让 DDP 的桶与 compile 后的图形状假设一致。

**EMA / checkpoint 不受影响的机制**：`self.raw_model`（L754/775/779）始终保存**未包装原模型**，EMA 更新与 state_dict 保存全部走 `raw_model`——编译/FSDP 包装层永远不进权重快照。

### 3.2 推理路径 A：整模型 reduce-overhead（modern_gpt.py:2084-2099）

`generate(compile=True)` 把**整个模型**编译为 `mode="reduce-overhead", fullgraph=False, dynamic=True`（L2087-2089），失败则警告并回退 eager（L2090-2098）。`fullgraph=False`：generate 循环里有 `torch.cat`、终止判断等天然图断裂，强行整图必失败；`dynamic=True`：prompt 长度不定，避免每换一个长度就重编译。

### 3.3 推理路径 B：单步 fullgraph（modern_gpt.py:2276-2344，`_generate_fullgraph`）

与 A 相反的思路：**不编译整个 generate，只编译「单 token 前向 + 采样」这一个函数**（L2301-2331 `_decode_step`），用 `fullgraph=True, mode="reduce-overhead", dynamic=True` 编译（L2336-2340），外层留一个纯 Python 循环。

能整图的前提被显式收紧（L2052-2060 `can_fullgraph`）：`use_cache=True`、无 repetition_penalty、无滑动窗口/paged cache、`prompt+max_new_tokens ≤ block_size-1`。**循环体里禁止数据依赖 break**（L2288-2293、L2386 注释）：EOS 判断改成 `torch.where` 掩码——已结束序列继续发射 eos（L2325-2330），牺牲一点空转换取每步同构。

另一个细节（L2309-2315）：eager 路径的 `past_kvs` 传 **KVCacheManager 对象**（走零拷贝 append 快路径），compiled 路径传 **value 列表**——dynamo 对自定义 cache 对象的 fullgraph 跟踪不可靠，编译态退回 value-form。这是「compile 友好性」与「最优 eager 路径」的显式权衡。

### 3.4 训练 MoE：静态形状的回报

§6 的 MoE padded grouped GEMM（`[E, cap, C]` 静态缓冲）正是 compile 喜欢的形状：容量截断换来「每次前向图结构完全一致」。验证见 §4.2。

---

## 4. Dynamo 层实测（本机可复跑）

> 以下代码块复制即跑（CPU 亦可；与 MIGRATION §8.1 方法一致：dynamo 层指标与 backend 无关）。

### 4.1 图捕获体检：`torch._dynamo.explain`

```python
import torch
import torch._dynamo as dynamo

def f(x, w):
    return torch.relu(x @ w) * 0.5

x, w = torch.randn(8, 16), torch.randn(16, 32)
exp = dynamo.explain(f)(x, w)
print(exp.graph_count)              # 捕获出的图数量（1 = 无断裂）
print(exp.break_reasons)            # 每处断裂的原因与代码行
print(exp.ops_per_graph)            # 每张图的算子数
```

> 属性名以 torch 2.5.1 实测为准：`ExplainOutput` 暴露的是 `break_reasons`（不是 `graph_break_reasons`）。本机实测输出：`graph_count=1`、`break_reasons=[]`、`ops_per_graph=[[matmul, relu, mul]]`。

`graph_count > 1` 即存在 graph break——逐条看 reason 决定「容忍（fallback 也正确）」还是「消除」。

### 4.2 重编译计数：什么在触发 recompile

```python
import torch
import torch._dynamo as dynamo

def g(x):
    return x * 2 if x.shape[0] <= 4 else x * 3     # 数据依赖分支

dynamo.reset()
compiled = torch.compile(g, backend="eager")       # 计数走 compile 路径
for n in (2, 4, 8, 8, 8):
    compiled(torch.randn(n, 3))
    print(n, dynamo.utils.counters["stats"]["unique_graphs"])
```

本机（torch 2.5.1）实测输出：`2→1`、`4→2`、`8→3`，之后 `n=8` 稳定在 3 不再涨（计数器是进程累计值——同一会话先跑过 §4.1 会整体 +1，模式不变）。两个教学点：

1. **计数器必须走 `torch.compile` 路径才会填充**——`dynamo.run(g)` 在 2.5.1 下不 populate `counters["stats"]`（恒 0），这正是 MIGRATION §8.1 用 `backend="eager"` 计数的原因（dynamo 层指标与 backend 无关，但入口必须是 `torch.compile`）。
2. **每个不同的 shape[0] 都触发一次重编译**：默认 `dynamic=False` 下 dynamo 按形状特化生成 guard，形状一变 guard 即失败 → 重编译（+1）——与分支翻不翻面无关。真正稳定的是**同形状重复步进计数不增长**，这就是 MIGRATION §8.1 对向量化 MoE 的验证语义（`tests/test_moe_vectorized.py::test_compile_no_recompile_growth`：`backend="eager"` 下 100 步 `unique_graphs` 稳定）。想避免形状诱导的重编译需 `dynamic=True` 或 `torch._dynamo.mark_dynamic`（§3.2 的 generate 用 `dynamic=True` 正是此理）。

### 4.3 fullgraph 约束感受一下

```python
import torch

def h(x):
    if x.sum() > 0:          # 数据依赖分支：dynamo 无法静态化
        return x + 1
    return x - 1

torch._dynamo.reset()
try:
    torch.compile(h, fullgraph=True, backend="eager")(torch.randn(4))
except torch._dynamo.exc.TorchDynamoException as e:
    print("graph break:", e)
```

`_generate_fullgraph` 用「删除分支 + torch.where 掩码」满足的正是这条约束（§3.3）。

---

## 5. Inductor 层：理论与本机断裂案例

### 5.1 本机为什么断了

```
torch.compile(model)  →  Inductor 编译时报：
cannot import name 'triton_key' from 'triton.runtime'
```

机理：**inductor 与 triton 强耦合**——torch 每个版本的 inductor 针对一个**特定的 triton 版本**开发与测试（torch 2.5.1 需 triton 3.1.x）。本机是 triton 3.8.0（为了手写 kernel 的新特性），inductor 的导入路径直接断裂。这不是 bug，是版本矩阵的现实：

| torch | 配套 triton（要求） | 本机实际 |
|-------|--------------------|----------|
| 2.5.1 | 3.1.x | **3.8.0 ✗ 断裂** |
| 2.6.x | 3.2.x | — |
| 2.7.x | 3.3.x | — |

**为什么不修**（用户裁定）：降级 triton 3.1 会破坏手写 kernel 依赖（`tools/infra/triton_kernels.py`），Windows 也没有官方 triton 3.1 wheel。于是分层处理：**dynamo 层本机全可用**（backend="eager"/explain，§4），**inductor 层走理论 + 云端 SOP**（§7）。教训：编译栈是「torch + triton + GPU 架构」三元版本矩阵，升级任何一个都要看另外两个。

### 5.2 fusion 理论要点（无 inductor 也能讲清楚的部分）

- **逐元素链可无条件合并**：`y = (a+b)*c` 的两次读中间结果 → 一个 kernel 一遍过。激活函数、缩放、掩码乘法全是这一类。
- **归约是融合边界**：mean/sum 需要完整输入才能出结果，只能与「喂给它的一方」融合（RMSNorm = mean + 逐元素缩放，可以一体——这正是 Triton 微实验里手写 RMSNorm kernel 做的事）。
- **GEMM 一般独立成 kernel**：cuBLAS/Triton GEMM 模板成熟，fusion 收益小；inductor 的 max-autotune 才会尝试 GEMM 重写。
- 呼应 §3.4：**静态形状让 guard 永不失效、fusion 计划可复用**——形状一变，fusion 产物全部作废重编译。

---

## 6. 实测数据解读：什么时候值得 compile

本仓两组 decode 实测（EXPERIMENTS §6.4，12 层 modern-full，单条解码，3 次重复）：

| 场景 | eager | compile | 结论 |
|------|-------|---------|------|
| prompt 128 / 64 新 token | 184 tok/s | 79 tok/s | **负结果**：编译 + 首步图捕获开销摊不平 64 步 |
| 64 新 token | 182 tok/s | 94 tok/s | 仍负，但差距收窄 |
| **256 新 token** | 185 tok/s | 294 tok/s | **约 1.59×**：编译成本被摊平，fusion 收益显现 |

（溯源：EXPERIMENTS.md §6.4 负结果清单；`docs/results/inference.json` compile 78.99 vs eager 184.06 tok/s。）

**分层归因**：compile 的固定成本 = 首次图捕获 +（reduce-overhead）首次 CUDA Graph 录制，约在数百 ms 量级；收益端 = 每步省下的 launch + fusion 带宽。**摊平边界 = 固定成本 / 每步净收益**——本机这个模型大约在 64~256 新 token 之间。所以：

- 长生成 / 形状稳定的高频小步 → 值得（`_generate_fullgraph` 正是为这个场景设计）；
- 一次性短解码 → 不值得（本仓 `generate(compile=...)` 默认关闭）；
- 训练态 → 步数以千计，摊平必然发生，值得关心的是 recompile（§4.2）与 FSDP 组合序（§3.1）。

**能说 / 不能说**：单卡 sm75、12 层小模型、单条解码的数据；多卡、大 batch、云 GPU 的边界不外推。

---

## 7. 云端 E-F-4：inductor 全链路验证 SOP

**环境**（见 RUNNING_EXPERIMENTS.md E-F-4 条目）：云 Linux + 官方配套 torch/triton（如 torch 2.5.1 + triton 3.1.0）+ CUDA GPU。

**命令**：对齐 F 组 SOP 格式——① pip 装配套 triton；② 复跑 `tests/test_moe_vectorized.py::test_compile_no_recompile_growth`（inductor 后端下）；③ `generate(compile='fullgraph')` 跑 256 token 解码并计时；④ 与本机 dynamo 层数据（§6 表）对照。

**产物**：`docs/results/` 新增云端 JSON（compile/eager 吞吐 + recompile 计数 + torch/triton 版本）。

**判据**：inductor 下 recompile 计数与 dynamo 层一致（图行为不因 backend 改变）；256 token 编译路径 ≥ eager 吞吐（摊平后应见正收益）；64 token 允许仍为负（与本机边界一致）。

---

## 8. 自测要点

| 验证点 | 位置 |
|--------|------|
| GPU 执行模型参数可复算（峰值/occupancy/带宽台阶） | §0.1/§0.2/§0.6，roofline.json |
| MoE 向量化不触发重编译 | `tests/test_moe_vectorized.py::test_compile_no_recompile_growth` |
| compile 回退安全（异常→警告→eager） | modern_gpt.py:2090-2098、L2333-2344 |
| fullgraph 前置条件收紧 | modern_gpt.py:2052-2060（`can_fullgraph`） |
| FSDP/compile 组合序与 raw_model 语义 | trainer_base.py:743-779 |

三个思考题：

0. decode 单步有效带宽 4.81 GB/s 距 spec 448 GB/s 有 93×，为什么 §0.6 判定「已解释」而不是「硬件坏了」？（提示：gap 的量级指向哪一段开销，§1 三段式。）
1. `generate(compile=True)` 用 `fullgraph=False`，`_generate_fullgraph` 用 `fullgraph=True`——同一个模型为什么前者不能整图？（提示：循环体 vs 单步函数，§3.2/§3.3。）
2. DDP 分支里 `static_graph=True` 对 compile 意味着什么？（提示：桶结构与图形状假设，§3.1。）
3. 本机 64 token 是负结果，为什么本仓还保留 `_generate_fullgraph`？（提示：固定成本摊平边界随步数移动，§6。）

---

*维护约定：dynamo 层实验代码保持「复制即跑」；行号锚点随重构同步更新；云端 E-F-4 执行后回填 §7 状态与数据。*
