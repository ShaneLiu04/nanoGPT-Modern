# 数值精度与混精深潜手册（Numerical Precision & Mixed Precision: fp32 / bf16 / fp16 / tf32 / fp8）

> 主题 8｜AR-v0.7.5 交付。配套微实验工具：`python tools/infra/precision_probe.py`（纯 CPU，
> 产物 `docs/results/precision_probe.json`）。本手册所有本仓数值均可溯源：system.json /
> memory-flops-arithmetic.md §2.1 / triton_kernels.json / tests/test_pretrain_precision.py。

## 1. 为什么混精：本仓 E-C 实测开场

先看结论数据（E-C 对照实验，`docs/results/system.json`）：

| 精度 | 吞吐（tokens/s） | 相对 fp32 |
| --- | --- | --- |
| fp32 | 70,210.75 | 1.00× |
| bf16（autocast） | 147,727.89 | **2.10×** |

同一模型、同一数据、同一迭代数，只把 matmul 类计算从 fp32 换成 bf16，吞吐翻倍。为什么
「减精度」能换速度？因为现代加速器的算力按「每秒多少次低精度乘加」标定——tensor core 的
bf16 峰值通常是 fp32 的 2-8×，而带宽压力同时减半（2 B vs 4 B/元素）。精度不是道德问题，
是**性价比问题**：在「不伤 loss 收敛」的前提下，用最便宜的精度完成最多的 FLOPs。

但这句话里藏着的全是工程细节：哪些算子能降、降到什么格式、累加用什么精度、梯度太小怎么办
——这就是本手册的主题。

## 2. 四精度表示学：位布局、ulp 与 RNE

### 2.1 位布局与关键数字

IEEE 754 浮点数 = 符号位 + 指数位 + 尾数位。**指数位决定范围，尾数位决定相对精度**：

| 格式 | 符号 | 指数 | 尾数 | max | 最小正规数 | eps（ulp@1） | torch dtype |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fp32 | 1 | 8 | 23 | 3.40e38 | 1.18e-38 | 1.19e-7（2^-23） | `torch.float32` |
| tf32 | 1 | 8 | **10** | 3.40e38 | 1.18e-38 | 9.77e-4（2^-10） | 无独立 dtype（见 §2.4） |
| bf16 | 1 | 8 | **7** | 3.40e38 | 1.18e-38 | 7.81e-3（2^-7） | `torch.bfloat16` |
| fp16 | 1 | 5 | **10** | 65,504 | 6.10e-5 | 9.77e-4（2^-10） | `torch.float16` |

三个关键观察（`precision_probe.py` 的 `dtype_limits` 实验用 `torch.finfo` 逐项复算）：

1. **bf16 = fp32 的指数 + fp16 的尾数砍半**：指数位与 fp32 完全相同（8 位），所以表示范围
   与 fp32 一样大；牺牲的是尾数（7 vs 23），相对精度约 2^-7≈0.8%。
2. **fp16 范围是瓶颈**：指数只有 5 位，max=65,504。深度学习里激活值轻松超过这个数
   （尤其 loss 放大、attention scores 未经 scale 时），**溢出即 inf 即 NaN 传播**——这是
   fp16 需要 GradScaler 的根因（§3.4）。
3. **bf16 为什么稳**：范围与 fp32 相同，训练中的任何值都不会因格式溢出；0.8% 的相对
   误差对梯度下降是噪声级——随机舍入下期望无偏，SGD 本身就带噪声。

### 2.2 ulp：舍入误差的最小货币

ulp（unit in the last place）= 相邻可表示数的间距。规格化区间 [2^e, 2^(e+1) 内 ulp 恒定
= 2^(e-尾数位)。**每个数被舍入到最近的可表示数，误差 ≤ ulp/2**——这就是「相对误差约
2^-(尾数位+1)」的来源。

手算例（bf16）：区间 [1, 2) 内 ulp = 2^-7 = 0.0078125。十进制 1.001 落在可表示数
1.0000 与 1.0078125 之间，按最近原则舍到 1.0000，绝对误差 ≈0.001 < ulp/2。而区间
[256, 512) 内 ulp = 2^-7 × 256 = 2——**离原点越远，ulp 越大，绝对误差越大，相对误差
不变**。这就是为什么混精训练要防「大数吃小数」：梯度累加里 1e-3 量级的更新遇到 1e2
量级的累加器，若累加器也是低精度，更新直接被 ulp 吞掉（§4.1 实测）。

### 2.3 RNE：为什么舍入到偶数

IEEE 754 默认 round-to-nearest-even：恰在中点时舍到尾数为偶数的一方。中点舍入若恒进位
（round-half-up），误差期望非零，会随累加次数系统性漂移；RNE 让中点上下各半，**期望
无偏**。构造案例（`precision_probe.py` 的 `rne_rounding` 实验）：fp16 中 2048 与 2050
的中点 2049 不可表示（区间 [2048, 4096) 内 ulp=2），RNE 落到 2048（偶尾数）而非 2050；
批量统计 tie 案例 tie_to_even 命中率应 = 1.0。

### 2.4 tf32 是什么：不是存储格式，是计算模式

tf32（NVIDIA Ampere+）只在 **matmul 内部**生效：输入仍是 fp32 存储，tensor core 读入时
截成 1+8+10 位参与乘加。所以它**不省显存**（参数还是 4 B），只省计算时间——精度相当于
fp16 的尾数 + fp32 的范围。本仓开关在 `training/trainer_base.py:49`：

```python
def enable_tf32() -> None:
    """Enable TF32 for CUDA matmul and cuDNN (A100/H100 fast path).

    Idempotent and safe on CPU-only builds — the backend flags are
    accepted regardless of CUDA availability, so all trainer scripts
    call this at module level (FR8-AC3: TF32 coverage across all
    trainers, previously pretrain-only).
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

注意设计取舍：tf32 对 fp32 训练是「免费加速但精度降级」，对需要严格复现的对照实验是
污染源——所以它作为显式开关存在，而不是默认全局打开。

## 3. AMP 机理：autocast 白名单 + GradScaler 循环

### 3.1 autocast 的语义：算子级降精度，不是全局降

`torch.amp.autocast` 是上下文管理器：进入后，**注册在白名单里的算子**自动把输入转成
低精度计算，黑名单算子强制 fp32。典型分工：

- **降精度**（matmul/conv/线性层）：受 tensor core 加速，误差噪声级；
- **保持 fp32**（softmax、layer norm、loss、reduction）：这些算子对精度敏感——softmax
  里 exp 溢出、norm 里均值/方差累加、loss 里跨 batch 求和，都是 §2.2 说的「ulp 累加」
  重灾区。

关键点：**参数（weights）始终保持 fp32 存储**。autocast 只在算子执行时临时转 bf16，
前向的 bf16 激活被保存用于反向，反向算出的梯度是 bf16，落回 fp32 参数的 `.grad` 时
自动升回 fp32。这就是「fp32 master weights」的 autocast 版实现（§4.2 是另一种方案）。

### 3.2 与本仓逐行对照：build_amp_context

`training/trainer_base.py:208`（全文 20 行，逐条注释）：

```python
def build_amp_context(device, use_bf16=True):
    if device == "cpu":
        return nullcontext(), None, torch.float32      # ① CPU 不混精：fp32 语义完整

    if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16                          # ② 首选 bf16（范围安全，无需 scaler）
    else:
        dtype = torch.float16                           # ③ 退路 fp16（老卡），必须配 scaler

    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    # fp16 needs GradScaler; bf16 does not.
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    return ctx, scaler, dtype                           # ④ scaler 只在 fp16 路径启用
```

四条设计决策：①CPU 路径显式短路，避免在无 tensor core 的设备上白付精度损失；②bf16
支持检测在前，**能用 bf16 绝不用 fp16**（§2.1 的范围论证）；③fp16 是硬件代际的退路而
非选择；④`GradScaler(enabled=...)` 构造而非分支构造——同一份训练代码无 if 地兼容两种
格式。配套测试 `tests/test_pretrain_precision.py` 4 用例固化了这套契约：
`--precision fp32` 显式关 AMP、默认走 AMP setup、fp16 分支在 CUDA 上启用 scaler、
fp32 分支在 CUDA 上短路。

### 3.3 混精反向传播的完整链路

以 bf16 autocast 一次迭代为例：

1. 前向：weights(fp32) → 算子入口转 bf16 → matmul(bf16) → 激活 bf16；norm/softmax
   强制 fp32；
2. 反向：链式求导在 bf16 域计算，梯度落回 fp32 `.grad`；
3. 优化器：fp32 参数 += lr × fp32 梯度（**更新精度是 fp32**，§4.1 论证为什么必须如此）。

### 3.4 GradScaler：fp16 的救命稻草，五步循环

fp16 的最小正规数是 6.10e-5，而典型梯度量级 1e-3 ~ 1e-8——小于 2^-24 的数直接 flush
to zero（**下溢**），梯度静默消失，loss 曲线看着正常但模型学不到东西。GradScaler 的
方案：**放大 loss 让梯度整体抬进可表示区间，更新前再除回去**：

```text
① scale = 2^16（初始，动态调整）
② loss_scaled = loss × scale → backward → 梯度全部 × scale（不再下溢）
③ unscale: optimizer 检查 .grad / scale 是否出现 inf/nan
④ 有 inf → skip 本次 step，scale 减半，下一步重试（infskip）
⑤ 无 inf → 正常 step；连续成功则周期性上调 scale（backoff 机制）
```

为什么 bf16 不需要（`build_amp_context` 里 `enabled=(dtype == torch.float16)`）：bf16
最小正规数 1.18e-38 与 fp32 相同，任何 fp32 能表示的梯度 bf16 都装得下，不存在下溢
问题。**fp16 的病在指数，bf16 的药也在指数**。

`precision_probe.py` 的 `loss_scaling_simulation` 实验在纯 CPU 上复现这条链路：构造
1e-8 量级梯度 → fp16 直接表示为 0（zero_fraction 统计）→ 乘 scale=65536 后恢复 →
unscale 检测 inf 语义。这是 GradScaler 的最小可复现版本。

## 4. 累加精度：大数吃小数与 fp32 master weights

### 4.1 为什么更新必须在 fp32 域

设参数 θ ≈ 1.0 量级，学习率更新 Δθ ≈ 1e-4 量级（lr×grad）。若参数本身存成 bf16：
θ=1.0 处 ulp = 2^-7 ≈ 7.8e-3，**Δθ < ulp/2，加法结果被舍回原值——更新彻底丢失**。
迭代几万步的累积效应就是「训练在原地踏步」。所以混精训练的铁律：**表示精度（bf16，
省带宽/算力）与更新精度（fp32，保更新存活）必须分离**——这就是 master weights。

两种实现：

| 方案 | 参数存储 | 本仓锚点 | 静态显存账（训练态/参数） |
| --- | --- | --- | --- |
| AMP autocast（本仓默认） | fp32（参数即主权重） | `build_amp_context`；memory-flops §2.1 | **16 B/param**（4 参数+4 梯度+8 Adam 状态） |
| bf16-pure（业界另一派） | bf16 + 独立 fp32 master | `--precision bf16-pure` | **12 B/param**（2+2+8） |

两口径的适用条件：autocast 省心（框架托管）、显存贵 4 B/param；bf16-pure 省显存但
master 同步逻辑自管（大模型训练的主流选择）。memory-flops-arithmetic.md §2.1 的完整
表格按此推导，50M 模型 autocast 口径静态账 = 59.56M × 16 B = **0.95 GB**。

### 4.2 matmul 内部：低精度乘，高精度累加

混精的另一半在算子内部：tensor core 做 **bf16 输入 × fp32 累加**。K 维度上每一步部分
和若用 bf16 累加，误差按 ulp 逐步吞噬小部分和（§2.2）；fp32 累加器把 2^-7 的乘法误差
保进 2^-23 的账本。本仓 Triton kernel 全部显式声明这条口径（`tools/infra/triton_kernels.py`）：

```python
# L162-163（swiglu 分块）：acc_g += tl.dot(x_blk, tl.trans(wg_blk), input_precision="ieee")
# L247（attention scores）：s_blk = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee") * scale
# L254（PV 累加）：acc = acc * alpha[:, None] + tl.dot(p, v_blk, input_precision="ieee")
```

`input_precision="ieee"`：Triton 的 dot 默认允许 tf32（Ampere 上），显式 ieee 强制
fp32 精度路径——教学 kernel 要的是**数值可解释**，快慢是 roofline 手册的事。
`precision_probe.py` 的 `accumulation_precision` 实验在纯 CPU 复现这条账：同一组
bf16 权重/激活，K=4096 的 matmul 分别用 fp32 累加器与 bf16 累加器分块求和，对照
相对误差（fp32 累加应显著小于 bf16 累加）。

## 5. 本仓口径交叉：一张表核对所有容差

| 口径 | 数值 | 溯源 |
| --- | --- | --- |
| E-C bf16 加速比 | 147,727.89 / 70,210.75 = **2.10×** | system.json（本手册 §1） |
| 训练静态显存（autocast） | **16 B/param**；50M → 0.95 GB | memory-flops §2.1 |
| bf16-pure 静态显存 | **12 B/param** | memory-flops §2.1 |
| Triton bf16 kernel 验证容差 | atol=0.01, rtol=0.01 | triton_kernels.json（9 组） |
| kernel 内累加精度 | fp32 累加 + `input_precision="ieee"` | triton_kernels.py L162/163/247/254 |
| tf32 开关 | `enable_tf32()`（matmul+cuDNN） | trainer_base.py:49 |
| AMP 上下文 | bf16 首选 / fp16 退路 + GradScaler | trainer_base.py:208 |
| 精度开关契约 | 4 用例（fp32 显式关/默认 AMP/fp16 scaler/CUDA 短路） | tests/test_pretrain_precision.py |

**为什么 Triton bf16 容差是 1e-2**：bf16 相对精度 2^-7 ≈ 0.78%，加上 K=4096 累加的
误差传播，atol/rtol=1e-2 是「bf16 数值语义下的机器精度级」容差——比它紧就是测 flake，
比它松就是放过真 bug。`precision_probe.py` 把这个口径写进产物 meta（`tolerance_refs`），
实验统计量与它对照。

## 6. fp8 理论展望（无实验，验证归 v0.9.0 量化总论）

fp8 两种格式（OCP FP8 标准）：

| 格式 | 指数 | 尾数 | max | eps | 定位 |
| --- | --- | --- | --- | --- | --- |
| E4M3 | 4 | 3 | 448 | 2^-3 = 0.125 | 权重/激活（前向） |
| E5M2 | 5 | 2 | 57,344 | 2^-2 = 0.25 | 梯度（范围优先） |

尾数只剩 2-3 位（相对精度 12.5%-25%），裸用的误差不可接受，所以 fp8 的核心配套是
**scaling**：per-tensor / per-row（行级 scale 把动态范围归一到 fp8 的窗口内），
transformation（如 DelayedScaling 在训练中动态统计 amax）。为什么**推理先行、训练
前沿**：推理是纯前向，权重静态可离线校准 scale（per-channel 精度高），激活范围可控；
训练里梯度分布漂移、scale 动态调整的失败模式（溢出/下溢震荡）是工程难点，且累加器、
优化器状态仍需 fp32——省的是带宽与 matmul，复杂度全在尺度管理。本仓
`quantize_model`（量化）无生产调用方，量化总论（含 fp8 fake-quant 实验）归 v0.9.0
主题 6，本节只立理论坐标。

## 7. 能说 / 不能说

**能说**：
- bf16 为什么稳：8 位指数=fp32 范围，溢出免疫；0.8% 相对精度对 SGD 是噪声级；
- fp16 为什么需要 scaler：5 位指数 → max 65,504/最小正规 6.1e-5，梯度下溢是静默失败，
  loss 放大+unscale 检 inf 是完整闭环；
- 混精省的是 matmul/带宽，不是存储：master weights 必须留在 fp32（16B/param autocast
  / 12B/param bf16-pure），省的是激活与计算；
- 本仓实测 bf16 2.10×、容差 1e-2、kernel ieee 累加——每个数字有出处。

**不能说**：
- 不能说「bf16 训练无损」：0.8% 相对误差对 sensitive 求和（大 K 累加、loss reduce）
  是有偏的，所以 autocast 黑名单强制 fp32；
- 不能从本手册推断 fp8 可直接用于训练（无实验验证，v0.9.0 前是理论）；
- E-C 的 2.10× 是**本仓 50M 模型 + 该 GPU** 的实测，不能外推到任意模型/卡（算力比
  上限才是上界，受 memory-bound 程度调制——见 memory-flops 手册 §3 roofline）。

## 8. 自测要点

1. bf16 与 fp16 各多少位指数/尾数？哪个会溢出、哪个不会，为什么？
2. ulp(1.0) 在 fp32/bf16/fp16 各是多少？区间 [256,512) 的 bf16 ulp 呢？（答：2^-7×256=2）
3. GradScaler 五步循环说出三步以上；bf16 为什么不需要？
4. autocast 口径 16 B/param 的账怎么算？bf16-pure 12 B 省在哪、代价是什么？
5. 运行 `python tools/infra/precision_probe.py`，对照 `precision_probe.json`：你的
   ulp/RNE/累加误差手算和工具输出一致吗？
6. 为什么 Triton kernel 容差是 1e-2 而不是 1e-6？（答：bf16 机器精度 2^-7≈0.78%）
