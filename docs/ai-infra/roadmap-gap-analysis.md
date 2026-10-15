# AI Infra 学习路径缺口分析（2026-10-14）

> 背景：AR-v0.7.5 归档后，对整个学习路径做一次系统性盘点（方法：主题×子领域矩阵 +
> 未利用代码资产清点 + 实验矩阵缺口扫描，三视角交叉），回答「针对学习 AI infra 的
> 需要，还应添加哪些内容」。本文档是后续 AR（v0.8.5 起）srs 范围的规划输入。
> 所有「本仓锚点」均经 Glob/Grep 验证存在。

## 一、现有覆盖盘点（摘要）

主题 1/2/3/8/9/10 已交付（手册+工具+微实验+EXPERIMENTS 记录节），主题 4/5/6/7 计划中
（v0.8.0/v0.9.0）。已交付主题的子领域覆盖见各手册章节结构；两点结构性事实：

1. **资产-主题失衡**：evaluation/（win_rate Judge 抽象、benchmark_suite lm-eval 接入、
   needle 网格、eval_alignment、measure_group_variance）+ utils/monitoring.py（五监视器
   含心跳看门狗）+ utils/checkpoint.py（RNG 状态/stage_to_cpu）是仓内资产最厚的一批
   代码，但主题地图无任何归属主题。
2. **主题 10 覆盖 < 仓内数据模块的 3/4**：data-pipeline.md 只讲了单语 web 语料主链，
   multilingual.py（LanguageDetector+可 resume 数据集）/ code.py（py/js/cpp 语法过滤）/
   mixer.py（MixtureStrategy 加权采样+state_dict resume）/ chat_templates.py（ChatML/
   Llama2/Gemma + build_kv_cache_prefix）四模块 grep 0 命中、完全未进手册。

## 二、三大缺口

### 缺口 ①：实验矩阵滞后于代码现状

| 项 | 现状 | 处置 |
|----|------|------|
| E-B-4 prefix cache 假设失效 | continuous_batching.py docstring 明言 prefix-cache 类已删，但实验行仍写「Prefix Cache 命中时超线性」 | 用 chat_templates.py:54/265（build_kv_cache_prefix / inject_system_prompt_kv_cache——仓内唯一现成 prefix 复用半成品）做 system-prompt KV 前缀复用实验补回；v0.9.0 交付前小 AR |
| MoE capacity factor 机制完整无实验 | modern_gpt.py:1007-1031（aux loss + Switch 式 capacity 溢出丢 token）+ test_moe_load_balance.py 在位，E-A-5 只看 loss | **v0.8.5 增补**：capacity 0.5/1.0/1.25 溢出率 + router 负载熵微实验（CPU 可跑，E-C-3 本机 2x 先例） |
| MTP 被遗漏 | modern_gpt.py:1307-1396（MTP 头+mtp_weight）+ test_mtp.py 在位，主题地图/实验矩阵均未提 | 主题 6（v0.9.0）或提前：draft-model vs MTP 两条自加速路线对照 |
| needle 网格现成未登记 | evaluation/needle_in_haystack.py（depth×context 网格）183 行闲置 | **v0.8.5 增补**：长上下文 AR 的检索实验锚点 |
| F 组四项未执行 | E-F-1/2/3/4 均需云端 GPU（NCCL 带宽/全量 roofline/KV 物化/inductor） | 待云端资源；E-F-1 内联脚本先固化为 tools/infra/bench_allreduce.py |
| 量化实验只有单点 | E-B-2 KV int8 短上下文单点；quantization.py:245/277（estimate_quantized_size/compute_quantization_mse）+ gguf_utils.py Q8_0 闲置 | v0.9.0 前置：新增 E-G 量化组（per-channel vs per-tensor MSE 扫描、GGUF 真实体积 vs 公式闭合——纯 CPU 可跑） |

### 缺口 ②：主题级空白——评测系统工程

评测的「指标可信度」本身值得成为学习对象：Judge 抽象（RuleJudge vs
LengthPenaltyJudge 一致性）、胜率口径、benchmark 污染-洁净对照、E-D-3「奖励升、
准确率不升」的指标解耦教学案例。**建议新增主题 11「评测系统工程」**，交付 AR 排
v0.9.0 之前（E-D 数据已齐可直接复用）。锚点：evaluation/ 全部四模块 +
utils/experiment.py + EXPERIMENTS.md §6.5。

### 缺口 ③：主题 10 数据构成增补

multilingual/code/mixer/chat_templates 四模块进 data-pipeline.md 新一节「语料构成」
（配比混合的期望 vs 实际采样频率对照、语法过滤通过率、ChatML/Llama2 token 开销、
mixer state_dict 断点恢复采样一致性——纯 CPU 微实验齐全）。chat_templates 的
system-prompt KV 前缀能力移交主题 6 服务化。

## 三、深化并入既有主题（不扩编地图）

| 方向 | 落点 | 要点 |
|------|------|------|
| GPU 执行模型（SM/warp/occupancy/SMEM-HBM/cp.async/TMA） | 主题 3 compile-stack.md 新前置章 | 读懂 triton_kernels.py BLOCK/num_warps 与 inductor fusion 决策的前提；用 roofline --quick 实测点反推理论峰值算术闭合 |
| 通信原语族（all-reduce/all-gather/reduce-scatter 通信量 + ZeRO stage 映射） | 主题 4 一节 | E-F-1 脚本固化 tools/infra/bench_allreduce.py；纯 CPU ZeRO 各 stage 通信量/显存算术表（infra_math 扩展） |
| TP/PP/CP 并行谱系 | 主题 4「本仓为什么没有 TP/PP」节 | 50M 规模不经济的诚实报告；ring_attention.py 作 CP 通信-计算重叠锚点 |
| KV 显存家族矩阵（MHA/GQA/MQA/MLA 的 B/token 全表） | **v0.8.5** | infra_math gqa_kv_materialization_slope 加 MQA 档+MLA 理论档（无代码，明标「无实验」） |
| checkpoint 心跳/RNG 一致性 | 主题 5 | stage_to_cpu、get/set_rng_state、MetricsHeartbeatWatchdog；断点前后 loss 无缝衔接微实验 |
| 可观测性 | 拆归主题 4/5 | MFU/trace 归 4；LossSpike/心跳归 5——不独立成主题 |
| 量化总论（W8A8/fp8/KV 量化/GPTQ/AWQ） | v0.9.0（既定） | E-G 实验组前置铺垫；GGUF/ONNX/HF 导出生态一并覆盖 |

## 四、建议新增内容总表（优先级序）

| # | 内容 | 主题/AR | 关键锚点 | 微实验 |
|---|------|---------|----------|--------|
| 1 | 评测系统工程 | **新主题 11**，v0.9.0 前 | evaluation/ 四模块 + utils/experiment.py | Judge 一致性、污染-洁净对照 |
| 2 | GPU 执行模型前置章 | 主题 3，随 v0.8.5 | triton_kernels.py + roofline.json | 峰值算术闭合 |
| 3 | prefix 复用 + E-B-4 修复 | 主题 6 前置小 AR | chat_templates.py:54/265 | N 请求 prefill 节省量 |
| 4 | 量化消融 E-G 组 | v0.9.0 前置 | quantization.py:245/277 + gguf_utils.py | MSE 网格 + 体积闭合 |
| 5 | 数据构成增补节 | 主题 10 小 AR | multilingual/code/mixer/chat_templates | 采样频率卡方 + resume 一致性 |
| 6 | KV 显存家族矩阵 | **v0.8.5** | infra_math:280 + needle_in_haystack.py | B/token 全表 + needle 网格 |
| 7 | MoE 路由容量微实验 | **v0.8.5** | modern_gpt.py:1007-1031 | capacity 扫描 + 负载熵 |
| 8 | 通信原语 + E-F-1 固化 | 主题 4（v0.8.0） | wrap_distributed + DDP/FSDP 测试 | ZeRO stage 算术表 + gloo 基线 |
| 9 | checkpoint 心跳深化 | 主题 5（v0.8.0） | utils/checkpoint.py + monitoring.py | RNG 一致性 + 落盘时间对比 |
| 10 | MTP 与 draft-free 投机 | 主题 6 或提前 | modern_gpt.py:1307-1396 + test_mtp.py | mtp_weight 消融 |

## 五、对路线图的影响

- **AR-v0.8.5（下一个）**：MoE 系统工程 + 长上下文，范围吸收本表 #6/#7（KV 家族
  矩阵、MoE 容量微实验）与 #2（GPU 执行模型前置章）；
- v0.8.5 与 v0.8.0 之间可插 **#5（数据构成小 AR）** 与 **#3（prefix 小 AR）**；
- **主题 11 评测工程** 建议独立 AR，排在 v0.9.0 之前；
- 主题地图保持 10 主题不扩编（深化并入优于横向铺开），仅新增主题 11。
