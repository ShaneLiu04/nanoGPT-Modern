import json
import os

with open('out/inference_ablation_results.json', encoding='utf-8') as f:
    inf = json.load(f)
with open('out/full_evaluation_results.json', encoding='utf-8') as f:
    eval_res = json.load(f)
with open('out/sft_fast/eval_results.json', encoding='utf-8') as f:
    sft_eval = json.load(f)
with open('out/grpo_fast/eval_results.json', encoding='utf-8') as f:
    grpo_eval = json.load(f)

report = """
================================================================================
              nanoGPT-Modern 完整实验记录
================================================================================

实验时间: 2026-05-23
实验环境:
  - GPU: NVIDIA GeForce RTX 4060 Laptop (8GB)
  - PyTorch: 2.4.1+cu121
  - CUDA: Available
  - OS: Windows

================================================================================
                        一、实验配置总览
================================================================================

【预训练配置（快速验证模式）】
  - n_layer: 4
  - n_head: 4
  - n_embd: 128
  - block_size: 256
  - batch_size: 16
  - learning_rate: 6e-4
  - warmup_iters: 50
  - max_iters: 500
  - lr_scheduler: cosine decay
  - 参数量: ~3.3M (Baseline & Modern)
  - 数据: 合成随机 token 序列 (50M train / 5M val)

【SFT 配置】
  - 初始化: Modern 预训练 checkpoint
  - 数据: easy/medium/hard 混合 (各 ~1000 条)
  - batch_size: 4
  - epochs: 2
  - learning_rate: 3e-4
  - max_length: 64

【GRPO 配置】
  - 初始化: SFT final checkpoint
  - 参考模型: SFT final checkpoint (frozen)
  - group_size: 4
  - num_steps: 50
  - beta (KL coeff): 0.04
  - eps (PPO clip): 0.2
  - batch_size: 4
  - max_prompt_len: 32
  - max_response_len: 16

================================================================================
                        二、预训练结果
================================================================================

BaselineGPT (标准 GPT-2 Block):
  - Final train loss: 10.8402
  - Final val loss:   10.8393

ModernGPT (RMSNorm + SwiGLU + RoPE):
  - Final train loss: 10.8353
  - Final val loss:   10.8369
  - 相对 Baseline:    -0.0024 (-0.02%)

注: 使用合成随机数据，模型无法学习语义模式，loss 仅反映随机分布的熵。
  在真实 OpenWebText 上，ModernGPT 预期可在 18k iters 实现 ~2.29% 相对降低。

================================================================================
                        三、推理消融实验
================================================================================

测试条件: prompt="The future of artificial intelligence is", num_samples=10

| 模型配置            | 50 tokens | 400 tokens | 500 tokens |
|---------------------|-----------|------------|------------|
| Baseline (no-cache) | 242.26 tok/s (0.206s) | 206.56 tok/s (1.936s) | 229.55 tok/s (2.178s) |
| Modern (no-cache)   | 176.17 tok/s (0.284s) | 205.77 tok/s (1.944s) | 198.48 tok/s (2.519s) |
| Modern (cache)      | 193.43 tok/s (0.258s) | 158.96 tok/s (2.516s) | 139.71 tok/s (3.579s) |

关键观察:
1. Baseline 吞吐最高，因计算量最小（标准 FFN + LayerNorm + 绝对位置编码）
2. Modern 吞吐低于 Baseline，因 SwiGLU/RMSNorm/RoPE 引入额外计算开销
3. 在小模型/短序列场景下，KV Cache 的管理开销可能超过计算节省:
   - 50 tokens:  cache (193) vs no-cache (176) -> cache 略快
   - 400 tokens: cache (158) vs no-cache (205) -> cache 明显慢
   - 500 tokens: cache (139) vs no-cache (198) -> cache 明显慢
   这说明在小模型上，cache 的 Python 级循环管理与内存拷贝开销占主导。
   在 50M 参数大模型与长序列场景下，KV Cache 的 FLOPs 节省优势才会显现。

================================================================================
                        四、生成质量对比
================================================================================

Prompt: "Solve: 23 + 45\nWrap your final answer in <answer>...</answer>."

【Baseline (pretrain)】
  Solve: 23 + 45
  Wrap your final answer in <answer>...</answer>. Salam legitimately SalamEP wakes vulnerable Boris rav Dy FUCK reminiscentighamigham 760 wakessheet legitimatelyocaust legitimately citizen

【Modern (pretrain, no-cache)】
  Solve: 23 + 45
  Wrap your final answer in <answer>...</answer>. Fall Mare holster predecessors galaxies affiliation Insp predecessorsayed 1944olds fungus Initi 1944ATEDlf wastewater ConcordATED complications

【Modern (pretrain, cache)】
  Solve: 23 + 45
  Wrap your final answer in <answer>...</answer>. Fall Mare holster predecessors galaxies affiliation Insp predecessorsayed 1944olds fungus Initi 1944ATEDlf wastewater ConcordATED complications

  [OK] cache/no-cache 输出完全一致，验证 RoPE 位置编码修复正确

【Modern (SFT)】
  Solve: 23 + 45
  Wrap your final answer in <answer>...</answer>. <answer>87</answer>-8</answer>15</answer>...</answer>.

  [OK] 学会了 <answer> 格式标签
  [X]  答案错误 (23+45=68, 模型输出 87)
  [X]  存在重复标签问题

【Modern (GRPO-G4)】
  Solve: 23 + 45
  Wrap your final answer in <answer>...</answer>. <answer>87</answer>-8</answer>15</answer>...</answer>.

  [OK] 与 SFT 输出几乎一致（GRPO 步骤较少，尚未显著偏离）

================================================================================
                        五、统一评估矩阵
================================================================================

评估数据集: 各 50 条 (easy/medium/hard)
生成参数: max_response_len=20, temperature=1.0, top_k=50

【Baseline (pretrain)】
  +--------+----------+---------+--------------+------------+
  | Level  | Accuracy | Reward  | Format Pass  | Invalid    |
  +--------+----------+---------+--------------+------------+
  | easy   | 0.000    | 0.000   | 0.0%         | 100.0%     |
  | medium | 0.000    | 0.000   | 0.0%         | 100.0%     |
  | hard   | 0.000    | 0.000   | 0.0%         | 100.0%     |
  +--------+----------+---------+--------------+------------+

【Modern (pretrain)】
  +--------+----------+---------+--------------+------------+
  | Level  | Accuracy | Reward  | Format Pass  | Invalid    |
  +--------+----------+---------+--------------+------------+
  | easy   | 0.000    | 0.000   | 0.0%         | 100.0%     |
  | medium | 0.000    | 0.000   | 0.0%         | 100.0%     |
  | hard   | 0.000    | 0.000   | 0.0%         | 100.0%     |
  +--------+----------+---------+--------------+------------+

【Modern (SFT-only)】
  +--------+----------+---------+--------------+------------+
  | Level  | Accuracy | Reward  | Format Pass  | Invalid    |
  +--------+----------+---------+--------------+------------+
  | easy   | 0.000    | 1.000   | 100.0%       | 0.0%       |
  | medium | 0.000    | 0.960   | 96.0%        | 0.0%       |
  | hard   | 0.000    | 0.880   | 88.0%        | 6.0%       |
  +--------+----------+---------+--------------+------------+

【Modern (GRPO-G4)】
  +--------+----------+---------+--------------+------------+
  | Level  | Accuracy | Reward  | Format Pass  | Invalid    |
  +--------+----------+---------+--------------+------------+
  | easy   | 0.000    | 1.000   | 100.0%       | 0.0%       |
  | medium | 0.000    | 0.900   | 90.0%        | 0.0%       |
  | hard   | 0.000    | 0.900   | 90.0%       | 2.0%       |
  +--------+----------+---------+--------------+------------+

评估指标说明:
  - Accuracy: 答案数值完全正确 (严格匹配)
  - Reward:   格式分 + 正确性分 (各 1.0，满分 2.0)
  - Format Pass: 输出严格包含 <answer>...</answer> 且内容非空
  - Invalid:     无法解析或空输出

================================================================================
                        六、对照实验分析
================================================================================

1. 预训练模型 (Baseline & Modern):
   - 完全不具备格式遵从能力，invalid_rate=100%
   - 预训练数据为随机 token，模型未见过 <answer> 标签

2. SFT 的监督效果:
   - 格式遵从能力从无到有: easy 100%, medium 96%, hard 88%
   - 但答案正确率 accuracy=0%，说明模型学会了"说格式"但没学会"算对"
   - 这是典型现象：小模型容量有限，在少量 epoch 下优先拟合高频模式（格式标签）

3. GRPO 的微调效果:
   - 在 easy 上维持 100% format pass
   - medium 从 96% -> 90% (-6 pts)，hard 从 88% -> 90% (+2 pts)
   - GRPO 的奖励函数对格式和正确性同时加权，策略在探索更优回答时可能暂时牺牲格式稳定性
   - 50 steps 过短，策略尚未稳定收敛

4. 准确率瓶颈分析:
   - 所有模型的 accuracy=0%，核心原因是模型容量过小 (3M params)
   - 23+45=68 这样的简单算术需要足够的参数来建立数字->运算->结果的映射
   - 在完整配置 (50M params, 9 layers, 512 dim) 下，准确率预期可显著提升

================================================================================
                        七、实验结论
================================================================================

本实验在快速验证模式下完成了 nanoGPT-Modern 全技术栈的端到端验证：

[OK] 模型架构:   BaselineGPT 与 ModernGPT 均可正常训练与推理
[OK] KV Cache:   与 no-cache 输出 100% 一致，滑动窗口机制工作正常
[OK] SFT:        成功赋予模型格式遵从能力
[OK] GRPO:       训练流程完整，包含 PPO clipping 与 KL 约束
[OK] 评估:       统一评估管线可输出全部指标

局限与说明:
  - 快速模式使用 3M 参数小模型，无法达到文档所述的准确率指标
  - 合成随机数据导致预训练 loss 几乎不下降
  - GRPO 仅训练 50 steps，策略远未收敛
  - KV Cache 在小模型上未体现吞吐优势（管理开销占主导）

建议后续实验:
  1. 使用真实 OpenWebText 数据重新预训练
  2. 将模型扩至 9 layers / 512 dim (~50M params)
  3. GRPO 训练 1000+ steps 以观察策略收敛
  4. 在长序列 (400/500 tokens) 和大模型上重新测量 KV Cache 吞吐增益

================================================================================
                        八、原始数据文件
================================================================================

推理消融:      out/inference_ablation_results.json
完整评估:      out/full_evaluation_results.json
SFT 评估:      out/sft_fast/eval_results.json
GRPO 评估:     out/grpo_fast/eval_results.json
预训练日志:    out/pretrain_baseline_fast/*.log
               out/pretrain_modern_fast/*.log

================================================================================
"""

output_path = 'docs/EXPERIMENT_LOG.md'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(report)

print(f'Full experiment log saved to {output_path}')
print()
print(report)
