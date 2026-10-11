# RUNNING_EXPERIMENTS.md — 实验运行手册

> 定位：**跑实验的唯一入口文档**。规划面向深度对比的实验矩阵、云端服务器 SOP、图表与报告产出规范，并跟踪每个实验的执行状态。
>
> **维护规则（强制）**：每次 AR 归档后必须同步更新本文档——①更新「实验状态跟踪表」中受影响实验的 AR 版本与状态；②若 AR 改变了实验相关代码路径（训练器/推理/缓存/编排器），同步修订对应实验的命令与判读要点；③新增能力（如新组件、新后端）评估是否补充新实验条目。此约定与 tasks.md 一样属于过程管理契约。

---

## 1. 实验矩阵总览

四大组、共 14 个实验。编号 `E-<组>-<序>`，与 EXPERIMENTS.md 章节对应。
所有云端实验固定 `seed 1337`、`out_root out/ablations`，保证与既有记录可比。

### A 组：架构组件消融（预训练收敛对比）

工具：`run_component_ablations.py`（固定 7 组矩阵，断点自动续跑，聚合 JSON+MD 摘要）。

| 编号 | 实验 | 对比项 | 假设 | 图表产出 |
|------|------|--------|------|----------|
| E-A-1 | 双轨总增益 | baseline vs modern-full | 现代组件组合在等参数量下 val loss 更低且收敛更快 | train/val loss 曲线（双线）+ 最终 val loss 条形图 |
| E-A-2 | 位置编码 | modern-full vs no-rope（learned abs） | RoPE 在等训练长度下外推性与损失均占优 | loss 曲线 + 长度外推 PPL 曲线（train 1024 / eval 1024-4096） |
| E-A-3 | FFN 变体 | modern-full vs no-swiglu（GELU 4x） | SwiGLU 8d/3 在等参数预算下损失更低 | loss 曲线 + 吞吐对比 |
| E-A-4 | 注意力分组 | modern-full (GQA-2) vs mha (n_kv=8) | GQA 以小幅损失代价换 75% KV 显存削减 | loss 曲线 + KV 显存条形图 + decode 吞吐曲线 |
| E-A-5 | MoE | modern-full vs moe (top-1, E=4) | 稀疏专家在等计算下损失更低（负载均衡损失可观察） | loss 曲线 + 专家负载分布图 |
| E-A-6 | QK-Norm | modern-full vs qk-norm | QK-Norm 对小模型损失影响中性，主要收益在训练稳定性 | loss 曲线 + 梯度范数分层曲线 |
| E-A-7 | NTK 长度外推 | modern-full checkpoint 评测 | NTK-aware RoPE 在 2-4x 训练长度下 PPL 退化显著小于基线 RoPE | PPL-上下文长度曲线（NTK on/off 双线） |

执行（云端，50M 规格 18k iters，约 7 组 x 4-6h，详见 MIGRATION.md §5）：

```bash
python run_component_ablations.py \
  --out_root out/ablations --stage all \
  --max_iters 18000 --batch_size 32 --device cuda
# E-A-7（外推评测，可独立于矩阵）：
python evaluation/eval_benchmark.py \
  --checkpoint out/ablations/modern-full/best_ckpt.pt \
  --context_lengths 1024 2048 4096 --output_json out/ablations/ntk_eval.json
```

### B 组：推理效率对比（上下文长度扫描）

统一规模：50M checkpoint（A 组产物）、prompt 长度扫描 {512, 1024, 2048, 4096}、
greedy decode、`--num_samples 30`。产出「吞吐-长度」「显存-长度」双曲线族。

| 编号 | 实验 | 对比项 | 假设 | 图表产出 |
|------|------|--------|------|----------|
| E-B-1 | KV Cache 家族 | no-cache vs ring vs append fast-path vs paged | append fast-path 在长上下文下吞吐与显存全面占优（本机已验证显存平坦，见 MIGRATION.md §8.2） | 吞吐-长度 + 峰值显存-长度曲线族 |
| E-B-2 | KV 量化 | fp16 KV vs int8 KV | int8 换 50% KV 显存，吞吐损失 <10%，输出 PPL 退化可接受 | 显存-长度 + PPL 退化条形图 |
| E-B-3 | 投机解码 | greedy vs speculative (draft-len 2/4/8) | 短 draft 在算术类分布下 2-3x 加速，接受率随 draft-len 递减 | 加速比-draft 长度 + 接受率曲线 |
| E-B-4 | 连续批处理 | 逐条串行 vs ContinuousBatchScheduler (1/4/16 并发) | 并发拼批吞吐近线性于并发数（Prefix Cache 命中时超线性） | 吞吐-并发数曲线 + Prefill 共享命中率 |
| E-B-5 | torch.compile | eager vs inductor（云端 triton 可用） | compile 对 50M decode 提速 1.2-1.8x 且无 recompile 增长 | 吞吐条形图 + recompile 计数表 |

执行示例（逐实验类似，完整命令见 `run_ablations.py --mode inference` 与
`tools/profiling/profile_inference.py`）：

```bash
# E-B-1/E-B-2：多长度 x 多 cache 变体
python run_ablations.py --mode inference \
  --checkpoint out/ablations/modern-full/best_ckpt.pt --output_dir out/ablations/inf
# E-B-3：投机解码消融
python inference/generate.py --checkpoint out/ablations/modern-full/best_ckpt.pt \
  --speculative --draft_model out/ablations/modern-full/draft_ckpt.pt \
  --draft_len 2 4 8 --num_samples 30 --output_json out/ablations/spec.json
```

### C 组：训练系统效率（同规模 500 iters 短程对比）

| 编号 | 实验 | 对比项 | 假设 | 图表产出 |
|------|------|--------|------|----------|
| E-C-1 | 混合精度 | fp32 vs AMP-bf16 | bf16 吞吐 +30-80%（Tensor Core），loss 曲线一致性 <1e-3 | tok/s 条形图 + loss 叠加曲线 |
| E-C-2 | 梯度累积 | batch 32 x 1 vs batch 8 x 4 | 等效有效 batch 下收敛曲线重合，吞吐差异 <5% | loss 叠加曲线 + tok/s 对比 |
| E-C-3 | MoE 向量化 kernel | 逐专家 for-loop vs grouped GEMM | GPU 上向量化 2-5x（CPU 已测 2x），专家数越多差距越大 | kernel 耗时-专家数曲线 |
| E-C-4 | 分布式 | 单卡 vs DDP vs FSDP（多卡实例，nccl） | DDP 近线性扩展；FSDP 在 50M 规模下因通信开销不占优（诚实报告分界点） | tok/s-卡数曲线（强扩展）+ 显存对比 |
| E-C-5 | KV 修复回归 | decode 手写 GQA kernel vs SDPA enable_gqa（修复前后） | 修复后峰值显存平坦（本机已验证 12,335 -> ~0 B/token），云端 flash 后端行为对照 | 显存-上下文长度曲线（修复前后 + 前端差异注解） |

### D 组：对齐流水线质量（复用 A 组 checkpoint）

| 编号 | 实验 | 对比项 | 假设 | 图表产出 |
|------|------|--------|------|----------|
| E-D-1 | SFT 格式遵从 | pretrain vs SFT 输出格式合法率 | SFT 后 `<answer>` 格式遵从率 >90% | 遵从率条形图（easy/medium/hard 分组） |
| E-D-2 | GRPO 组大小 | G4 vs G8 vs G16 | 更大组优势估计方差更低，reward 曲线更平稳 | reward 曲线族 + 最终胜率 |
| E-D-3 | GRPO vs DPO | 同源轨迹的 DPO/GRPO | DPO 样本效率更高，GRPO 探索性更强（胜率-训练步数） | 胜率曲线 + KL-Reward 散点（对齐 tax 图） |

---

## 2. AutoDL 云端 SOP（服务器就绪后逐项执行）

### 2.1 连接与清理（保证充足空间）

```bash
ssh -p <PORT> root@<REGION>.autodl.com      # 连接信息由用户提供后填入
df -h /root/autodl-tmp                      # 数据盘空间检查
nvidia-smi                                  # GPU 型号/显存/驱动确认
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.backends.cuda.is_flash_attn_enabled or 'n/a')"
```

清理顺序（保守，先看后删）：

```bash
du -sh /root/autodl-tmp/* 2>/dev/null | sort -rh | head    # 大文件清单
rm -rf /root/autodl-tmp/tmp_* /root/autodl-tmp/*.bin.tmp   # 明确的临时产物
pip cache purge; conda clean -a -y                          # 包缓存（通常数 GB）
# 旧环境仅在确认不用后删除：conda env list -> conda env remove -n <name>
```

验收：数据盘剩余空间 >= 120 GB（数据 17.1 + checkpoints 峰值 ~60 + 环境 ~15）。

### 2.2 环境重建

优先复用 AutoDL 预装 PyTorch 镜像（torch 2.x + cu12x）；版本不满足
`torch ~=2.5,<2.9` 时用 uv（内网镜像见 MIGRATION.md §2）：

```bash
cd /root/autodl-tmp && git clone <gitee-repo-url> nanoGPT-Modern && cd nanoGPT-Modern
uv sync --extra eval --extra quant --extra server
python data/validate.py data/openwebtext/train.bin    # 数据上传后验证（MIGRATION.md §3）
```

### 2.3 数据上传（本地执行）

```powershell
# scp/rsync 从本机推 17.1GB 五件套（MIGRATION.md §1 清单）；AutoDL 也可走网盘中转
scp -P <PORT> data/openwebtext/train.bin root@<REGION>.autodl.com:/root/autodl-tmp/nanoGPT-Modern/data/openwebtext/
```

### 2.4 执行（tmux 后台 + 断点续跑）

```bash
tmux new -s ablations
python run_component_ablations.py --out_root out/ablations --stage all \
  --max_iters 18000 --batch_size 32 --device cuda
# 抢占/中断后重跑同命令即自动组内 --resume（FR3）
```

### 2.5 结果回传（本地执行）

```powershell
# 关键产物：实验摘要 JSON/MD、metrics.jsonl、checkpoint（best）、图源数据
scp -P <PORT> -r root@<REGION>.autodl.com:/root/autodl-tmp/nanoGPT-Modern/out/ablations/experiments .\cloud_results\
# best checkpoint 按需单独拉取（每个 ~600MB）
```

---

## 3. 图表与报告规范

- **工具**：matplotlib，统一风格（无网格底纹、色盲友好配色、每图一个结论性标题句，如「GQA-2KV 削减 75% KV 显存，val loss 代价 <0.02」）。脚本落 `tools/plotting/plot_experiments.py`（输入摘要 JSON，输出 `docs/assets/experiments/*.png`）。
- **每张图三要素**：数据（可复现命令）、结论句（标题）、上下文注解（规模/环境/种子）。
- **解读深度**：不止报数字——回答「为什么」（机制归因：如 GQA 显存节省来自 KV 头数 8->2；MoE 负载均衡损失下降曲线说明 gating 学习动态）与「so what」（工程决策含义：何时值得用什么配置）。
- **沉淀路径**：图表 + 解读先进 EXPERIMENTS.md 对应小节（E-编号对齐），跨实验综合结论进 TECH_REPORT.md；README 只保留最亮眼的 3-5 张图与一句话结论。
- **诚实报告**：负面结果（如 FSDP 在 50M 规模不占优、小模型 cache 开销反转）必须保留并给出适用边界——这是学习仓库可信度的核心。

---

## 4. 实验状态跟踪表

> 状态取值：`planned` / `running`（云端执行中）/ `done`（数据齐）/ `reported`（图+解读已沉淀文档）。AR 列记录最近一次影响该实验的 AR 版本。

| 编号 | 状态 | AR 版本 | 执行环境 | 产出位置 |
|------|------|---------|----------|----------|
| E-A-1..6 | planned（云实例待租） | AR-v0.4.0（矩阵定义）/ AR-v0.5.0（编排器加固） | 云 GPU 50M 18k iters | `out/ablations/<group>/experiments/` |
| E-A-7 | planned | AR-v0.4.0 | 云 GPU（依赖 E-A-1 checkpoint） | `out/ablations/ntk_eval.json` |
| E-B-1 | part-done（本机 sm75：显存平坦已验证，吞吐数据待云测） | AR-v0.5.0 + GPU 演练修复（2026-10-11） | 本机 Quadro RTX 5000 -> 云 | MIGRATION.md §8.2 / 待 `out/ablations/inf/` |
| E-B-2..5 | planned | — | 云 GPU | 待定 |
| E-C-1..4 | planned | AR-v0.5.0（DDP/FSDP 栈就绪） | 云 GPU | 待定 |
| E-C-5 | part-done（本机：12,335 -> -1,090 B/token；云 flash 后端对照待测） | GPU 演练修复（2026-10-11） | 本机 -> 云 | MIGRATION.md §8.2 |
| E-D-1 | done（3.3M 快速验证：88-100% 遵从率） | AR-v0.3.0 | RTX 4060 Laptop | EXPERIMENTS.md §5.1 |
| E-D-2 | done（G4, 50 steps 曲线） | AR-v0.3.0 | RTX 4060 Laptop | EXPERIMENTS.md §5.2 |
| E-D-3 | planned | — | 云（依赖 A 组 checkpoint） | 待定 |

### 更新历史

| 日期 | AR / 事件 | 更新内容 |
|------|-----------|----------|
| 2026-10-11 | GPU 演练（ST 遗留清单） | 手册创建；E-B-1/E-C-5 记录本机修复成果（SDPA GQA 物化 bug） |
| (下次) | (AR 归档时) | (按维护规则更新状态表与命令) |
