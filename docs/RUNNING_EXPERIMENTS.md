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
  --config config/pretrain_cloud.yaml \
  --out_root out/ablations --stage all \
  --max_iters 18000
# E-A-7 用训练好的 modern-full checkpoint，而不是 eval_benchmark 的 context_lengths 参数：
python tools/experiments/run_ntk_eval.py \
  --checkpoint out/ablations/ablation-modern-full/best_ckpt.pt \
  --output_json out/cloud/ntk_eval.json
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
# E-B-3：投机解码消融（suite 内部自动训 4 层 draft 并扫 draft_len 2/4/8；
# inference/generate.py CLI 不暴露 speculative 参数，勿用它跑 E-B-3）
python tools/experiments/run_inference_suite.py \
  --checkpoint out/ablations/modern-full/best_ckpt.pt \
  --output_json out/ablations/inf_suite.json \
  --draft_iters 1200 --draft_out out/ablations/draft
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

### F 组：AI infra 微基准（AR-v0.6.0 新增，待多卡/高算力 GPU 执行）

> 来源：AR-v0.6.0-ai-infra-foundations（T005）。本组不训练模型，全部是分钟级微基准，
> 用于补齐手册（`docs/ai-infra/memory-flops-arithmetic.md`）心算口径的实测锚点：
> 手算总量只到量级（§7 残差 7~9 GB vs 26.26 GB），结构归因靠本组 + profiler。
> 四要素齐备才可登记 `running`：环境 / 命令 / 产物 / 判据。

| 编号 | 实验 | 对比项 | 假设 | 图表产出 |
|------|------|--------|------|----------|
| E-F-1 | NCCL 通信带宽 | all-reduce 实测带宽 vs PCIe/NVLink 规格值 | 集合通信达到链路规格的 70%+；无 NVLink 时 PCIe 是多卡扩展的真实瓶颈 | all-reduce 带宽-消息大小曲线（8B..256MB） |
| E-F-2 | 多卡 roofline + MFU | `tools/infra/run_roofline.py` 全量 + 训练 MFU（`utils/infra_math.py` flops 口径） | 云卡 GEMM 稳态算力达规格 80%+；50M 训练 MFU 落在 30-60% 区间（amp 口径） | roofline 三段表 + MFU 单点（对照手册 §5/§7 残差） |
| E-F-3 | KV 物化 GPU 复测 | 手册 §4 案例（hd=96：18,432→4,608 B/token）与本仓主配置（hd=64：12,288 B/token） | GPU 上逐层分配实测斜率与公式闭合（±5%），修复后曲线平坦（对照 MIGRATION §8.2） | 斜率表（公式 vs 实测）+ 显存-长度曲线 |
| E-F-4 | inductor 全链路验证 | 云端配套 torch/triton 下复跑 dynamo 层实测（compile-stack.md §4）与 decode 编译路径 | inductor 后端下 recompile 计数与 dynamo 层一致（图行为不因 backend 改变）；256 token 编译路径吞吐 ≥ eager（64 token 允许仍为负，与本机摊平边界一致） | recompile 计数表 + compile/eager 吞吐对照（回填 compile-stack.md §7） |

**E-F-1 通信带宽**（环境：≥2 卡同型号实例，nccl；命令：临时脚本，不新增仓库文件）

```bash
# torchrun 2 进程，各绑一卡；消息 8B..256MB 对数扫描，每档 warmup 3 + 计时 10
cat > /tmp/bench_allreduce.py <<'EOF'
import os, torch, torch.distributed as dist
dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
for nbytes in [8, 1<<10, 1<<20, 16<<20, 64<<20, 256<<20]:
    x = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
    for _ in range(3): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier()
    t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(10): dist.all_reduce(x)
    t1.record(); torch.cuda.synchronize()
    ms = t0.elapsed_time(t1) / 10
    if dist.get_rank() == 0:
        # 总线带宽口径：环形 all-remove 每卡收发 2*(N-1)/N 倍消息
        n = dist.get_world_size(); print(f"{nbytes:>12} B  {ms:10.3f} ms  {2*(n-1)/n*nbytes/ms*1e3/1e9:8.1f} GB/s")
dist.destroy_process_group()
EOF
torchrun --standalone --nproc_per_node=2 /tmp/bench_allreduce.py
```

产物：`docs/results/nccl_bw.json`（可后续把脚本固化进 tools/infra/）。判据：峰值带宽 ≥ 链路规格 70% 记 PASS；低于 40% 先查拓扑（`nvidia-smi topo -m`）再下结论。

**E-F-2 多卡 roofline + MFU**（环境：任一云卡；命令：本仓工具，FR4 交付物）

```bash
python tools/infra/run_roofline.py --out docs/results/roofline_<gpu>.json   # 全量 13 点
# MFU：用 infra_math 口径算 50M 训练一步的理论 FLOPs，除以 (实测 step 时间 x 卡规格算力)
python -m utils.infra_math   # param/flops/显存分解（口径见手册 §3/§5）
```

产物：`docs/results/roofline_<gpu>.json` + MFU 记录进 EXPERIMENTS.md。判据：GEMM 稳态（8192³ 档）≥ 规格 fp32 算力 80%；MFU 单点给出即可，不设阈值（50M 小模型 MFU 低是常态，重点是口径闭合）。

**E-F-3 KV 物化复测**（环境：任一 GPU；命令：本仓推理路径）

```bash
python tools/profiling/profile_inference.py --checkpoint <ckpt> \
  --context_lengths 512 1024 2048 4096 --use_cache append   # 峰值显存-长度斜率
```

产物：斜率表（公式 12,288 B/token @ hd=64 fp32 vs 实测；修复前 12,335 对照 MIGRATION §8.2）。判据：实测斜率与公式差 ±5% 内 PASS；显著偏离先查 dtype/头数参数集再下结论（手册 §4「看数字先问参数集」）。

**E-F-4 inductor 全链路验证**（环境：云 Linux + 官方配套 torch/triton（如 torch 2.5.1 + triton 3.1.0）+ CUDA GPU；命令：本仓测试与推理路径；来源：AR-v0.7.0 T004，本机 triton 3.8 vs torch 2.5.1 inductor 断裂为教学案例（compile-stack.md §5），云端空闲后执行，非本 AR 阻塞项）

```bash
pip install "triton==3.1.0"                    # 配套 torch 2.5.1（版本矩阵见 compile-stack.md §5.1）
python -m pytest tests/test_moe_vectorized.py -k recompile -q   # inductor 后端下 recompile 计数
python -m tools.experiments.run_inference_suite --checkpoint <ckpt> \
  --output_json docs/results/inductor_cloud.json \
  --prompt_lengths 128 256 --new_tokens 64    # 摊平边界对照；再跑一次 --new_tokens 256
```

产物：`docs/results/inductor_cloud.json`（compile/eager 吞吐 + recompile 计数 + torch/triton 版本），结论回填 compile-stack.md §7。判据：inductor 下 recompile 计数与 dynamo 层（`backend="eager"`，MIGRATION §8.1）一致；256 token 编译路径 ≥ eager 吞吐；64 token 允许仍为负——与本机 EXPERIMENTS §6.4 摊平边界（64~256 之间）一致。

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

2026-10-12 这台实例的数据盘配额是 50 GB，装不下手册初稿假设的 120 GB（全量 17 GB 语料 + 七份完整 checkpoint）。实际做法是流式下载 OpenWebText parquet、tokenize 后删除分片，只保留每组的 best/latest checkpoint，Python 依赖装到数据盘。50 GB 够用：跑完七组后约 24 GB 已用。全量 80 个分片仍然需要更大的盘。

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
| E-A-1..6 | reported | AR-v0.4.0 / AR-v0.5.0 | RTX 6000D，18000 step，OpenWebText 前缀 | `docs/results/ablation-summary.json`，EXPERIMENTS.md §6.2 |
| E-A-7 | reported | AR-v0.4.0 | 同上，modern-full checkpoint | `docs/results/ntk_eval.json`，EXPERIMENTS.md §6.3 |
| E-B-1 | reported | AR-v0.5.0 | RTX 6000D，prompt 128/512/896 | `docs/results/inference.json`，EXPERIMENTS.md §6.4 |
| E-B-2 | reported | — | int8 KV；短上下文上显存差很小 | 同上 |
| E-B-3 | reported | — | 弱草稿，接受率 0 | 同上 |
| E-B-4 | reported | — | 并发 1/4/16 | 同上 |
| E-B-5 | reported | — | 64 token 解码上 compile 更慢 | 同上 |
| E-C-1 | reported | AR-v0.5.0 | 300 iter，bf16 为 fp32 的 2.10 倍 | `docs/results/system.json`，EXPERIMENTS.md §6.6 |
| E-C-2 | reported | — | 累积步数与 iter 计数未对齐，见 §6.6 | 同上 |
| E-C-3 | reported | — | 16 专家时分组 GEMM 5.8 倍 | `docs/results/moe_kernel.json` |
| E-C-4 | reported（单卡） | AR-v0.5.0 | 单卡 FSDP 与 bf16 几乎相同，没有强扩展数据 | `docs/results/system.json` |
| E-C-5 | part-done | GPU 演练修复（2026-10-11） | 修复前后斜率仍以本机 MIGRATION.md §8.2 为准；云端只测了修复后的 append 显存 | MIGRATION.md §8.2 |
| E-D-1 | reported | AR-v0.3.0 + 云端复测 | 格式分上限 0.5；2000 条平均分 0.45（上限的 90%），invalid 0–2.5%。加到 20000 条后平均分 0.492 | EXPERIMENTS.md §6.5 |
| E-D-2 | reported | AR-v0.3.0 + 云端 G4/G8/G16 | RTX 6000D，各 120 step | `docs/results/alignment.json`，EXPERIMENTS.md §6.5 |
| E-D-3 | reported | — | 同一 SFT 上的 DPO/IPO，1333 对；奖励升、准确率不升 | `docs/results/dpo_compare.json`，EXPERIMENTS.md §6.5 |
| E-F-1 | planned | AR-v0.6.0 | 需 ≥2 卡实例（nccl）；脚本见 F 组 SOP 内联 | `docs/results/nccl_bw.json`（待产出） |
| E-F-2 | part-done | AR-v0.6.0 | 本机 Quadro RTX 5000 已跑 --quick（EXPERIMENTS.md §6.8）；全量 13 点 + MFU 待云卡 | `docs/results/roofline.json`；`roofline_<gpu>.json`（待产出） |
| E-F-3 | planned | AR-v0.6.0 | 任一 GPU；公式口径 hd=64 fp32 = 12,288 B/token | 斜率表（待产出；本机修复记录 MIGRATION.md §8.2） |
| E-F-4 | planned | AR-v0.7.0 | 云 Linux + 配套 triton（本机断裂为教学案例）；Triton 三件套本机已测（EXPERIMENTS.md §6.9） | `docs/results/inductor_cloud.json`（待产出） |

### 更新历史

| 日期 | AR / 事件 | 更新内容 |
|------|-----------|----------|
| 2026-10-11 | GPU 演练（ST 遗留清单） | 手册创建；E-B-1/E-C-5 记录本机修复成果（SDPA GQA 物化 bug） |
| 2026-10-12 | 云端实测 | RTX 6000D 上跑完 E-A、E-B、E-C 与 E-D-1/E-D-2。数据是 12.44 亿 token 前缀，不是全量 17 GB。多卡扩展仍未测。解读在 EXPERIMENTS.md §6 |
| 2026-10-12 | E-D-3 | 在 RTX PRO 6000 Blackwell 上用冻结的 SFT 采样做 DPO/IPO。奖励上升来自过程分，准确率没有上升 |
| 2026-10-13 | AR-v0.6.0（T005） | 新增 F 组（E-F-1/2/3）微基准 SOP：通信带宽/多卡 roofline+MFU/KV 物化复测，四要素齐备；E-F-2 本机 --quick 部分完成（§6.8） |
| 2026-10-14 | AR-v0.7.0（T004） | 新增 E-F-4 inductor 全链路验证 SOP（四要素齐备）；Triton 微实验三件套本机实测完成（EXPERIMENTS.md §6.9，`docs/results/triton_kernels.json`） |
