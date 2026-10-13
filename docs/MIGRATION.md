# MIGRATION.md — 云租用 GPU 实例迁移 Runbook

> 目标：在全新的 Linux + NVIDIA GPU（sm_80+，A100/H100 级）实例上，从零到「可执行 AR-v0.4.0 实验矩阵」≤ 1 小时。
> 本文档是 AR-v0.5.0-hpc-readiness FR2 的交付物，环境依赖契约见 `pyproject.toml` / `uv.lock`（torch `~=2.5,<2.9`，cu12x 变体）。
>
> **演练记录（2026-10-08，Windows 本机）**：干净 venv `uv sync` 1m20s → 数据验证 27s → 50 步冒烟 30s，核心链路合计 **2m17s**（≤1h 达标）。两个平台差异已固化到排障表：Windows PyPI torch 为 CPU-only（云 Linux manylinux wheel 自带 cu128 不受影响）；Windows 需 `--num_workers 0`（spawn 竞态已知问题）。
>
> **GPU 演练记录（2026-10-11，Windows 本机 Quadro RTX 5000 16 GiB / sm75）**：见 §8。四项演练全部 PASS：① torch.compile recompile 计数验证（dynamo 层）；② KV 显存斜率抓到并修复 SDPA GQA head-expansion 物化 bug（decode 峰值显存平坦化）；③ 单卡 DDP（FR9a）全链路 + 修复 ReduceOp.AVG gloo 兼容与 DDP device_ids CPU 兼容 2 个真 bug；④ 单卡 FSDP（FR9b）GPU 全链路含 FULL_STATE_DICT gather + resume round-trip。多卡真分片留待云实例。

## 0. 前置检查

```bash
nvidia-smi                    # 驱动 ≥ CUDA 12.1（cu12x wheel 最低要求）
df -h /workspace              # 数据 ~17.1 GB + checkpoints 预留 ≥ 100 GB
python3 --version             # 3.9-3.11（lock 按 3.9/3.11 双版本解析）
```

## 1. 必带文件清单（~17.1 GB）

从源机器（或对象存储/rsync）带以下 5 个文件，目录结构保持 `data/openwebtext/`：

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `data/openwebtext/train.bin` | 17,027 MB | 8,926,903,359 tokens / 7,913,597 docs（uint16） |
| `data/openwebtext/val.bin` | 21 MB | 11,147,787 tokens / 10,000 docs |
| `data/openwebtext/train.idx` | 0.2 MB | dtype/vocab/total_tokens/num_docs 元数据 |
| `data/openwebtext/val.idx` | 0.2 MB | 同上（val） |
| `data/openwebtext/train.bin.shuffle.idx` | 66 MB | 全局 shuffle 索引（`.npy`，AR-v0.4.0 T006 产物，**训练严格依赖**） |

**明确不带**（共 ~39.1 GB 冗余，见 §6 清理）：
- `data/openwebtext/train_shards/`（16.6 GB）—— `train.bin` 的分片中间产物，已在拼接后冗余
- `data/owt_parquet/`（22.5 GB）—— HF 原始 parquet 缓存，仅 `prepare.py --data_files` 重建数据时需要

rsync 示例（源机器执行；跳过冗余目录）：

```bash
rsync -avP --exclude='train_shards' --exclude='owt_parquet' \
  data/openwebtext/ user@gpu-host:/workspace/nanoGPT-Modern/data/openwebtext/
```

## 2. 环境重建（uv，~5-10 分钟）

```bash
git clone <repo> && cd nanoGPT-Modern
# 带上本仓库（含 uv.lock），rsync/git 传输均可

# 安装 uv（Linux）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 一键同步（lock 精确复现依赖；torch 解析为 2.8.0 cu12x）
uv sync --extra eval --extra quant --extra server

# cu12x 变体说明：PyPI 默认 torch wheel 即 cu12x 系列；若需特定变体：
#   uv pip install "torch~=2.5,<2.9" --index-url https://download.pytorch.org/whl/cu126
```

无外网时使用内部镜像（华为内网示例）：

```bash
UV_INDEX_URL=http://mirrors.tools.huawei.com/pypi/simple/ uv sync --extra eval --extra quant --extra server
```

## 3. 30 秒数据验证

```bash
python data/validate.py data/openwebtext/train.bin
python data/validate.py data/openwebtext/val.bin
```

**通过判据**（对照源机器 `train.idx`）：

| 指标 | 期望值 |
| --- | --- |
| Tokens | 8,926,903,359 |
| Token range | 0 - 50256（vocab_size=50257 内） |
| EOT count | 7,913,597（≈ 每文档 1 个，avg doc length ≈ 1128 tokens） |
| Top-20 tokens | GPT-2 常见 token（198=the 类），分布无异常尖峰 |

数据校验失败 → 检查传输完整性（`sha256sum train.bin` 与源机比对）；vocab/EOT 异常 → 重新 rsync。

## 4. 50 步 GPU 冒烟（~2 分钟）

```bash
python -m training.train_pretrain \
  --out_dir out/migration_smoke \
  --batch_size 8 --max_iters 50 --eval_interval 25 --eval_iters 5 --log_interval 10 \
  --device cuda --model modern \
  --n_layer 4 --n_head 4 --n_embd 256 \
  --data_dir data/openwebtext \
  --num_workers 0 \
  --seed 42
```

> `--num_workers 0` 让冒烟命令在 Windows/Linux 双平台通用（Windows 下多 worker DataLoader 有 spawn 竞态已知问题；正式训练在 Linux 云端可用默认 4）。CPU-only 环境（如 PyPI Windows wheel）改用 CI 同款小参数：`--device cpu --model baseline --n_layer 2 --n_head 2 --n_embd 128 --batch_size 2`。

**通过判据**：`out/migration_smoke/` 生成 `latest_ckpt.pt`（+ `best_ckpt.pt`、`config.yaml`）；`out/migration_smoke/experiments/migration_smoke/metrics.jsonl` 含 ≥5 条 `type=train` 事件 + 2 条 `type=eval` 事件；val loss 从 ~10.8 单调下降；无 OOM/NaN。

## 5. 启动 AR-v0.4.0 实验矩阵（编排器）

```bash
python run_component_ablations.py \
  --out_root out/ablations \
  --groups baseline modern-full modern-gqa modern-moe modern-ntk modern-kvint8 \
  --max_iters 18000 --batch_size 32 --device cuda
```

- 抢占场景：重跑同命令即自动组内 `--resume`（AR-v0.5.0 FR3），summary 含 `resumed_from_iter`
- 详见 `docs/ARCHITECTURE.md`（训练基座）与 `docs/EXPERIMENTS.md`（实验语义）

## 6. 磁盘冗余清理（可选，省 ~39.1 GB）

若误传了冗余目录，确认训练正常后可删除：

```bash
rm -rf data/openwebtext/train_shards   # 16.6 GB：train.bin 的分片中间产物
rm -rf data/owt_parquet                # 22.5 GB：HF 原始 parquet，仅在重建数据时需要
```

> 注意：仅在 §3/§4 验证全部通过后清理；若将来需要用 `prepare.py --data_files` 重建数据，需重新下载 parquet。

## 7. 数据重建（仅当未携带数据时，~2-4 小时）

```bash
# 全量重建 train（7,913,597 docs）—— 必须带 --full，否则默认 1M 子集！
python data/prepare.py --split train --full
python data/prepare.py --split val            # val 默认 10K 即全量

# 重建后需重生成 shuffle 索引（见 AR-v0.4.0 T006 语义）
python -c "from data.openwebtext import build_shuffle_index; build_shuffle_index('data/openwebtext', 'train')"

python data/validate.py data/openwebtext/train.bin   # 对照 §3 判据
```

## 排障速查

| 症状 | 处置 |
| --- | --- |
| `uv sync` 拉取超时 | 配置镜像 `UV_INDEX_URL`（§2） |
| `Torch not compiled with CUDA enabled` | Windows 平台 PyPI torch wheel 为 CPU-only；Linux manylinux wheel 自带 cu128 不受影响。Windows 验证用 `--device cpu` + 小参数冒烟 |
| 训练启动即 spawn/多进程崩溃 | Windows 下加 `--num_workers 0`（已知 spawn 竞态；Linux 默认 4 无此问题） |
| 训练报 shuffle idx 缺失 | 检查 `train.bin.shuffle.idx` 是否随迁（§1 第 5 行） |
| 冒烟 OOM | GPU：正常不应出现（A100 80GB）；CPU 冒烟会因 50257-vocab logits OOM，务必用 CI 同款小参数 |
| val loss 不降 | 核对数据验证输出（§3）与 `--data_dir` 路径 |
| `torch.compile` 报 `cannot import name 'triton_key'` | triton 与 torch 版本断裂（本机 triton 3.8.0 vs torch 2.5.1；torch 2.5.1 需 triton 3.1.0，Windows 无官方 wheel）。inductor 不可用，dynamo 层（recompile 计数、graph break 检查）仍可用 `backend="eager"` 验证；云 Linux 装匹配 triton 后不受影响 |
| decode 峰值显存随上下文线性增长（长上下文 OOM） | 已修复（§8.2）：SDPA `enable_gqa` 在无原生 GQA 内核的后端会物化 KV head 扩展（~3×[B,n_head,S,D] 每层每步）。T==1 decode 现走 `_decode_gqa_attention` 手写内核，峰值显存平坦 |
| `generate(compile='fullgraph')` 在 Windows 直接回退 eager | 同 triton 断裂根因；云 Linux 正常。回退后 eager 步走 cache 对象零拷贝 fast path，行为一致 |
| `Cannot use ReduceOp.AVG with Gloo` | 已修复（§8.3）：损失 all-reduce 改 SUM + world_size 除法（全后端通用，数值等价） |
| DDP 报 `device_ids and output_device arguments only work with ... GPU or CPU modules` | 已修复（§8.3）：CPU 模块下 device_ids 传 None；GPU 不受影响 |
| Windows 下 `torchrun` 报 `use_libuv was requested but PyTorch was build without libuv support` | elastic agent 的 TCPStore 硬编码 use_libuv=True（Python 默认参数，env 不生效）。改用进程内 env-init（`RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT` + `USE_LIBUV=0`，同 tests/test_fsdp_checkpoint.py 模式）；云 Linux build 自带 libuv 无此问题 |

## 8. GPU 演练记录（本机 Quadro RTX 5000 16 GiB，2026-10-11）

ST-024 遗留 GPU 迁移演练清单的本机执行结果。云实例（sm_80+）预期更优：flash 后端原生 GQA + 匹配 triton。

### 8.1 torch.compile recompile 稳定性（FR 关联：MoE 向量化）

- **现象**：`torch.compile` 首次调用即 `cannot import name 'triton_key'`（triton 3.8.0 与 torch 2.5.1 API 断裂；torch 2.5.1 配套 triton 3.1.0 无 Windows wheel）。
- **验证**：recompile 计数是 **dynamo 层指标，与 backend 无关**。`backend="eager"` 下 100 个相同形状步进 `unique_graphs` 稳定不增长 → 向量化 MoE 层无 shape/dict 诱导的 graph break。云 Linux inductor 可用后同一测试自动走完整 inductor（`tests/test_moe_vectorized.py::test_compile_no_recompile_growth`）。

### 8.2 KV 显存斜率 → 抓到真 bug 并修复

- **测量**（24L/16H/8KV/1024d fp32，prompt=8，decode 至 512/1024/2048/4096）：峰值显存斜率 12,335 B/token（理论 KV 98,304 的 1/8），R²=0.99999 完美线性——预分配 append buffer 之上还有 O(T) 增长。
- **定位**（`torch.cuda.memory._record_memory_history` 分配栈追踪）：`F.scaled_dot_product_attention(..., enable_gqa=True)` 每层每 decode 步分配 ~[B,16,S,64] fp32 ×3（sm75 无 flash 内核，mem-efficient 后端内部物化 8→16 KV head 扩展）。12,288 B/token ≈ 3×4096×24 精确吻合。零拷贝 cache view 被 SDPA 内部复制击败；grouped 5D 广播同样物化（~2×kv-native）。
- **修复**：新增 `_decode_gqa_attention`（modern_gpt.py）——T==1 且无 additive mask 的 GQA decode 统一走手写 batched-matmul 内核，scores 在原生 KV-head 分辨率 `[B,n_kv,rep,S]`（比扩展形式小 2×n_rep 倍，与 backend 无关）。fast path（cache view）与 value path（cat 产物）同走此内核，**保持 T013 bit-identity 契约**；数值与 SDPA allclose（fp32 噪声级 4.6e-08）。prefill（T>1）与带 mask 的步进保持 SDPA 不变。
- **顺带**：`_generate_fullgraph` 的 eager fallback 步改传 cache 对象（零拷贝 fast path），compiled 步保持 list（dynamo 对自定义对象 fullgraph 兼容性未验证，云上再评估）。
- **效果**：峰值显存斜率 12,335 → **-1,090 B/token（平坦）**；单步瞬时分配 57.7MB → 0.5MB（L=4096，114×）；GPU decode 吞吐 68→64.5 ms/step（+5%，物化在 GPU 上只是带宽拷贝；收益主要在显存容量——长上下文/大 batch 不再 OOM）。测试：`tests/test_decode_gqa_attention.py` 8 用例（Red→Green），全量 627 passed + perf 2 passed。

### 8.3 单卡 DDP 构造冒烟（FR9a）→ 修复 2 个真 bug

- **验证**：单进程 gloo + CUDA（Quadro RTX 5000）真实构造 `DDP(static_graph=True, gradient_as_bucket_view=True)` + 4 步训练 + checkpoint + metrics 全链路 PASS（loss 10.83→10.67 正常下降）。
- **bug 1**：`train_pretrain.py` 损失 all-reduce 用 `ReduceOp.AVG`——**NCCL-only**，gloo 直接 `RuntimeError`（云 nccl 无碍，但 gloo 调试/降级路径全断）。修复：`SUM + world_size 除法`（数值等价、全后端通用，与 estimate_loss 既有写法一致）。既有分布式测试要么 stub 掉 all_reduce 要么只到构造，从未在真实 gloo 组上跑过 train()——新增 `tests/test_ddp_shard.py::test_distributed_train_real_gloo_loss_all_reduce` 锁定。
- **bug 2**：`wrap_distributed` 的 `DDP(device_ids=[local_rank])` 对 CPU 模块非法（gloo+CPU 分布式冒烟/调试场景构造即炸）。修复：CUDA 设备才传 `device_ids`，CPU 传 None。
- **平台注意**：本机 Windows 无 nccl（Linux-only）；gloo 支持 CUDA 张量。**torchrun 在 Windows 被 elastic agent 的 `TCPStore(use_libuv=True)` 硬编码阻断**（Python 端默认参数，env 不生效；进程内 env-init 路径则尊重 `USE_LIBUV=0`）——云 Linux build 自带 libuv，torchrun 正常，不受影响。

### 8.4 单卡 FSDP 全链路（FR9b）——补上 CPU 测试覆盖不到的 GPU 侧

- **验证**：单进程 gloo + CUDA 构造 FSDP（`transformer_auto_wrap_policy` per-block + `use_orig_params=True`）→ 4 步训练 → `FULL_STATE_DICT` gather checkpoint（23 keys，orig-param 命名，无 `_fsdp_wrapped_module` 前缀）→ **resume round-trip**（新 trainer 载入后继续训练 2 步，21/23 float 参数如预期移动）全链路 PASS。
- **说明**：`FULL_STATE_DICT` 真实 gather 是 CUDA-only 能力（`tests/test_fsdp_checkpoint.py` 在 CPU 上只能 stub），本演练首次在 GPU 上验证了完整 gather + 回读语义。world_size=1 时 FSDP 自动降级 `FULL_SHARD → NO_SHARD`（torch 内建 warning，预期行为）——**真 sharding 语义需云实例多卡**，本机验证的是构造/wrap/train/gather/checkpoint 链路。
- **云上待验证**（多卡实例租用后）：≥2 卡 nccl `FULL_SHARD` 真分片 + 跨卡 all-gather 数值一致性 + 多卡 checkpoint 合并。
