# MIGRATION.md — 云租用 GPU 实例迁移 Runbook

> 目标：在全新的 Linux + NVIDIA GPU（sm_80+，A100/H100 级）实例上，从零到「可执行 AR-v0.4.0 实验矩阵」≤ 1 小时。
> 本文档是 AR-v0.5.0-hpc-readiness FR2 的交付物，环境依赖契约见 `pyproject.toml` / `uv.lock`（torch `~=2.5,<2.9`，cu12x 变体）。
>
> **演练记录（2026-10-08，Windows 本机）**：干净 venv `uv sync` 1m20s → 数据验证 27s → 50 步冒烟 30s，核心链路合计 **2m17s**（≤1h 达标）。两个平台差异已固化到排障表：Windows PyPI torch 为 CPU-only（云 Linux manylinux wheel 自带 cu128 不受影响）；Windows 需 `--num_workers 0`（spawn 竞态已知问题）。

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
