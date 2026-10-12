#!/bin/bash
# End-to-end cloud campaign: wait for the OpenWebText bins, train the
# seven-way architecture matrix, then run the inference, system and
# alignment suites and write the figures.
set -u
cd /root/autodl-tmp/nano-gpt-modern
export PYTHONPATH="/root/autodl-tmp/nano-gpt-modern:/root/autodl-tmp/pydeps${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/root/miniconda3/bin:$PATH"
PY=/root/miniconda3/bin/python
LOG=/root/autodl-tmp/prepare_train.log
mkdir -p out/cloud docs/assets/experiments docs/results

echo "=== wait for tokenized OpenWebText ==="
while ! grep -q "done train_tokens" "$LOG"; do
  if ! pgrep -f prepare_openwebtext_shards.py >/dev/null; then
    echo "prepare process exited before writing the bins"
    tail -n 40 "$LOG"
    exit 1
  fi
  sleep 20
done
tail -n 5 "$LOG"
df -h /root/autodl-tmp

echo "=== shuffle index ==="
$PY - << 'PY'
from data.openwebtext import generate_shuffle_index
generate_shuffle_index("data/openwebtext/train.bin", seed=1337, block_size=1024)
print("shuffle index ready")
PY

echo "=== real-data smoke ==="
$PY -m training.train_pretrain \
  --config config/pretrain_cloud.yaml \
  --model modern --n_kv_head 2 \
  --max_iters 15 --eval_interval 1000 --eval_iters 1 --log_interval 5 \
  --out_dir out/smoke_real --exp_id smoke_real
rm -rf out/smoke_real

echo "=== choose step budget from the token count ==="
$PY - << 'PY'
import json
import numpy as np
from pathlib import Path
data = np.memmap("data/openwebtext/train.bin", dtype=np.uint16, mode="r")
n_tokens = int(data.shape[0])
per_iter = 32 * 1024
one_epoch = int(0.95 * n_tokens / per_iter)
max_iters = min(18000, max(2000, one_epoch))
max_iters = max(500, max_iters // 500 * 500)
warmup = max(100, max_iters // 9)
budget = {
    "train_tokens": n_tokens,
    "tokens_per_iter": per_iter,
    "one_epoch_iters": one_epoch,
    "max_iters": max_iters,
    "warmup_iters": warmup,
    "batch_size": 32,
    "block_size": 1024,
    "seed": 1337,
    "corpus": "Skylion007/openwebtext parquet prefix, GPT-2 tokens",
}
Path("out/cloud").mkdir(parents=True, exist_ok=True)
Path("out/cloud/budget.json").write_text(json.dumps(budget, indent=2), encoding="utf-8")
print(json.dumps(budget))
PY

MAX_ITERS=$($PY -c "import json; print(json.load(open('out/cloud/budget.json'))['max_iters'])")
WARMUP=$($PY -c "import json; print(json.load(open('out/cloud/budget.json'))['warmup_iters'])")
echo "training max_iters=$MAX_ITERS warmup=$WARMUP"

# The orchestrator reads warmup from the yaml. Rewrite the two fields so the
# cosine schedule matches the step budget actually used.
$PY - << PY
from pathlib import Path
text = Path("config/pretrain_cloud.yaml").read_text(encoding="utf-8")
lines = []
for line in text.splitlines():
    if line.startswith("max_iters:"):
        lines.append("max_iters: $MAX_ITERS")
    elif line.startswith("warmup_iters:"):
        lines.append("warmup_iters: $WARMUP")
    elif line.startswith("lr_decay_iters:"):
        lines.append("lr_decay_iters: $MAX_ITERS")
    else:
        lines.append(line)
Path("config/pretrain_cloud.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "=== E-A component ablations ==="
$PY run_component_ablations.py \
  --config config/pretrain_cloud.yaml \
  --out_root out/ablations \
  --stage all \
  --max_iters "$MAX_ITERS" || echo "ablation orchestrator returned non-zero"

CKPT=out/ablations/ablation-modern-full/best_ckpt.pt
if [ ! -f "$CKPT" ]; then
  CKPT=out/ablations/ablation-modern-full/latest_ckpt.pt
fi
echo "modern checkpoint: $CKPT"

if [ -f "$CKPT" ]; then
  echo "=== E-A-7 NTK ==="
  $PY tools/experiments/run_ntk_eval.py \
    --checkpoint "$CKPT" \
    --output_json out/cloud/ntk_eval.json || echo "ntk failed"
  echo "=== E-B inference ==="
  $PY tools/experiments/run_inference_suite.py \
    --checkpoint "$CKPT" \
    --output_json out/cloud/inference.json || echo "inference failed"
  echo "=== E-D alignment ==="
  $PY tools/experiments/run_alignment_suite.py \
    --init_from "$CKPT" \
    --output_json out/cloud/alignment.json || echo "alignment failed"
fi

echo "=== E-C system ==="
$PY tools/experiments/run_system_suite.py \
  --output_json out/cloud/system.json \
  --moe_json out/cloud/moe_kernel.json || echo "system failed"

echo "=== figures ==="
$PY tools/plotting/plot_experiments.py \
  --out_root out/ablations \
  --results out/cloud \
  --fig_dir docs/assets/experiments || echo "plots failed"

echo "=== campaign done ==="
df -h /root/autodl-tmp
