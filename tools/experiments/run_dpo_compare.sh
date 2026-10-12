#!/bin/bash
# E-D-3: offline DPO/IPO on one frozen SFT rollout, scored like GRPO.
set -euo pipefail
cd /root/autodl-tmp/nano-gpt-modern
export PYTHONPATH=/root/autodl-tmp/nano-gpt-modern:/root/autodl-tmp/pydeps
export PYTHONUNBUFFERED=1
PY=/root/miniconda3/bin/python
SFT=out/alignment/sft/best_sft-only.pt
if [ ! -f "$SFT" ]; then
  SFT=out/alignment/sft/final_sft-only.pt
fi
mkdir -p out/cloud out/alignment/dpo out/alignment/ipo
echo "SFT $SFT" 
"$PY" tools/experiments/build_preference_pairs.py \
  --checkpoint "$SFT" \
  --train_out out/cloud/pref_train.jsonl \
  --val_out out/cloud/pref_val.jsonl \
  --summary out/cloud/pref_summary.json \
  --per_level 512 \
  --val_per_level 64 \
  --group_size 8 \
  --max_new_tokens 64 \
  --batch_sequences 128 \
  --seed 20260
for loss in dpo ipo; do
  echo "=== train $loss ==="
  "$PY" -m training.train_dpo \
    --init_from "$SFT" \
    --ref_from "$SFT" \
    --preference_source jsonl \
    --preference_jsonl out/cloud/pref_train.jsonl \
    --preference_val_jsonl out/cloud/pref_val.jsonl \
    --preference_loss "$loss" \
    --out_dir "out/alignment/$loss" \
    --batch_size 32 \
    --max_length 192 \
    --epochs 3 \
    --learning_rate 5e-6 \
    --min_lr 5e-7 \
    --warmup_iters 20 \
    --weight_decay 0.0 \
    --beta 0.1 \
    --eval_interval 40 \
    --seed 1337 \
    --device cuda
done
"$PY" tools/experiments/eval_dpo_compare.py \
  --ref "$SFT" \
  --summary out/cloud/pref_summary.json \
  --output out/cloud/dpo_compare.json \
  --num_samples 80 \
  --max_response_len 64
"$PY" tools/plotting/plot_experiments.py
echo "=== DPO COMPARE DONE ==="
