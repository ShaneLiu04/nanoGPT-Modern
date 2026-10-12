#!/bin/bash
# Follow-ups aimed at the four measured limits:
# SFT format vs data budget, GRPO group-size noise, and a same-width draft.
set -u
cd /root/autodl-tmp/nano-gpt-modern
export PYTHONPATH="/root/autodl-tmp/nano-gpt-modern:/root/autodl-tmp/pydeps${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PATH="/root/miniconda3/bin:$PATH"
PY=/root/miniconda3/bin/python
CKPT=out/ablations/ablation-modern-full/best_ckpt.pt
SFT=out/alignment/sft/best_sft-only.pt
mkdir -p out/cloud docs/assets/experiments

nvidia-smi dmon -s um -d 15 -c 400 > /root/autodl-tmp/gpu_dmon2.log 2>&1 &

echo "=== memory probe batch 256 ==="
BATCH=256
if ! $PY - << 'PY'
import torch
from training.trainer_base import load_model_from_checkpoint
model, _ = load_model_from_checkpoint(
    "out/ablations/ablation-modern-full/best_ckpt.pt", device="cuda"
)
idx = torch.randint(0, 1000, (256, 256), device="cuda")
targets = torch.randint(0, 1000, (256, 256), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    _, loss, _ = model(idx, targets=targets)
loss.backward()
torch.cuda.synchronize()
gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
print(f"probe_gb {gb:.1f}", flush=True)
if gb > 80:
    raise SystemExit(2)
PY
then
  BATCH=128
  echo "probe failed, using batch $BATCH"
fi
echo "SFT batch $BATCH"

echo "=== group variance ==="
$PY tools/experiments/measure_group_variance.py \
  --checkpoint "$SFT" \
  --output_json out/cloud/group_variance.json \
  || echo VARIANCE_FAIL

run_sft() {
  local n="$1"
  local epochs="$2"
  local out="out/alignment/sft-n${n}-e${epochs}"
  echo "=== SFT n=$n epochs=$epochs $(date) ==="
  $PY -m training.train_sft \
    --init_from "$CKPT" \
    --out_dir "$out" \
    --variant sft-only \
    --epochs "$epochs" \
    --num_train "$n" \
    --num_val 200 \
    --batch_size "$BATCH" \
    --max_length 256 \
    --eval_interval 100 \
    --seed 1337 \
    --device cuda || { echo "SFT n=$n FAILED"; return; }
  local ckpt="$out/best_sft-only.pt"
  if [ ! -f "$ckpt" ]; then
    ckpt="$out/final_sft-only.pt"
  fi
  $PY evaluation/eval_alignment.py \
    --checkpoint "$ckpt" \
    --num_samples 80 \
    --max_response_len 64 \
    --batch_size 32 \
    --save_failures 0 || echo "EVAL n=$n FAILED"
}

run_sft 2000 8
run_sft 8000 3
run_sft 20000 2

echo "=== collect sft scale ==="
$PY - << 'PY'
import json
from pathlib import Path

rows = []
base = json.loads(Path("out/alignment/sft/eval_results.json").read_text(encoding="utf-8"))
levels = ["easy", "medium", "hard"]
mean = sum(base[k]["format_pass_rate"] for k in levels) / 3
rows.append({
    "label": "2k x 2ep",
    "num_train": 2000,
    "epochs": 2,
    "format_mean": mean,
    "by_level": {k: base[k]["format_pass_rate"] for k in levels},
    "accuracy_mean": sum(base[k]["accuracy"] for k in levels) / 3,
})
for n, epochs in ((2000, 8), (8000, 3), (20000, 2)):
    path = Path(f"out/alignment/sft-n{n}-e{epochs}/eval_results.json")
    if not path.is_file():
        print("missing", path)
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "label": f"{n//1000}k x {epochs}ep",
        "num_train": n,
        "epochs": epochs,
        "format_mean": sum(data[k]["format_pass_rate"] for k in levels) / 3,
        "by_level": {k: data[k]["format_pass_rate"] for k in levels},
        "accuracy_mean": sum(data[k]["accuracy"] for k in levels) / 3,
    })
Path("out/cloud/sft_scale.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
PY

echo "=== matched-width draft ==="
$PY -m training.train_pretrain \
  --config config/pretrain_cloud.yaml \
  --model modern \
  --n_layer 8 --n_head 8 --n_embd 512 --n_kv_head 2 \
  --data_dir data/openwebtext \
  --out_dir out/ablations/draft-8l512 \
  --exp_id draft-8l512 \
  --max_iters 4000 \
  --warmup_iters 400 \
  --lr_decay_iters 4000 \
  --eval_interval 1000 \
  --eval_iters 4 \
  --batch_size 64 \
  || echo DRAFT_FAIL

echo "=== speculative with matched draft ==="
$PY - << 'PY'
import json
from pathlib import Path
import torch
from tools.experiments.run_inference_suite import speculative_rows, time_generate
from training.trainer_base import load_model_from_checkpoint

target, _ = load_model_from_checkpoint(
    "out/ablations/ablation-modern-full/best_ckpt.pt", device="cuda"
)
target.eval()
draft_path = Path("out/ablations/draft-8l512/best_ckpt.pt")
if not draft_path.is_file():
    draft_path = Path("out/ablations/draft-8l512/latest_ckpt.pt")
draft, _ = load_model_from_checkpoint(str(draft_path), device="cuda")
draft.eval()
rows = speculative_rows(target, draft, 64)
compile_rows = []
for new_tokens in (64, 256):
    eager = time_generate(target, 128, new_tokens, True, 2)
    compile_rows.append({"new_tokens": new_tokens, "compile": False, **eager})
    prompt = torch.randint(0, 1000, (1, 128), device="cuda")
    target.generate(
        prompt, max_new_tokens=8, temperature=0.0, top_k=1, use_cache=True, compile=True
    )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    target.generate(
        prompt,
        max_new_tokens=new_tokens,
        temperature=0.0,
        top_k=1,
        use_cache=True,
        compile=True,
    )
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end) / 1000.0
    compile_rows.append({
        "new_tokens": new_tokens,
        "compile": True,
        "tok_per_s": new_tokens / max(elapsed, 1e-6),
        "seconds": elapsed,
        "eager_tok_per_s": eager["tok_per_s"],
    })
    print(compile_rows[-1], flush=True)
payload = {
    "draft": "8L/512/4000 steps, same width as the 12L target",
    "rows": rows,
    "compile_rows": compile_rows,
}
Path("out/cloud/spec_matched.json").write_text(
    json.dumps(payload, indent=2), encoding="utf-8"
)
print("wrote spec_matched", flush=True)
PY

echo "=== plots ==="
$PY tools/plotting/plot_experiments.py \
  --out_root out/ablations --results out/cloud --fig_dir docs/assets/experiments \
  || echo PLOT_FAIL
echo "=== QUALITY FOLLOWUP DONE $(date) ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
