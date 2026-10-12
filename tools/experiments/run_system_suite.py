"""E-C: short training-system comparisons on the same data and seed.

Each training row is a few hundred steps. Throughput is the quantity these
rows are sized to measure. The MoE kernel row is a microbenchmark of the
production grouped GEMM against a per-expert loop on the same weights.
Single-GPU DDP/FSDP runs, when they launch, measure wrapper overhead only.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--moe_json", required=True)
    parser.add_argument("--max_iters", type=int, default=300)
    parser.add_argument("--data_dir", default="data/openwebtext")
    return parser.parse_args()


def summarize_run(name, out_dir):
    metrics = Path(out_dir) / "experiments" / name / "metrics.jsonl"
    speeds, losses = [], []
    if metrics.is_file():
        with open(metrics, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "train" and "tokens_per_sec" in event:
                    speeds.append(float(event["tokens_per_sec"]))
                if event.get("type") == "eval" and "val_loss" in event:
                    losses.append(float(event["val_loss"]))
    row = {"name": name, "out_dir": str(out_dir)}
    if speeds:
        row["tok_per_s"] = statistics.median(speeds)
    if losses:
        row["final_val_loss"] = losses[-1]
    return row


def run_train(name, extra):
    out_dir = f"out/system/{name}"
    cmd = [
        sys.executable,
        "-m",
        "training.train_pretrain",
        "--config",
        "config/pretrain_cloud.yaml",
        "--model",
        "modern",
        "--n_kv_head",
        "2",
        "--data_dir",
        "data/openwebtext",
        "--out_dir",
        out_dir,
        "--exp_id",
        name,
        *extra,
    ]
    print("run", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, text=True, capture_output=True)
    row = summarize_run(name, out_dir)
    row["returncode"] = result.returncode
    if result.returncode != 0:
        row["error"] = (result.stderr or result.stdout)[-2000:]
    else:
        print(row, flush=True)
    return row


def moe_benchmark(path):
    from model.modern_gpt import ModernGPTConfig, SwiGLU

    device = "cuda"
    experts, grouped_ms, loop_ms = [], [], []
    tokens = 4096
    for num_experts in (2, 4, 8, 16):
        config = ModernGPTConfig(
            vocab_size=50257,
            block_size=128,
            n_layer=2,
            n_head=8,
            n_embd=512,
            n_kv_head=2,
            num_experts=num_experts,
        )
        module = SwiGLU(config).to(device=device, dtype=torch.bfloat16).eval()
        x = torch.randn(1, tokens, config.n_embd, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            for _ in range(5):
                module(x)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                module(x)
            torch.cuda.synchronize()
            grouped = (time.perf_counter() - start) / 20 * 1000

            # Reference loop: one expert matmul at a time on that expert's rows.
            # Routing matches the production top-1 choice.
            x_flat = x.view(tokens, config.n_embd).float()
            router = torch.softmax(x_flat @ module.router.weight.float().T, dim=-1)
            selected = router.argmax(dim=-1)
            gate_w = module.gate_w.float()
            up_w = module.up_w.float()
            down_w = module.down_w.float()

            def loop_once(
                x_flat=x_flat,
                selected=selected,
                gate_w=gate_w,
                up_w=up_w,
                down_w=down_w,
                num_experts=num_experts,
            ):
                out = torch.zeros_like(x_flat)
                for expert in range(num_experts):
                    mask = selected == expert
                    if int(mask.sum()) == 0:
                        continue
                    hidden_in = x_flat[mask]
                    gate = torch.nn.functional.silu(hidden_in @ gate_w[expert].T)
                    up = hidden_in @ up_w[expert].T
                    out[mask] = (gate * up) @ down_w[expert].T
                return out

            for _ in range(3):
                loop_once()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                loop_once()
            torch.cuda.synchronize()
            loop = (time.perf_counter() - start) / 10 * 1000
        experts.append(num_experts)
        grouped_ms.append(grouped)
        loop_ms.append(loop)
        print(f"experts={num_experts} grouped={grouped:.2f}ms loop={loop:.2f}ms", flush=True)
    payload = {"experts": experts, "grouped_ms": grouped_ms, "loop_ms": loop_ms, "tokens": tokens}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = get_args()
    short = [
        "--max_iters",
        str(args.max_iters),
        "--warmup_iters",
        str(max(20, args.max_iters // 10)),
        "--lr_decay_iters",
        str(args.max_iters),
        "--eval_interval",
        str(args.max_iters),
        "--eval_iters",
        "8",
        "--log_interval",
        "20",
        "--batch_size",
        "32",
    ]
    rows = [
        run_train("bf16", [*short, "--precision", "bf16"]),
        run_train("fp32", [*short, "--precision", "fp32"]),
        run_train(
            "accum",
            [
                *short,
                "--batch_size",
                "8",
                "--gradient_accumulation_steps",
                "4",
            ],
        ),
        run_train("fsdp", [*short, "--fsdp"]),
    ]
    payload = {
        "note": "Single GPU. FSDP here is wrapper overhead, not multi-GPU scaling.",
        "rows": rows,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    moe_benchmark(args.moe_json)
    print("wrote", out)


if __name__ == "__main__":
    main()
