"""E-D: SFT format compliance and GRPO group-size comparison.

The pretrain checkpoint is the format baseline. SFT is a short arithmetic
run. GRPO then continues from that SFT checkpoint at group sizes 4, 8 and 16.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_from", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--sft_epochs", type=int, default=2)
    parser.add_argument("--num_train", type=int, default=2000)
    parser.add_argument("--grpo_steps", type=int, default=120)
    parser.add_argument("--eval_samples", type=int, default=40)
    parser.add_argument("--skip_sft", action="store_true")
    return parser.parse_args()


def run(cmd):
    print("run", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(cmd)}")


def eval_ckpt(checkpoint, samples, response_len):
    run(
        [
            sys.executable,
            "evaluation/eval_alignment.py",
            "--checkpoint",
            checkpoint,
            "--num_samples",
            str(samples),
            "--max_response_len",
            str(response_len),
            "--batch_size",
            "8",
            "--save_failures",
            "0",
        ]
    )
    path = Path(checkpoint).parent / "eval_results.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def grpo_curve(out_dir):
    metrics = list(Path(out_dir).glob("experiments/**/metrics.jsonl"))
    steps, reward = [], []
    if not metrics:
        return steps, reward
    with open(metrics[0], encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = event.get("reward", event.get("mean_reward"))
            if value is None:
                continue
            step = event.get("step", event.get("iter"))
            if step is None:
                continue
            steps.append(int(step))
            reward.append(float(value))
    return steps, reward


def main():
    args = get_args()
    sft_dir = Path("out/alignment/sft")
    sft_ckpt = sft_dir / "best_sft-only.pt"
    if not sft_ckpt.is_file():
        sft_ckpt = sft_dir / "final_sft-only.pt"
    if not (args.skip_sft and sft_ckpt.is_file()):
        run(
            [
                sys.executable,
                "-m",
                "training.train_sft",
                "--init_from",
                args.init_from,
                "--out_dir",
                str(sft_dir),
                "--variant",
                "sft-only",
                "--epochs",
                str(args.sft_epochs),
                "--num_train",
                str(args.num_train),
                "--num_val",
                "200",
                "--batch_size",
                "16",
                "--max_length",
                "256",
                "--eval_interval",
                "100",
                "--seed",
                "1337",
                "--device",
                "cuda",
            ]
        )
    sft_ckpt = sft_dir / "best_sft-only.pt"
    if not sft_ckpt.is_file():
        sft_ckpt = sft_dir / "final_sft-only.pt"

    evals = [
        {
            "name": "pretrain",
            "by_level": eval_ckpt(args.init_from, args.eval_samples, 64),
        },
        {
            "name": "sft",
            "by_level": eval_ckpt(str(sft_ckpt), args.eval_samples, 64),
        },
    ]
    curves = []
    for group in (4, 8, 16):
        out_dir = Path(f"out/alignment/grpo-g{group}")
        run(
            [
                sys.executable,
                "-m",
                "training.train_grpo",
                "--init_from",
                str(sft_ckpt),
                "--ref_from",
                str(sft_ckpt),
                "--out_dir",
                str(out_dir),
                "--group_size",
                str(group),
                "--num_steps",
                str(args.grpo_steps),
                "--batch_size",
                "4",
                "--max_prompt_len",
                "64",
                "--max_response_len",
                "64",
                "--eval_interval",
                "20",
                "--seed",
                "1337",
                "--device",
                "cuda",
            ]
        )
        steps, reward = grpo_curve(out_dir)
        curves.append({"name": f"G{group}", "steps": steps, "reward": reward})
        print(f"grpo G{group} points={len(steps)}", flush=True)

    payload = {"evals": evals, "grpo_curves": curves, "sft_checkpoint": str(sft_ckpt)}
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
