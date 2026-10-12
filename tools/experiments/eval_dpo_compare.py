"""Score SFT, GRPO, DPO, and IPO with the same arithmetic evaluator.

KL is always against the SFT reference, so the scatter is an alignment-tax
plot on one reward definition.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_samples", type=int, default=80)
    parser.add_argument("--max_response_len", type=int, default=64)
    return parser.parse_args()


def first_existing(paths):
    for path in paths:
        if Path(path).is_file():
            return path
    return None


def evaluate(checkpoint, ref, samples, response_len):
    cmd = [
        sys.executable,
        "evaluation/eval_alignment.py",
        "--checkpoint",
        checkpoint,
        "--ref_checkpoint",
        ref,
        "--num_samples",
        str(samples),
        "--max_response_len",
        str(response_len),
        "--batch_size",
        "16",
        "--save_failures",
        "0",
        "--seed",
        "1337",
    ]
    print("run", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"eval failed: {checkpoint}")
    path = Path(checkpoint).parent / "eval_results.json"
    with open(path, encoding="utf-8") as handle:
        by_level = json.load(handle)
    levels = [level for level in ("easy", "medium", "hard") if level in by_level]

    def mean(key):
        return sum(by_level[level][key] for level in levels) / len(levels)

    return {
        "checkpoint": checkpoint,
        "reward": mean("reward"),
        "accuracy": mean("accuracy"),
        "format_pass_rate": mean("format_pass_rate"),
        "invalid_rate": mean("invalid_rate"),
        "kl": mean("kl_divergence"),
        "by_level": by_level,
    }


def main():
    args = get_args()
    methods = [
        (
            "SFT",
            [
                "out/alignment/sft/best_sft-only.pt",
                "out/alignment/sft/final_sft-only.pt",
            ],
        ),
        (
            "GRPO-G4",
            [
                "out/alignment/grpo-g4/best_grpo_g4.pt",
                "out/alignment/grpo-g4/final_grpo_g4.pt",
            ],
        ),
        (
            "GRPO-G8",
            [
                "out/alignment/grpo-g8/best_grpo_g8.pt",
                "out/alignment/grpo-g8/final_grpo_g8.pt",
            ],
        ),
        (
            "GRPO-G16",
            [
                "out/alignment/grpo-g16/best_grpo_g16.pt",
                "out/alignment/grpo-g16/final_grpo_g16.pt",
            ],
        ),
        (
            "DPO",
            [
                "out/alignment/dpo/best_ckpt.pt",
                "out/alignment/dpo/final_ckpt.pt",
            ],
        ),
        (
            "IPO",
            [
                "out/alignment/ipo/best_ckpt.pt",
                "out/alignment/ipo/final_ckpt.pt",
            ],
        ),
    ]
    rows = []
    for name, paths in methods:
        checkpoint = first_existing(paths)
        if checkpoint is None:
            print("missing", name, flush=True)
            continue
        row = evaluate(checkpoint, args.ref, args.num_samples, args.max_response_len)
        row["name"] = name
        rows.append(row)
        print(
            f"{name} reward={row['reward']:.3f} acc={row['accuracy']:.3f} " f"kl={row['kl']:.4f}",
            flush=True,
        )
    with open(args.summary, encoding="utf-8") as handle:
        pairs = json.load(handle)
    payload = {"pairs": pairs, "methods": rows}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
