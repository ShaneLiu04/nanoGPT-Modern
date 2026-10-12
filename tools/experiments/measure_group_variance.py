"""Measure within-group reward noise for GRPO group sizes.

The statistic is the standard deviation of rewards inside one prompt's group,
then averaged over prompts. The standard error of the group mean is that
standard deviation divided by sqrt(group size). Advantages that are normalized
inside the group have standard deviation near 1 for every group size, so they
do not answer this question.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tiktoken
import torch

from data.arithmetic import generate_easy, generate_hard, generate_medium
from rewards.rule_reward import compute_reward_batch
from training.trainer_base import load_model_from_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--prompts_per_level", type=int, default=16)
    parser.add_argument("--max_prompt_len", type=int, default=64)
    parser.add_argument("--max_response_len", type=int, default=64)
    parser.add_argument("--gen_batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def encode_prompts(tokenizer, rows, max_len):
    ids = []
    answers = []
    for row in rows:
        toks = tokenizer.encode(row["prompt"])[-max_len:]
        ids.append(toks)
        answers.append(row["answer"])
    return ids, answers


def generate_all(model, prompt_ids, group, max_new, eos, gen_batch):
    """Generate ``group`` responses for every prompt, in wide batches."""
    repeated = prompt_ids * group
    device = next(model.parameters()).device
    decoded = []
    tokenizer = tiktoken.get_encoding("gpt2")
    for start in range(0, len(repeated), gen_batch):
        chunk = repeated[start : start + gen_batch]
        by_len = {}
        for i, toks in enumerate(chunk):
            by_len.setdefault(len(toks), []).append(i)
        out_ids = [None] * len(chunk)
        for length, indexes in by_len.items():
            idx = torch.tensor([chunk[i] for i in indexes], dtype=torch.long, device=device)
            generated = model.generate(
                idx,
                max_new_tokens=max_new,
                temperature=1.0,
                top_k=50,
                use_cache=True,
                eos_token_id=eos,
            )
            for row, src in enumerate(indexes):
                out_ids[src] = generated[row, length:].tolist()
        for resp in out_ids:
            if eos in resp:
                resp = resp[: resp.index(eos)]
            decoded.append(tokenizer.decode(resp))
    return decoded


def summarize(rewards, group, batch):
    stds = []
    means = []
    for b in range(batch):
        sample = [rewards[g * batch + b] for g in range(group)]
        stds.append(float(np.std(sample)))
        means.append(float(np.mean(sample)))
    reward_std = float(np.mean(stds))
    return {
        "group_size": group,
        "prompts": batch,
        "mean_reward": float(np.mean(means)),
        "within_group_reward_std": reward_std,
        "std_error_of_group_mean": reward_std / (group**0.5),
    }


def main():
    args = get_args()
    tokenizer = tiktoken.get_encoding("gpt2")
    model, _ = load_model_from_checkpoint(args.checkpoint, device="cuda")
    model.eval()
    rows_src = []
    rows_src.extend(generate_easy(args.prompts_per_level, seed=args.seed))
    rows_src.extend(generate_medium(args.prompts_per_level, seed=args.seed + 1))
    rows_src.extend(generate_hard(args.prompts_per_level, seed=args.seed + 2))
    prompt_ids, answers = encode_prompts(tokenizer, rows_src, args.max_prompt_len)
    payload = {"checkpoint": args.checkpoint, "rows": []}
    for group in (4, 8, 16):
        decoded = generate_all(
            model,
            prompt_ids,
            group,
            args.max_response_len,
            tokenizer.eot_token,
            args.gen_batch,
        )
        rewards, _, _, _ = compute_reward_batch(decoded, answers * group)
        row = summarize(rewards, group, len(prompt_ids))
        payload["rows"].append(row)
        print(row, flush=True)
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
