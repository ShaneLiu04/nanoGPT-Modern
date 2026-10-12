"""Freeze one SFT rollout into chosen/rejected pairs for DPO and IPO.

The policy is the same SFT checkpoint GRPO started from. Each prompt is
sampled ``group_size`` times. The highest rule reward is chosen and the
lowest is rejected. Ties are dropped, so the pair file records a real
preference rather than noise.
"""

import argparse
import json
from collections import defaultdict

import torch

from data.arithmetic import generate_easy, generate_hard, generate_medium
from rewards.rule_reward import compute_reward_batch
from training.trainer_base import load_model_from_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--val_out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--per_level", type=int, default=512)
    parser.add_argument("--val_per_level", type=int, default=64)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_prompt_len", type=int, default=120)
    parser.add_argument("--batch_sequences", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def prompts_for(n, seed):
    """Balanced easy/medium/hard prompts from seeds that eval does not use."""
    rows = []
    rows.extend(generate_easy(n, seed))
    rows.extend(generate_medium(n, seed + 1))
    rows.extend(generate_hard(n, seed + 2))
    return rows


def roll_out(model, rows, args, encoder):
    """Sample a group per prompt and keep the reward extremes."""
    buckets = defaultdict(list)
    for row in rows:
        ids = encoder.encode(row["prompt"])
        if not ids or len(ids) > args.max_prompt_len:
            continue
        item = dict(row)
        item["prompt_ids"] = ids
        buckets[len(ids)].append(item)

    pairs = []
    ties = 0
    seen = 0
    width = max(1, args.batch_sequences // args.group_size)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    for length in sorted(buckets):
        group = buckets[length]
        for start in range(0, len(group), width):
            chunk = group[start : start + width]
            repeated = []
            for row in chunk:
                repeated.extend([row] * args.group_size)
            idx = torch.tensor(
                [row["prompt_ids"] for row in repeated],
                dtype=torch.long,
                device=args.device,
            )
            with torch.no_grad(), ctx:
                out = model.generate(
                    idx,
                    max_new_tokens=args.max_new_tokens,
                    temperature=1.0,
                    use_cache=True,
                )
            for i, row in enumerate(chunk):
                seen += 1
                texts = []
                sequences = []
                prompt_len = len(row["prompt_ids"])
                for g in range(args.group_size):
                    seq = out[i * args.group_size + g].tolist()
                    new = seq[prompt_len : prompt_len + args.max_new_tokens]
                    texts.append(encoder.decode(new))
                    sequences.append(row["prompt_ids"] + new)
                rewards, _, _, acc = compute_reward_batch(texts, [row["answer"]] * args.group_size)
                winner = max(range(args.group_size), key=lambda g: rewards[g])
                loser = min(range(args.group_size), key=lambda g: rewards[g])
                if rewards[winner] <= rewards[loser] + 1e-8:
                    ties += 1
                    continue
                chosen = sequences[winner]
                rejected = sequences[loser]
                pairs.append(
                    {
                        "chosen": chosen,
                        "rejected": rejected,
                        "chosen_mask": [0] * prompt_len + [1] * (len(chosen) - prompt_len),
                        "rejected_mask": ([0] * prompt_len + [1] * (len(rejected) - prompt_len)),
                        "chosen_reward": rewards[winner],
                        "rejected_reward": rewards[loser],
                        "chosen_accuracy": acc[winner],
                        "rejected_accuracy": acc[loser],
                        "level": row["level"],
                    }
                )
            print(
                f"len {length}: pairs {len(pairs)} ties {ties} seen {seen}",
                flush=True,
            )
    return pairs, ties, seen


def write_jsonl(path, pairs):
    with open(path, "w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row) + "\n")


def summarize(pairs, ties, seen):
    if not pairs:
        return {"pairs": 0, "ties": ties, "seen": seen, "tie_rate": 1.0}
    margin = [row["chosen_reward"] - row["rejected_reward"] for row in pairs]
    acc_gap = [row["chosen_accuracy"] - row["rejected_accuracy"] for row in pairs]
    return {
        "pairs": len(pairs),
        "ties": ties,
        "seen": seen,
        "tie_rate": ties / max(seen, 1),
        "mean_chosen_reward": sum(row["chosen_reward"] for row in pairs) / len(pairs),
        "mean_rejected_reward": sum(row["rejected_reward"] for row in pairs) / len(pairs),
        "mean_margin": sum(margin) / len(margin),
        "accuracy_gap": sum(acc_gap) / len(acc_gap),
    }


def main():
    args = get_args()
    import tiktoken

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    encoder = tiktoken.get_encoding("gpt2")
    model, _ = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()

    train_rows = prompts_for(args.per_level, args.seed)
    val_rows = prompts_for(args.val_per_level, args.seed + 1000)
    train_pairs, train_ties, train_seen = roll_out(model, train_rows, args, encoder)
    val_pairs, val_ties, val_seen = roll_out(model, val_rows, args, encoder)
    write_jsonl(args.train_out, train_pairs)
    write_jsonl(args.val_out, val_pairs)
    summary = {
        "checkpoint": args.checkpoint,
        "group_size": args.group_size,
        "max_new_tokens": args.max_new_tokens,
        "train": summarize(train_pairs, train_ties, train_seen),
        "val": summarize(val_pairs, val_ties, val_seen),
    }
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
