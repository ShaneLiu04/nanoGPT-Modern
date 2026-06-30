"""
Unified evaluation for alignment experiments.

Metrics: accuracy, reward, format_pass_rate, invalid_rate, KL divergence.
KL is computed only over response tokens (prompt tokens are masked out).
"""

import os
import argparse
import re
import json

import numpy as np
import torch


from data.arithmetic import generate_easy, generate_medium, generate_hard
from rewards.rule_reward import compute_reward_batch
from training.trainer_base import load_model_from_checkpoint
from inference.generate_utils import generate_by_length
from utils.rl_utils import compute_kl_divergence, compute_token_logprobs
import tiktoken


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--ref_checkpoint", type=str, default=None, help="Reference model for KL"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--max_response_len", type=int, default=128)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for logprob/KL computation",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument(
        "--save_failures",
        type=int,
        default=10,
        help="Number of failure examples to save",
    )
    return parser.parse_args()


def _batch_logprobs(model, sequences, response_masks, device, ctx):
    """Compute token log-probs for variable-length sequences in a batch.

    Parameters
    ----------
    model : nn.Module
    sequences : list[list[int]]
    response_masks : list[list[int]]
        1 for response tokens, 0 otherwise (per target position).
    device : str
    ctx : autocast context

    Returns
    -------
    token_logprobs : list[torch.Tensor]
        Per-sequence logprobs (only valid where mask == 1).
    """
    if not sequences:
        return []

    pad_id = 0  # value does not matter for masked positions
    max_len = max(len(s) for s in sequences)
    B = len(sequences)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    targets = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)

    for i, seq in enumerate(sequences):
        L = len(seq)
        input_ids[i, : L - 1] = torch.tensor(seq[:-1], dtype=torch.long, device=device)
        targets[i, : L - 1] = torch.tensor(seq[1:], dtype=torch.long, device=device)
        attention_mask[i, : L - 1] = True

    with torch.no_grad():
        with ctx:
            logits, _, _ = model(input_ids, attention_mask=attention_mask)
        token_logprobs = compute_token_logprobs(logits, targets, mask=attention_mask)

    result = []
    for i, mask in enumerate(response_masks):
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device)
        # token_logprobs is padded to max_len; slice to the actual target length.
        result.append(token_logprobs[i, : len(mask_t)][mask_t])
    return result


def evaluate(
    model, ref_model, data, tokenizer, device, max_response_len, batch_size, ctx
):
    prompts = [d["prompt"] for d in data]
    references = [d["answer"] for d in data]

    prompt_lens = [len(tokenizer.encode(p)) for p in prompts]

    responses, response_ids_list = generate_by_length(
        model,
        prompts,
        tokenizer,
        max_new_tokens=max_response_len,
        device=device,
        batch_size=batch_size,
        temperature=1.0,
        top_k=50,
        eos_token_id=tokenizer.eot_token,
        return_ids=True,
    )

    rewards, fmt_scores, proc_scores, acc_scores = compute_reward_batch(
        responses, references
    )

    # Compute KL divergence only over response tokens, batched.
    kl_sum = 0.0
    kl_count = 0
    if ref_model is not None:
        ref_model.eval()
        sequences = []
        masks = []
        for toks, resp_ids, p_len in zip(
            [tokenizer.encode(p) for p in prompts], response_ids_list, prompt_lens
        ):
            full_ids = toks + resp_ids
            if len(full_ids) > model.config.block_size:
                # Keep the most recent block_size tokens; truncate prompt prefix.
                trim = len(full_ids) - model.config.block_size
                full_ids = full_ids[trim:]
                p_len = max(0, p_len - trim)
            if len(full_ids) < 2:
                continue

            # Mask: 1 only for response target positions.
            seq_mask = [0] * (len(full_ids) - 1)
            # Response targets occupy positions [p_len-1, p_len+len(resp_ids)-2].
            resp_start = max(0, p_len - 1)
            resp_end = max(0, p_len + len(resp_ids) - 1)
            seq_mask[resp_start:resp_end] = [1] * (resp_end - resp_start)
            sequences.append(full_ids)
            masks.append(seq_mask)

        # Process in mini-batches to avoid OOM.
        all_policy_logp = []
        all_ref_logp = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            batch_masks = masks[i : i + batch_size]
            all_policy_logp.extend(
                _batch_logprobs(model, batch_seqs, batch_masks, device, ctx)
            )
            all_ref_logp.extend(
                _batch_logprobs(ref_model, batch_seqs, batch_masks, device, ctx)
            )

        for p_logp, r_logp in zip(all_policy_logp, all_ref_logp):
            # Use the same reverse-KL form as GRPO training:
            # KL(ref || policy) = ref_logp - policy_logp
            kl_sum += compute_kl_divergence(r_logp, p_logp, reduction="sum").item()
            kl_count += p_logp.shape[0]

    total = len(data)
    accuracy = sum(acc_scores) / total
    avg_reward = sum(rewards) / total
    format_pass = sum(fmt_scores) / total
    avg_process = sum(proc_scores) / total

    invalid = 0
    failure_examples = []
    for i, r in enumerate(responses):
        parsed = re.search(r"<answer>(.*?)</answer>", r, re.DOTALL)
        if not parsed or not parsed.group(1).strip():
            invalid += 1
        if acc_scores[i] < 1.0 and len(failure_examples) < args.save_failures:
            failure_examples.append(
                {
                    "prompt": prompts[i],
                    "reference": references[i],
                    "response": r,
                    "reward": rewards[i],
                    "format": fmt_scores[i],
                    "accuracy": acc_scores[i],
                }
            )

    invalid_rate = invalid / total
    avg_kl = kl_sum / max(kl_count, 1)

    return {
        "accuracy": accuracy,
        "reward": avg_reward,
        "format_pass_rate": format_pass,
        "process_score": avg_process,
        "invalid_rate": invalid_rate,
        "kl_divergence": avg_kl,
        "responses": responses,
        "failure_examples": failure_examples,
    }


def main():
    global args
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    tokenizer = tiktoken.get_encoding("gpt2")

    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    ref_model = None
    if args.ref_checkpoint:
        ref_model, _ = load_model_from_checkpoint(args.ref_checkpoint, device=device)
        ref_model.eval()

    ctx = (
        torch.amp.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
        )
        if device.startswith("cuda") and torch.cuda.is_bf16_supported()
        else None
    )
    if ctx is None:
        from contextlib import nullcontext

        ctx = nullcontext()

    generators = {
        "easy": lambda n, s: generate_easy(n, s),
        "medium": lambda n, s: generate_medium(n, s),
        "hard": lambda n, s: generate_hard(n, s),
    }

    results = {}
    for level in args.levels:
        data = generators[level](
            args.num_samples,
            args.seed + {"easy": 0, "medium": 1, "hard": 2}[level],
        )
        res = evaluate(
            model,
            ref_model,
            data,
            tokenizer,
            device,
            args.max_response_len,
            args.batch_size,
            ctx,
        )
        results[level] = res
        print(f"\n=== {level.upper()} ===")
        print(f"  accuracy        : {res['accuracy']:.3f}")
        print(f"  reward          : {res['reward']:.3f}")
        print(f"  format_pass_rate: {res['format_pass_rate']:.3f}")
        print(f"  process_score   : {res['process_score']:.3f}")
        print(f"  invalid_rate    : {res['invalid_rate']:.3f}")
        print(f"  kl_divergence   : {res['kl_divergence']:.6f}")

    # save results
    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: vv for kk, vv in v.items() if kk not in ("responses",)}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
