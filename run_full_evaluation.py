"""
Full evaluation matrix across all trained checkpoints.
"""

import os
import json
import torch


from model.modern_gpt import ModernGPT, ModernGPTConfig
from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from data.arithmetic import generate_easy, generate_medium, generate_hard
from rewards.rule_reward import compute_reward_batch
import tiktoken


def load_model(path, model_type, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get(
        "config", ModernGPTConfig() if model_type == "modern" else BaselineGPTConfig()
    )
    model = ModernGPT(config) if model_type == "modern" else BaselineGPT(config)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, data, tokenizer, device, max_response_len=20):
    prompts = [d["prompt"] for d in data]
    references = [d["answer"] for d in data]
    responses = []
    with torch.no_grad():
        for prompt in prompts:
            toks = tokenizer.encode(prompt)
            input_ids = torch.tensor([toks], dtype=torch.long, device=device)
            if (
                hasattr(model, "generate")
                and "use_cache" in model.generate.__code__.co_varnames
            ):
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_response_len,
                    temperature=1.0,
                    top_k=50,
                    use_cache=True,
                )
            else:
                generated = model.generate(
                    input_ids,
                    max_new_tokens=max_response_len,
                    temperature=1.0,
                    top_k=50,
                )
            response_ids = generated[0, len(toks) :].tolist()
            response_text = tokenizer.decode(response_ids)
            responses.append(response_text)

    rewards, fmt_scores, proc_scores, acc_scores = compute_reward_batch(
        responses, references
    )
    total = len(data)
    accuracy = sum(acc_scores) / total
    avg_reward = sum(rewards) / total
    format_pass = sum(fmt_scores) / total
    avg_process = sum(proc_scores) / total

    invalid = 0
    import re

    for r in responses:
        content = re.search(r"<answer>(.*?)</answer>", r, re.DOTALL)
        if not content or not content.group(1).strip():
            invalid += 1
    invalid_rate = invalid / total

    return {
        "accuracy": accuracy,
        "reward": avg_reward,
        "format_pass_rate": format_pass,
        "process_score": avg_process,
        "invalid_rate": invalid_rate,
        "sample_responses": responses[:5],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")

    # Generate evaluation data
    eval_data = {
        "easy": generate_easy(50, seed=999),
        "medium": generate_medium(50, seed=998),
        "hard": generate_hard(50, seed=997),
    }

    checkpoints = [
        ("Baseline (pretrain)", "out/pretrain_baseline_fast/best_ckpt.pt", "baseline"),
        ("Modern (pretrain)", "out/pretrain_modern_fast/best_ckpt.pt", "modern"),
        ("Modern (SFT)", "out/sft_fast/final_sft-only.pt", "modern"),
        ("Modern (GRPO-G4)", "out/grpo_fast/final_grpo_g4.pt", "modern"),
    ]

    all_results = {}
    for name, path, mtype in checkpoints:
        print(f"\n=== Evaluating: {name} ===")
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue
        model = load_model(path, mtype, device)
        model_results = {}
        for level, data in eval_data.items():
            res = evaluate_model(model, data, tokenizer, device, max_response_len=20)
            model_results[level] = res
            print(
                f"  {level:8s}: acc={res['accuracy']:.3f}, reward={res['reward']:.3f}, fmt={res['format_pass_rate']:.3f}, proc={res['process_score']:.3f}, invalid={res['invalid_rate']:.3f}"
            )
        all_results[name] = model_results
        del model
        torch.cuda.empty_cache()

    out_path = "out/full_evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
