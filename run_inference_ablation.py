"""Batch inference ablation with CUDA-event timing and prefill/decode breakdown."""
import os, json, torch
from model.modern_gpt import ModernGPT, ModernGPTConfig
from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
import tiktoken

def load_model(ckpt_path, model_type, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw = ckpt.get("config", None)
    if raw is None:
        config = ModernGPTConfig() if model_type == "modern" else BaselineGPTConfig()
    elif isinstance(raw, dict) and model_type == "modern":
        config = ModernGPTConfig.from_dict(raw)
    else:
        config = raw
    model = (ModernGPT if model_type == "modern" else BaselineGPT)(config)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model

def benchmark(model, prompt_ids, max_new_tokens, num_samples, device):
    from inference.generate import benchmark as gen_bench
    return gen_bench(model, prompt_ids, max_new_tokens, num_samples, device)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "The future of artificial intelligence is"
    toks = tokenizer.encode(prompt)
    toks = toks * ((50 // len(toks)) + 1)
    toks = toks[:50]
    prompt_ids = torch.tensor([toks], dtype=torch.long, device=device)

    experiments = [
        ("out/pretrain_baseline_fast/best_ckpt.pt", "baseline"),
        ("out/pretrain_modern_fast/best_ckpt.pt", "modern"),
    ]
    lengths = [50, 400, 500]
    num_samples = 10

    results = {}
    for ckpt_path, model_type in experiments:
        print(f"\n=== Loading {model_type} ===")
        model = load_model(ckpt_path, model_type, device)
        for use_cache in ([False, True] if model_type == "modern" else [False]):
            label = f"{model_type}_{'cache' if use_cache else 'nocache'}"
            results[label] = {}
            for nt in lengths:
                r = benchmark(model, prompt_ids, nt, num_samples, device)
                results[label][nt] = r
                print(f"  {label:20s} gen={nt:3d}: "
                      f"total={r['total_time_ms']:.0f}ms "
                      f"prefill={r['prefill_ms']:.0f}ms "
                      f"decode={r['decode_ms']:.0f}ms "
                      f"decode_tps={r['decode_tok_s']:.0f} "
                      f"total_tps={r['total_tok_s']:.0f} "
                      f"mem={r['peak_mem_mb']:.1f}MB")
        del model
        torch.cuda.empty_cache()

    os.makedirs("out", exist_ok=True)
    out_path = "out/inference_ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
