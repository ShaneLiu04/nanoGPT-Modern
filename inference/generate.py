"""Inference script with KV Cache toggle and throughput benchmarking.

Uses CUDA events for precise wall-time measurement and separates
prefill (prompt encoding) from decode (token-by-token generation).
"""

import argparse
import json
import torch
from model.attention_utils import set_attention_backend, print_attention_backend
from model.modern_gpt import ModernGPT, ModernGPTConfig
from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from utils.config import parse_args_with_config


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--model", type=str, default="modern", choices=["baseline", "modern"]
    )
    p.add_argument(
        "--prompt", type=str, default="The future of artificial intelligence is"
    )
    p.add_argument("--max_new_tokens", type=int, nargs="+", default=[400, 500])
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument(
        "--top_p", type=float, default=None, help="Nucleus sampling threshold"
    )
    p.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Penalty for repeated tokens (>1.0 discourages repetition)",
    )
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--output_json", type=str, default=None)
    p.add_argument(
        "--prompt_len", type=int, default=50, help="Pad/trim prompt to this many tokens"
    )
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    p.add_argument(
        "--attn_backend",
        type=str,
        default="auto",
        choices=["auto", "flash", "mem_efficient", "math", "default"],
        help="Force SDPA attention backend (auto lets PyTorch choose)",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Compile the forward pass with torch.compile (reduce-overhead) "
        "to lower kernel launch overhead in the token-by-token loop. "
        "Only effective on CUDA; CPU falls back to eager.",
    )
    return parse_args_with_config(p)


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
    return model, config


def _make_prompt(tokenizer, text, prompt_len, device):
    toks = tokenizer.encode(text)
    if len(toks) > prompt_len:
        toks = toks[:prompt_len]
    else:
        # Pad short prompts with the EOT token instead of repeating the text,
        # which preserves the original semantic while reaching the target length.
        pad_id = getattr(tokenizer, "eot_token", 50256)
        toks = toks + [pad_id] * (prompt_len - len(toks))
        toks = toks[:prompt_len]
    return torch.tensor([toks], dtype=torch.long, device=device)


def benchmark(
    model,
    prompt_ids,
    max_new_tokens,
    num_samples,
    device,
    temperature=1.0,
    top_k=50,
    top_p=None,
    repetition_penalty=None,
    use_cache=True,
    compile=False,
):
    """Returns a dict with prefill/decode/total timing and throughput.

    Parameters
    ----------
    use_cache : bool
        If True and the model supports it, run a manual prefill+decode loop
        with KV cache.  Otherwise use ``model.generate`` without cache.
    """
    from inference.generate_utils import _sample_logits

    is_cuda = "cuda" in str(device)
    sync = torch.cuda.synchronize if is_cuda else (lambda: None)

    has_cache_api = "use_cache" in model.generate.__code__.co_varnames
    use_cache = use_cache and has_cache_api

    # ---- optional torch.compile for the manual decode loop ----
    forward_mod = model
    if compile and is_cuda:
        if getattr(model, "_compiled_forward", None) is None:
            try:
                model._compiled_forward = torch.compile(
                    model, mode="reduce-overhead", fullgraph=False, dynamic=True
                )
            except Exception as e:
                print(f"[compile warning] torch.compile failed: {e}; using eager.")
                model._compiled_forward = model
        forward_mod = model._compiled_forward

    def _forward(*args, **kwargs):
        nonlocal forward_mod
        try:
            return forward_mod(*args, **kwargs)
        except Exception as e:
            if compile and forward_mod is not model:
                print(f"[compile warning] compiled forward failed: {e}; using eager.")
                forward_mod = model
                return model(*args, **kwargs)
            raise

    prefill_ms = []
    decode_ms = []
    total_ms = []
    decode_tok_s = []

    for run in range(num_samples + 1):  # first is warmup
        inp = prompt_ids.clone()
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        sync()
        if is_cuda:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev2 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        # --- Prefill: process the prompt ---
        if use_cache:
            # First step: feed the whole prompt (prefill)
            logits, _, past_kvs = _forward(inp, use_cache=True, start_pos=0)
            if is_cuda:
                ev1.record()
            # Decode: generate the rest
            out_ids = inp.clone()
            start_pos = inp.shape[1]
            for _ in range(max_new_tokens):
                next_inp = out_ids[:, -1:]
                logits, _, past_kvs = _forward(
                    next_inp, past_kvs=past_kvs, use_cache=True, start_pos=start_pos
                )
                logits_last = logits[:, -1, :]
                if repetition_penalty is not None and repetition_penalty != 1.0:
                    for b in range(out_ids.shape[0]):
                        for tid in out_ids[b].tolist():
                            logits_last[b, tid] /= repetition_penalty
                idx_next = _sample_logits(
                    logits_last, temperature=temperature, top_k=top_k, top_p=top_p
                ).unsqueeze(-1)
                out_ids = torch.cat([out_ids, idx_next], dim=1)
                # sliding window: keep at most block_size - 1 past tokens so the
                # next single-token step stays within block_size total context.
                max_cl = model.config.block_size - 1
                for li in range(len(past_kvs)):
                    k, v = past_kvs[li]
                    if k.shape[2] > max_cl:
                        trim = k.shape[2] - max_cl
                        past_kvs[li] = (k[:, :, trim:, :], v[:, :, trim:, :])
                        start_pos += trim
                start_pos += 1
        else:
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                use_cache=False,
            )
            if "compile" in model.generate.__code__.co_varnames:
                gen_kwargs["compile"] = compile
            out_ids = model.generate(inp, **gen_kwargs)
            if is_cuda:
                ev1.record()  # no separate prefill/decode

        if is_cuda:
            ev2.record()
        sync()

        if is_cuda:
            t_prefill = ev0.elapsed_time(ev1)  # ms
            t_decode = ev1.elapsed_time(ev2)  # ms
            t_total = ev0.elapsed_time(ev2)
        else:
            t_total = t_prefill = 0.0

        if run == 0:  # warmup
            continue
        prefill_ms.append(t_prefill)
        decode_ms.append(t_decode)
        total_ms.append(t_total)
        if t_decode > 0:
            decode_tok_s.append(max_new_tokens / (t_decode / 1000.0))

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if is_cuda else 0.0

    def med(vals):
        s = sorted(vals)
        return s[len(s) // 2]

    return {
        "total_time_ms": med(total_ms),
        "prefill_ms": med(prefill_ms),
        "decode_ms": med(decode_ms),
        "decode_tok_s": med(decode_tok_s) if decode_tok_s else 0.0,
        "total_tok_s": (
            max_new_tokens / (med(total_ms) / 1000.0) if med(total_ms) > 0 else 0.0
        ),
        "peak_mem_mb": round(peak_mem, 1),
        "prompt_len": prompt_ids.shape[1],
        "generated_len": max_new_tokens,
    }


def main():
    args = get_args()
    set_attention_backend(args.attn_backend)
    print_attention_backend()
    device = args.device

    model, config = load_model(args.checkpoint, args.model, device)

    try:
        import tiktoken

        tokenizer = tiktoken.get_encoding("gpt2")
    except Exception:
        tokenizer = None

    if tokenizer:
        prompt_ids = _make_prompt(tokenizer, args.prompt, args.prompt_len, device)
    else:
        # fallback: ord(char)
        s = args.prompt[: config.block_size]
        prompt_ids = torch.tensor(
            [[ord(c) for c in s]], dtype=torch.long, device=device
        )

    print(
        f"Model: {args.model} | params={sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"Prompt: {args.prompt!r}")
    print(f"Prompt tokens={prompt_ids.shape[1]} | samples={args.num_samples}")
    print()

    all_results = {"no_cache": None, "cache": None}
    headers = [
        "config",
        "total_ms",
        "prefill_ms",
        "decode_ms",
        "decode_tok/s",
        "total_tok/s",
        "mem_MB",
    ]

    for use_cache in [False, True]:
        label = "cache" if use_cache else "no-cache"
        results = {}
        for nt in args.max_new_tokens:
            r = benchmark(
                model,
                prompt_ids,
                nt,
                args.num_samples,
                device,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                use_cache=use_cache,
                compile=args.compile,
            )
            results[nt] = r
            row = [
                f"{label} (gen={nt})",
                f"{r['total_time_ms']:.1f}",
                f"{r['prefill_ms']:.1f}",
                f"{r['decode_ms']:.1f}",
                f"{r['decode_tok_s']:.0f}",
                f"{r['total_tok_s']:.0f}",
                f"{r['peak_mem_mb']:.1f}",
            ]
            print("  " + " | ".join(f"{h:>12s}" for h in headers))
            print("  " + " | ".join(f"{v:>12s}" for v in row))
            print()
        all_results[label] = results

        # BaselineGPT has no cache
        if args.model == "baseline":
            break

    if args.output_json:
        # Convert keys to str for JSON
        out = {}
        for k, v in all_results.items():
            if v:
                out[k] = {str(nt): d for nt, d in v.items()}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
