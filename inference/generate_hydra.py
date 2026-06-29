"""Hydra entry point for text generation / inference benchmarking."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import hydra
import torch
from omegaconf import DictConfig


from inference.generate import benchmark, load_model, _make_prompt
from model.attention_utils import set_attention_backend, print_attention_backend
from utils.hydra_utils import to_namespace


@hydra.main(config_path="../config/hydra", config_name="generate", version_base=None)
def main(cfg: DictConfig) -> None:
    args: Any = to_namespace(cfg)
    set_attention_backend(getattr(args, "attn_backend", "auto"))
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
        s = args.prompt[: config.block_size]
        prompt_ids = torch.tensor(
            [[ord(c) for c in s]], dtype=torch.long, device=device
        )

    print(f"Model: {args.model} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Prompt: {args.prompt!r}")
    print(f"Prompt tokens={prompt_ids.shape[1]} | samples={args.num_samples}")
    print()

    all_results: Dict[str, Optional[Dict[int, Any]]] = {"no_cache": None, "cache": None}
    headers = ["config", "total_ms", "prefill_ms", "decode_ms", "decode_tok/s", "total_tok/s", "mem_MB"]

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

        if args.model == "baseline":
            break

    if args.output_json:
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in all_results.items():
            if v:
                out[k] = {str(nt): d for nt, d in v.items()}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
