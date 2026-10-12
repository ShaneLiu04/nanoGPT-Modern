"""E-A-7: naive RoPE vs NTK-aware RoPE perplexity at longer contexts.

The checkpoint was trained at block_size 1024. Naive extrapolation keeps
the original RoPE base and only extends the allowed context. NTK-aware
extrapolation rescales the base so wavelengths cover ``factor * 1024``.
"""

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from model.modern_gpt import RotaryEmbedding
from training.trainer_base import load_model_from_checkpoint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_bin", default="data/openwebtext/val.bin")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train_block", type=int, default=1024)
    return parser.parse_args()


def ntk_theta(dim, factor, base=10000.0):
    """NTK-aware RoPE base. factor==1 leaves the original base unchanged."""
    if factor == 1.0:
        return base
    return base * (factor ** (dim / (dim - 2)))


def retune_rope(model, factor, max_len, train_block):
    """Point every RoPE table at ``factor`` and allow sequences up to ``max_len``."""
    if factor <= 1.0:
        model.config.block_size = max_len
        model.config.rope_scaling = None
    else:
        model.config.block_size = train_block
        model.config.rope_scaling = {"type": "ntk", "factor": float(factor)}
    for module in model.modules():
        if not isinstance(module, RotaryEmbedding):
            continue
        theta = ntk_theta(module.dim, factor)
        inv = 1.0 / (theta ** (torch.arange(0, module.dim, 2).float() / module.dim))
        module.inv_freq.copy_(inv.to(device=module.inv_freq.device, dtype=module.inv_freq.dtype))
        module.max_seq_len = max_len
        module.base = theta
        module._cos_cached = None
        module._sin_cached = None


def windows(data, length, batch_size, num_batches, seed):
    """Yield fixed token windows. The same seed gives the same windows."""
    rng = np.random.default_rng(seed)
    usable = len(data) - length - 1
    if usable <= 0:
        raise RuntimeError(f"val set shorter than context {length}")
    for _ in range(num_batches):
        starts = rng.integers(0, usable, size=batch_size)
        batch = np.stack([data[s : s + length + 1] for s in starts])
        yield batch


@torch.no_grad()
def perplexity(model, data, length, batch_size, num_batches, seed, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in windows(data, length, batch_size, num_batches, seed):
        tokens = torch.from_numpy(batch.astype(np.int64)).to(device)
        x = tokens[:, :-1].contiguous()
        y = tokens[:, 1:].contiguous()
        logits, loss, _ = model(x, targets=y)
        if loss is None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += float(loss.item()) * y.numel()
        total_tokens += y.numel()
    nll = total_loss / max(total_tokens, 1)
    return {"nll": nll, "ppl": math.exp(nll), "tokens": total_tokens}


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()
    data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")
    rows = []
    max_len = max(args.lengths)
    for length in args.lengths:
        factor = max(1.0, length / float(args.train_block))
        for name, used_factor in (("naive", 1.0), ("ntk", factor)):
            retune_rope(model, used_factor, max_len, args.train_block)
            stats = perplexity(
                model,
                data,
                length,
                args.batch_size,
                args.num_batches,
                args.seed + length,
                device,
            )
            row = {
                "length": length,
                "method": name,
                "factor": used_factor,
                **stats,
            }
            rows.append(row)
            print(
                f"L={length} {name} factor={used_factor:.2f} "
                f"ppl={stats['ppl']:.3f} nll={stats['nll']:.4f}",
                flush=True,
            )
    payload = {
        "checkpoint": args.checkpoint,
        "train_block": args.train_block,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "rows": rows,
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print("wrote", args.output_json)


if __name__ == "__main__":
    main()
