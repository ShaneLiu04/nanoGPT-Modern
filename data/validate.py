"""Validate tokenized binary datasets.

Checks: token range, vocabulary coverage, EOT frequency, decoded samples.
"""
import os
import sys
import argparse
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("bin_path", type=str)
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--eot_token", type=int, default=50256)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="Random seed for decoded samples")
    p.add_argument("--histogram_tokens", type=int, default=10_000_000,
                   help="Number of leading tokens to use for the top-token histogram")
    return p.parse_args()


def main():
    args = get_args()
    if not os.path.exists(args.bin_path):
        print(f"File not found: {args.bin_path}")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    # detect dtype
    idx_path = args.bin_path.replace(".bin", ".idx")
    dtype = np.uint16
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            for line in f:
                if line.startswith("dtype="):
                    if line.strip().split("=")[1] == "uint32":
                        dtype = np.uint32
                    break

    data = np.memmap(args.bin_path, dtype=dtype, mode="r")
    total = len(data)

    print(f"File:     {args.bin_path}")
    print(f"Dtype:    {dtype.__name__}")
    print(f"Tokens:   {total:,}")
    print(f"Size:     {os.path.getsize(args.bin_path)/1024**2:.1f} MB")
    print()

    # token range
    min_tok = int(data.min())
    max_tok = int(data.max())
    in_range = max_tok < args.vocab_size
    print(f"Token range:     {min_tok} - {max_tok}")
    print(f"Vocab check:     {'OK' if in_range else 'FAIL - tokens exceed vocab_size=' + str(args.vocab_size)}")
    print()

    # EOT frequency
    eot_count = int((data == args.eot_token).sum())
    eot_ratio = eot_count / total if total > 0 else 0
    print(f"EOT count:       {eot_count:,} ({eot_ratio*100:.2f}%)")
    print(f"Avg doc length:  {total/max(eot_count,1):.0f} tokens")
    print()

    # token histogram (top-20)
    print("Top-20 tokens:")
    hist_tokens = min(total, max(args.histogram_tokens, 1))
    counts = np.bincount(data[:hist_tokens].astype(np.int64))
    top_indices = np.argsort(counts)[-20:][::-1]
    for i, idx in enumerate(top_indices):
        pct = counts[idx] / hist_tokens * 100
        bar = "#" * int(pct * 5)
        print(f"  {i+1:2d}. token {idx:5d}: {counts[idx]:>10,} ({pct:5.2f}%) {bar}")
    print()

    # decoded samples
    try:
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")
        print(f"Random samples ({args.num_samples}):")
        for i in range(args.num_samples):
            start = rng.integers(0, max(total - 200, 1))
            segment = data[start:start+100]
            text = tokenizer.decode(segment.tolist())
            text = text.replace("\n", "\\n")
            print(f"  [{start:>10,}]: {text[:200]}...")
    except ImportError:
        print("(install tiktoken for decoded samples)")


if __name__ == "__main__":
    main()
