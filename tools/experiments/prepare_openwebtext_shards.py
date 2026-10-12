"""Download official OpenWebText parquet shards and tokenize them.

``datasets`` 5.x cannot stream the legacy OpenWebText loading script, and the
Hugging Face hub client on this image fails while listing the repo. This
script uses curl against the public parquet files (the same
``Skylion007/openwebtext`` release, ``plain_text/train-XXXXX-of-00080``)
and tiktoken GPT-2 encoding, then writes the ``{split}.bin`` + ``.idx``
layout ``data/openwebtext.py`` already trains on.

Disk policy: each shard is deleted after it is tokenized. Training shards
stop once ``--target_tokens`` is reached. One held-out shard supplies the
validation documents.
"""

import argparse
import os
import subprocess
import sys

import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/openwebtext")
    parser.add_argument("--target_tokens", type=int, default=1_200_000_000)
    parser.add_argument("--max_train_shards", type=int, default=16)
    parser.add_argument("--val_shard", type=int, default=79)
    parser.add_argument("--val_docs", type=int, default=10_000)
    parser.add_argument("--n_shards_total", type=int, default=80)
    parser.add_argument(
        "--base_url",
        default="https://hf-mirror.com/datasets/Skylion007/openwebtext/resolve/main",
    )
    parser.add_argument("--eot_token", type=int, default=50256)
    return parser.parse_args()


def shard_name(index, n_total):
    return f"plain_text/train-{index:05d}-of-{n_total:05d}.parquet"


def _header_value(text, name):
    prefix = name.lower() + ":"
    for line in text.splitlines():
        if line.lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def download(url, dest, connections=12):
    """Download ``url`` with parallel HTTP ranges. The mirror is slow per connection."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print("download", url, flush=True)
    head = subprocess.check_output(["curl", "-sI", url], text=True)
    loc = _header_value(head, "location") or url
    remote = subprocess.check_output(["curl", "-sI", loc], text=True)
    size = int(_header_value(remote, "content-length") or "0")
    if size <= 0:
        raise RuntimeError(f"could not read content-length for {url}")
    chunk = (size + connections - 1) // connections
    parts = []
    procs = []
    for index in range(connections):
        start = index * chunk
        if start >= size:
            break
        end = min(size - 1, start + chunk - 1)
        part = f"{dest}.p{index}"
        parts.append(part)
        procs.append(
            subprocess.Popen(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--retry",
                    "5",
                    "--range",
                    f"{start}-{end}",
                    "-o",
                    part,
                    loc,
                ]
            )
        )
    failed = [proc.wait() for proc in procs]
    if any(code != 0 for code in failed):
        raise RuntimeError(f"range download failed: {failed}")
    with open(dest, "wb") as out:
        for part in parts:
            with open(part, "rb") as inp:
                out.write(inp.read())
            os.remove(part)
    got = os.path.getsize(dest)
    if got != size:
        raise RuntimeError(f"size mismatch for {dest}: {got} != {size}")


def read_texts(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["text"])
    return table.column("text").to_pylist()


def tokenize_docs(texts, tokenizer, eot, dtype, limit=None):
    """Encode documents and return one uint array with an EOT after each doc."""
    kept = []
    for text in texts:
        if not isinstance(text, str) or not text:
            continue
        kept.append(text)
        if limit is not None and len(kept) >= limit:
            break
    if not kept:
        return np.empty(0, dtype=dtype), 0
    encoded = tokenizer.encode_ordinary_batch(kept)
    parts = [np.asarray(toks + [eot], dtype=dtype) for toks in encoded]
    return np.concatenate(parts), len(parts)


def write_index(path, dtype, total_tokens, num_docs, eot):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"dtype={np.dtype(dtype).name}\n")
        handle.write("vocab_size=50257\n")
        handle.write(f"total_tokens={total_tokens}\n")
        handle.write(f"num_docs={num_docs}\n")
        handle.write(f"eot_token={eot}\n")
        handle.write("source=Skylion007/openwebtext parquet shards\n")


def append_tokens(bin_path, tokens):
    with open(bin_path, "ab") as handle:
        tokens.tofile(handle)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cache = os.path.join(args.output_dir, "_parquet_cache")
    os.makedirs(cache, exist_ok=True)

    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    dtype = np.uint16

    train_bin = os.path.join(args.output_dir, "train.bin")
    val_bin = os.path.join(args.output_dir, "val.bin")
    for path in (train_bin, val_bin):
        if os.path.exists(path):
            os.remove(path)

    train_tokens = 0
    train_docs = 0
    for index in range(args.max_train_shards):
        if train_tokens >= args.target_tokens:
            break
        name = shard_name(index, args.n_shards_total)
        local = os.path.join(cache, os.path.basename(name))
        download(args.base_url + "/" + name, local)
        texts = read_texts(local)
        tokens, n_docs = tokenize_docs(texts, tokenizer, args.eot_token, dtype)
        append_tokens(train_bin, tokens)
        train_tokens += int(tokens.size)
        train_docs += n_docs
        os.remove(local)
        print(
            f"train shard {index}: +{n_docs:,} docs, " f"running {train_tokens:,} tokens",
            flush=True,
        )

    val_name = shard_name(args.val_shard, args.n_shards_total)
    val_local = os.path.join(cache, os.path.basename(val_name))
    download(args.base_url + "/" + val_name, val_local)
    val_tokens, val_docs = tokenize_docs(
        read_texts(val_local),
        tokenizer,
        args.eot_token,
        dtype,
        limit=args.val_docs,
    )
    append_tokens(val_bin, val_tokens)
    os.remove(val_local)
    os.rmdir(cache)

    write_index(
        os.path.join(args.output_dir, "train.idx"),
        dtype,
        train_tokens,
        train_docs,
        args.eot_token,
    )
    write_index(
        os.path.join(args.output_dir, "val.idx"),
        dtype,
        int(val_tokens.size),
        val_docs,
        args.eot_token,
    )
    print(
        f"done train_tokens={train_tokens:,} train_docs={train_docs:,} "
        f"val_tokens={int(val_tokens.size):,} val_docs={val_docs:,}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        sys.exit(1)
