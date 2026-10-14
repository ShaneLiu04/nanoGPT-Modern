"""Pure-CPU data pipeline probe (AR-v0.7.5 FR6, task T006).

Deep-dive companion for docs/ai-infra/data-pipeline.md (topic 10). Real-profile
experiments memmap-sample a uint16 token bin (default: the repo's OpenWebText
train.bin); synthetic experiments run on tool-internal corpora so the test
suite never depends on the 17GB file. The MinHash dedup curve calls the
production implementation in data/dedup.py (read-only import).

CLI:  python tools/infra/data_probe.py [--bin PATH] [--quick] [--output-json PATH]
Exit: 0 success (incl. missing-bin degradation: synthetic experiments still
      run) / 1 interrupted (no partial JSON) / 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = REPO_ROOT / "data" / "openwebtext" / "train.bin"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "results" / "data_probe.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dedup import MinHashDeduplicator  # noqa: E402 -- production impl, read-only

EOT = 50256  # GPT-2 end-of-text token (vocab 50257, uint16 ceiling)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data pipeline probe (data-pipeline.md companion): memmap "
        "token-bin profiling + synthetic dedup/packing experiments."
    )
    parser.add_argument("--bin", default=str(DEFAULT_BIN), help="uint16 token bin path")
    parser.add_argument("--quick", action="store_true", help="shrink sample sizes")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Real-profile experiments (source: real)
# ---------------------------------------------------------------------------


def exp_bin_profile(bin_path: Path, quick: bool) -> dict:
    """Strided memmap sample: dtype/value range/EOT frequency/top tokens.

    Stride sampling keeps the 17GB file out of the page cache (design D4):
    sample_n positions evenly spread, one page-fault per sample batch.
    """
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    total = int(data.shape[0])
    sample_n = 50_000 if quick else 200_000
    stride = max(1, total // sample_n)
    sampled = np.asarray(data[::stride])
    values, counts = np.unique(sampled, return_counts=True)
    order = np.argsort(-counts)[:10]
    top_tokens = [[int(values[i]), float(counts[i] / sampled.size)] for i in order]
    eot_freq = float(np.mean(sampled == EOT))
    return {
        "name": "bin_profile",
        "source": "real",
        "size_bytes": bin_path.stat().st_size,
        "dtype": "uint16",
        "total_tokens": total,
        "sampled_tokens": int(sampled.size),
        "stride": stride,
        "value_range": [int(sampled.min()), int(sampled.max())],
        "eot_freq": eot_freq,
        "top_tokens": top_tokens,
    }


def exp_doc_length_dist(bin_path: Path, quick: bool) -> dict:
    """EOT-separated document lengths over contiguous windows.

    Doc lengths need contiguous token context (strided sampling would distort
    gaps), so this experiment reads windows instead of a strided sample.
    """
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    total = int(data.shape[0])
    window = 500_000 if quick else 2_000_000
    if total <= window:
        segments = [np.asarray(data)]
    else:
        k = 8
        w = min(window, total // k)
        segments = [np.asarray(data[int(i * (total - w) / (k - 1)) :][:w]) for i in range(k)]

    lengths: list[int] = []
    for seg in segments:
        eot_positions = np.where(seg == EOT)[0]
        prev = -1
        for pos in eot_positions:
            lengths.append(int(pos - prev))
            prev = int(pos)

    if not lengths:
        return {
            "name": "doc_length_dist",
            "source": "real",
            "note": "no EOT found in windows",
            "mean": None,
            "p50": None,
            "p99": None,
            "n_docs": 0,
        }
    arr = np.asarray(lengths, dtype=float)
    return {
        "name": "doc_length_dist",
        "source": "real",
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "n_docs": len(lengths),
        "note": "doc length = gap between EOT positions (contiguous windows)",
    }


# ---------------------------------------------------------------------------
# Synthetic-corpus experiments (source: synthetic)
# ---------------------------------------------------------------------------


def exp_contamination_demo() -> dict:
    """Needle contamination demo: planted needles 100% found, 0 false hits."""
    rng = np.random.default_rng(1337)
    needle = [9001, 7007, 5003]
    docs = [rng.integers(0, 40_000, 150).tolist() for _ in range(20)]
    planted = (0, 3, 7)
    for i in planted:
        docs[i] = docs[i][:20] + needle + docs[i][20:]

    def contains(seq: list[int], pat: list[int]) -> bool:
        pat_len = len(pat)
        return any(seq[j : j + pat_len] == pat for j in range(len(seq) - pat_len + 1))

    hits = sum(contains(docs[i], needle) for i in planted) / len(planted)
    false_positives = sum(contains(docs[i], needle) for i in range(20) if i not in planted) / (
        20 - len(planted)
    )
    return {
        "name": "contamination_demo",
        "source": "synthetic",
        "needle": needle,
        "n_docs": 20,
        "n_planted": len(planted),
        "needle_hit_rate": float(hits),
        "false_positive_rate": float(false_positives),
        "note": "detection = substring search + threshold; see data-pipeline.md section 9",
    }


def exp_minhash_dedup_curve(quick: bool) -> dict:
    """Duplicate-rate sweep against the production MinHash+LSH deduplicator.

    At each dup rate a corpus is built where the first `n_dup` of n_docs-1
    docs are exact copies of a base document; detected rate is the fraction
    the production deduplicator flags. Exact duplicates must be caught
    (~all copies), independent docs must survive (~zero kills).
    """
    dup_rates = (0.0, 0.5, 1.0) if quick else (0.0, 0.25, 0.5, 0.75, 1.0)
    n_docs = 12
    detected: list[float] = []
    for rate in dup_rates:
        rng = np.random.default_rng(20261014)
        base = " ".join(f"base{rng.integers(0, 10**6)}" for _ in range(30))
        n_dup = int(round(rate * (n_docs - 1)))
        texts = [base]
        for i in range(1, n_docs):
            if i <= n_dup:
                texts.append(base)
            else:
                texts.append(" ".join(f"w{rng.integers(0, 10**6)}" for _ in range(30)))
        dedup = MinHashDeduplicator(num_hashes=128, num_bands=8, rows_per_band=16, threshold=0.85)
        dedup.fit(texts)
        mask = dedup.duplicate_mask()  # True = keep, False = drop (later dups)
        detected.append(float(1.0 - np.mean(np.asarray(mask))))
    return {
        "name": "minhash_dedup_curve",
        "source": "synthetic",
        "dup_rates": list(dup_rates),
        "detected_rates": detected,
        "band_widths": [8, 16],
        "num_hashes": 128,
        "threshold": 0.85,
        "implementation": "data/dedup.py MinHashDeduplicator (production, read-only)",
    }


def _greedy_pack_lengths() -> tuple:
    """One greedy packing pass shared by greedy_packing + doc_boundary_stats."""
    rng = np.random.default_rng(20261014)
    lengths = rng.integers(50, 150, 300)
    block = 1024
    blocks_fill: list[int] = []
    docs_per_block: list[int] = []
    cur = docs_here = 0
    for doc_len in lengths:
        if cur + doc_len > block and cur > 0:
            blocks_fill.append(cur)
            docs_per_block.append(docs_here)
            cur = docs_here = 0
        cur += int(doc_len)
        docs_here += 1
    if cur:
        blocks_fill.append(cur)
        docs_per_block.append(docs_here)
    return lengths, len(blocks_fill), sum(docs_per_block), blocks_fill, docs_per_block


def exp_greedy_packing() -> dict:
    """Greedy packing on a short-doc synthetic corpus (fill ratio vs padding)."""
    lengths, n_blocks, _, fills, _ = _greedy_pack_lengths()
    total_tokens = int(lengths.sum())
    capacity = n_blocks * 1024
    fill_ratio = total_tokens / capacity
    return {
        "name": "greedy_packing",
        "source": "synthetic",
        "block_size": 1024,
        "n_docs": int(lengths.size),
        "n_blocks": n_blocks,
        "fill_ratio": float(fill_ratio),
        "padding_waste": float(1.0 - fill_ratio),
        "note": "padding = block capacity not covered by doc tokens; cf. "
        "PackingDataset._build_samples (data/openwebtext.py:349)",
    }


def exp_doc_boundary_stats() -> dict:
    """doc_boundary semantics on the same greedy packing run."""
    lengths, n_blocks, total_docs, fills, docs_per_block = _greedy_pack_lengths()
    cross = sum(1 for d in docs_per_block if d > 1) / n_blocks
    return {
        "name": "doc_boundary_stats",
        "source": "synthetic",
        "block_size": 1024,
        "docs_per_block_mean": float(total_docs / n_blocks),
        "cross_doc_fraction": float(cross),
        "note": "cross_doc_fraction = blocks containing >=2 documents; the "
        "document_ids tensor masks attention across these boundaries",
    }


def _summary_md(meta: dict, experiments: list[dict]) -> str:
    lines = [f"# data_probe (quick={meta['quick']}, device=cpu)", ""]
    for exp in experiments:
        if exp.get("skipped"):
            lines.append(f"- {exp['name']}: SKIPPED ({exp['skipped']})")
        elif exp["name"] == "bin_profile":
            lines.append(
                f"- bin_profile: {exp['sampled_tokens']} tokens sampled, "
                f"eot_freq={exp['eot_freq']:.2e}, value_range={exp['value_range']}"
            )
        elif exp["name"] == "doc_length_dist":
            lines.append(f"- doc_length_dist: mean={exp['mean']}, n_docs={exp['n_docs']}")
        elif exp["name"] == "minhash_dedup_curve":
            lines.append(
                f"- minhash_dedup_curve: dup={exp['dup_rates']} -> "
                f"detected={[round(d, 3) for d in exp['detected_rates']]}"
            )
        elif exp["name"] == "greedy_packing":
            lines.append(f"- greedy_packing: fill_ratio={exp['fill_ratio']:.3f}")
        elif exp["name"] == "contamination_demo":
            lines.append(
                f"- contamination_demo: hit={exp['needle_hit_rate']:.0%}, "
                f"fp={exp['false_positive_rate']:.0%}"
            )
        else:
            lines.append(f"- doc_boundary_stats: docs/block={exp['docs_per_block_mean']:.1f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bin_path = Path(args.bin)
    output = Path(args.output_json)

    if not output.parent.is_dir():
        print(
            f"error: --output-json parent directory does not exist: {output.parent}",
            file=sys.stderr,
        )
        return 2

    bin_available = bin_path.is_file() and bin_path.stat().st_size > 0
    if not bin_available:
        print(
            f"warning: token bin not found (or empty), skipping real profiling: {bin_path}",
            file=sys.stderr,
        )

    try:
        experiments: list[dict] = []
        if bin_available:
            experiments.append(exp_bin_profile(bin_path, args.quick))
            experiments.append(exp_doc_length_dist(bin_path, args.quick))
        else:
            experiments.append(
                {"name": "bin_profile", "source": "real", "skipped": f"bin not found: {bin_path}"}
            )
            experiments.append(
                {
                    "name": "doc_length_dist",
                    "source": "real",
                    "skipped": f"bin not found: {bin_path}",
                }
            )
        experiments.append(exp_contamination_demo())
        experiments.append(exp_minhash_dedup_curve(args.quick))
        experiments.append(exp_greedy_packing())
        experiments.append(exp_doc_boundary_stats())

        meta = {
            "tool": "data_probe",
            "device": "cpu",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "quick": bool(args.quick),
            "bin_path": str(bin_path),
            "bin_available": bin_available,
        }
        payload = {"meta": meta, "experiments": experiments}
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, output)
    except Exception as exc:  # noqa: BLE001 -- single exit-1 funnel, no partial JSON
        print(f"error: measurement interrupted: {exc}", file=sys.stderr)
        return 1

    print(_summary_md(meta, experiments))
    return 0


if __name__ == "__main__":
    sys.exit(main())
