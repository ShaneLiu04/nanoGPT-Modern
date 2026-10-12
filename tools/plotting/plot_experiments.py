"""Build the experiment figures from JSON artifacts.

Figures use a colorblind-safe palette, no background grid, and a title
that states the measured comparison. Matplotlib text is English so the
figures render without a CJK font. Interpretation lives in the docs.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000"]


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "medium",
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 170,
            "savefig.bbox": "tight",
        }
    )


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("figure", path)


def load_metrics(path):
    rows = []
    path = Path(path)
    if not path.is_file():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def series(events, kind, value_key):
    xs, ys = [], []
    for event in events:
        if event.get("type") != kind:
            continue
        if value_key not in event:
            continue
        step = event.get("step", event.get("iter"))
        if step is None:
            continue
        xs.append(int(step))
        ys.append(float(event[value_key]))
    return np.asarray(xs), np.asarray(ys)


def plot_loss_curves(out_root, dest):
    root = Path(out_root)
    groups = [
        "baseline",
        "modern-full",
        "no-rope",
        "no-swiglu",
        "mha",
        "moe",
        "qk-norm",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    endpoints = {}
    for color, name in zip(COLORS, groups):
        path = root / f"ablation-{name}" / "experiments" / f"ablation-{name}" / "metrics.jsonl"
        xs, ys = series(load_metrics(path), "eval", "val_loss")
        if len(xs) == 0:
            continue
        endpoints[name] = float(ys[-1])
        ax.plot(xs, ys, color=color, lw=2.0, label=name)
    if not endpoints:
        plt.close(fig)
        return
    best = min(endpoints, key=endpoints.get)
    worst = max(endpoints, key=endpoints.get)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.set_title(
        f"{best} finishes at {endpoints[best]:.3f}; " f"{worst} finishes at {endpoints[worst]:.3f}"
    )
    ax.legend(ncol=2, fontsize=9)
    save(fig, dest)


def plot_final_bars(summary, dest):
    if not summary:
        return
    rows = [row for row in summary.get("groups", []) if row.get("best_val_loss") is not None]
    if not rows:
        return
    names = [row["name"] for row in rows]
    vals = [row["best_val_loss"] for row in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]
    ax.bar(names, vals, color=colors, width=0.72)
    ax.set_ylabel("Best validation loss")
    finite = [v for v in vals if v == v]
    if finite:
        lo, hi = min(finite), max(finite)
        pad = max(0.01, (hi - lo) * 0.35)
        ax.set_ylim(lo - pad, hi + pad)
    best_i = int(np.argmin(vals))
    ax.set_title(f"{names[best_i]} has the lowest val loss ({vals[best_i]:.3f})")
    for idx, value in enumerate(vals):
        ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    if finite:
        ax.set_ylim(lo - pad, hi + pad * 1.8)
    ax.tick_params(axis="x", rotation=25)
    save(fig, dest)


def plot_ntk(payload, dest):
    if not payload:
        return
    rows = payload.get("rows", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for color, method in zip(COLORS, ("naive", "ntk")):
        pts = [row for row in rows if row["method"] == method]
        if not pts:
            continue
        ax.plot(
            [p["length"] for p in pts],
            [p["ppl"] for p in pts],
            color=color,
            marker="o",
            lw=2.0,
            label=method,
        )
    naive = [p for p in rows if p["method"] == "naive"]
    ntk = [p for p in rows if p["method"] == "ntk"]
    title = "NTK-aware RoPE versus naive extrapolation"
    if naive and ntk:
        title = (
            f"At {naive[-1]['length']} tokens, NTK PPL is {ntk[-1]['ppl']:.0f} "
            f"and naive RoPE is {naive[-1]['ppl']:.0f}"
        )
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Validation perplexity")
    ax.set_title(title)
    ax.legend()
    save(fig, dest)


def _first_rows(rows, key_fn):
    seen = set()
    kept = []
    for row in rows:
        key = key_fn(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def plot_inference(payload, dest_throughput, dest_memory):
    if not payload:
        return
    rows = payload.get("rows", [])
    variants = ["no-cache", "append", "paged", "int8"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    drawn = False
    for color, variant in zip(COLORS, variants):
        pts = _first_rows(
            [
                r
                for r in rows
                if r.get("variant") == variant and "prompt_len" in r and "tok_per_s" in r
            ],
            lambda row: row["prompt_len"],
        )
        if not pts:
            continue
        drawn = True
        ax.plot(
            [p["prompt_len"] for p in pts],
            [p["tok_per_s"] for p in pts],
            color=color,
            marker="o",
            lw=2.0,
            label=variant,
        )
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Decode throughput (tok/s)")
    ax.set_title("Append KV cache stays near 180 tok/s as the prompt grows")
    ax.legend(fontsize=8)
    save(fig, dest_throughput)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    drawn = False
    for color, variant in zip(COLORS, variants):
        pts = _first_rows(
            [
                r
                for r in rows
                if r.get("variant") == variant and "prompt_len" in r and "peak_mb" in r
            ],
            lambda row: row["prompt_len"],
        )
        if not pts:
            continue
        drawn = True
        ax.plot(
            [p["prompt_len"] for p in pts],
            [p["peak_mb"] for p in pts],
            color=color,
            marker="o",
            lw=2.0,
            label=variant,
        )
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Peak GPU memory (MB)")
    ax.set_title("Append and int8 KV caches grow slower than full recompute")
    ax.legend(fontsize=8)
    save(fig, dest_memory)


def plot_serving(payload, dest_spec, dest_continuous):
    if not payload:
        return
    rows = payload.get("rows", [])
    spec = _first_rows(
        [
            r
            for r in rows
            if str(r.get("variant", "")).startswith("speculative") and "tok_per_s" in r
        ],
        lambda row: row.get("draft_len"),
    )
    if spec:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        names = [f"draft {r.get('draft_len')}" for r in spec]
        vals = [r["tok_per_s"] for r in spec]
        ax.bar(names, vals, color=COLORS[4], width=0.72)
        rates = [r.get("accept_rate") for r in spec]
        if rates and all(rate == 0 for rate in rates):
            ax.set_title("Speculative decode accepts nothing with this weak draft")
        else:
            ax.set_title("Speculative decode throughput by draft length")
        ax.set_ylabel("Decode throughput (tok/s)")
        save(fig, dest_spec)
    cont = _first_rows(
        [
            r
            for r in rows
            if str(r.get("variant", "")).startswith("continuous") and "tok_per_s" in r
        ],
        lambda row: row.get("concurrency"),
    )
    if cont:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        cont = sorted(cont, key=lambda row: row.get("concurrency", 0))
        ax.plot(
            [r["concurrency"] for r in cont],
            [r["tok_per_s"] for r in cont],
            color=COLORS[0],
            marker="o",
            lw=2.0,
        )
        ax.set_xlabel("Concurrent requests")
        ax.set_ylabel("Decode throughput (tok/s)")
        ax.set_title(
            f"Continuous batching: {cont[0]['tok_per_s']:.0f} tok/s at 1 request, "
            f"{cont[-1]['tok_per_s']:.0f} at {cont[-1]['concurrency']}"
        )
        save(fig, dest_continuous)


def plot_system(payload, dest):
    if not payload:
        return
    rows = payload.get("rows", [])
    rows = [r for r in rows if r.get("tok_per_s")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    names = [r["name"] for r in rows]
    vals = [r["tok_per_s"] for r in rows]
    ax.bar(names, vals, color=[COLORS[i % len(COLORS)] for i in range(len(names))], width=0.72)
    ax.set_ylabel("Training throughput (tok/s)")
    by_name = {r["name"]: r["tok_per_s"] for r in rows}
    if "bf16" in by_name and "fp32" in by_name and by_name["fp32"] > 0:
        ratio = by_name["bf16"] / by_name["fp32"]
        ax.set_title(f"bf16 trains at {ratio:.2f}x the throughput of fp32")
    else:
        ax.set_title("Training-system throughput on the same 50M recipe")
    ax.tick_params(axis="x", rotation=20)
    save(fig, dest)


def plot_alignment(payload, dest_format, dest_reward):
    if not payload:
        return
    evals = payload.get("evals", [])
    if evals:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        levels = ["easy", "medium", "hard"]
        width = 0.35
        for idx, item in enumerate(evals):
            ys = [item["by_level"].get(level, {}).get("format_pass_rate", 0) for level in levels]
            xs = np.arange(len(levels)) + (idx - 0.5) * width
            ax.bar(xs, ys, width=width, color=COLORS[idx % len(COLORS)], label=item["name"])
        ax.set_xticks(np.arange(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Format pass rate")
        ax.set_title("Answer-format compliance before and after SFT")
        ax.legend()
        save(fig, dest_format)
    curves = payload.get("grpo_curves", [])
    if curves:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for color, curve in zip(COLORS, curves):
            ax.plot(curve["steps"], curve["reward"], color=color, lw=2.0, label=curve["name"])
        ax.set_xlabel("GRPO step")
        ax.set_ylabel("Mean reward")
        last = [curve["reward"][-1] for curve in curves if curve.get("reward")]
        if last:
            ax.set_title(
                f"At step {curves[0]['steps'][-1]}, group rewards sit between "
                f"{min(last):.3f} and {max(last):.3f}"
            )
        else:
            ax.set_title("GRPO reward by group size")
        ax.legend()
        save(fig, dest_reward)


def plot_sft_scale(payload, dest):
    if not payload or not payload.get("rows"):
        return
    rows = payload["rows"]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    labels = [row["label"] for row in rows]
    # _format_score returns at most 0.5 for one valid <answer> block.
    ceiling = 0.5
    values = [row["format_mean"] / ceiling for row in rows]
    ax.bar(labels, values, color=COLORS[0], width=0.72)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Format score / 0.5 ceiling")
    ax.set_title(f"At the largest SFT budget the format score is {max(values):.0%} of its 0.5 cap")
    ax.tick_params(axis="x", rotation=20)
    save(fig, dest)


def plot_group_variance(payload, dest):
    if not payload or not payload.get("rows"):
        return
    rows = payload["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    groups = [row["group_size"] for row in rows]
    ax.plot(
        groups,
        [row["std_error_of_group_mean"] for row in rows],
        color=COLORS[0],
        marker="o",
        lw=2.0,
        label="SE of group mean",
    )
    ax.plot(
        groups,
        [row["within_group_reward_std"] for row in rows],
        color=COLORS[1],
        marker="o",
        lw=2.0,
        label="within-group reward std",
    )
    ax.set_xlabel("Group size")
    ax.set_ylabel("Reward noise")
    ses = [row["std_error_of_group_mean"] for row in rows]
    if ses[-1] < ses[0]:
        title = f"Group {groups[-1]} cuts reward SE from {ses[0]:.3f} to {ses[-1]:.3f}"
    else:
        title = "Within-group reward noise by group size"
    ax.set_title(title)
    ax.legend()
    save(fig, dest)


def plot_spec_compare(payload, dest):
    if not payload or not payload.get("rows"):
        return
    rows = [row for row in payload["rows"] if "accept_rate" in row]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    names = [f"draft {row['draft_len']}" for row in rows]
    ax.bar(names, [row["accept_rate"] for row in rows], color=COLORS[2], width=0.72)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accept rate")
    best = max(row["accept_rate"] for row in rows)
    ax.set_title(f"Matched-width draft accept rate peaks at {best:.0%}")
    save(fig, dest)


def plot_dpo_compare(payload, reward_dest, tax_dest):
    rows = [] if not payload else payload.get("methods") or []
    rows = [row for row in rows if "reward" in row and "kl" in row]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    names = [row["name"] for row in rows]
    ax.bar(names, [row["reward"] for row in rows], color=COLORS[0], width=0.72)
    ax.set_ylabel("Mean rule reward")
    best = max(rows, key=lambda row: row["reward"])
    ax.set_title(f"{best['name']} reaches mean reward {best['reward']:.3f}")
    ax.tick_params(axis="x", rotation=20)
    save(fig, reward_dest)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for i, row in enumerate(rows):
        ax.scatter(
            [row["kl"]],
            [row["reward"]],
            s=80,
            color=COLORS[i % len(COLORS)],
            zorder=3,
            label=row["name"],
        )
    ax.set_ylim(0.0, 0.8)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_xlabel("Mean token log-ratio (log SFT - log policy)")
    ax.set_ylabel("Mean rule reward")
    ax.set_title(
        f"{best['name']} reward {best['reward']:.3f}; "
        f"log-ratio {best['kl']:.2f} on its own samples"
    )
    save(fig, tax_dest)


def plot_moe(payload, dest):
    if not payload or not payload.get("experts"):
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    experts = payload["experts"]
    ax.plot(
        experts, payload["grouped_ms"], color=COLORS[0], marker="o", lw=2.0, label="grouped GEMM"
    )
    ax.plot(
        experts, payload["loop_ms"], color=COLORS[1], marker="o", lw=2.0, label="per-expert loop"
    )
    ax.set_xlabel("Number of experts")
    ax.set_ylabel("Forward time (ms)")
    ratio = payload["loop_ms"][-1] / payload["grouped_ms"][-1]
    ax.set_title(f"At {experts[-1]} experts, grouped GEMM is {ratio:.1f}x faster than the loop")
    ax.legend()
    save(fig, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="out/ablations")
    parser.add_argument("--results", default="out/cloud")
    parser.add_argument("--fig_dir", default="docs/assets/experiments")
    args = parser.parse_args()
    style()
    fig_dir = Path(args.fig_dir)
    results = Path(args.results)
    plot_loss_curves(args.out_root, fig_dir / "ea_val_loss_curves.png")
    plot_final_bars(
        load_json(Path(args.out_root) / "experiments" / "ablation-summary.json"),
        fig_dir / "ea_best_val_loss.png",
    )
    plot_ntk(load_json(results / "ntk_eval.json"), fig_dir / "ea7_ntk_ppl.png")
    inference = load_json(results / "inference.json")
    plot_inference(
        inference,
        fig_dir / "eb_decode_throughput.png",
        fig_dir / "eb_decode_memory.png",
    )
    plot_serving(
        inference,
        fig_dir / "eb_speculative.png",
        fig_dir / "eb_continuous.png",
    )
    plot_system(load_json(results / "system.json"), fig_dir / "ec_train_throughput.png")
    plot_moe(load_json(results / "moe_kernel.json"), fig_dir / "ec3_moe_kernel.png")
    plot_alignment(
        load_json(results / "alignment.json"),
        fig_dir / "ed_format_pass.png",
        fig_dir / "ed_grpo_reward.png",
    )
    plot_sft_scale(load_json(results / "sft_scale.json"), fig_dir / "ed_sft_scale.png")
    plot_group_variance(
        load_json(results / "group_variance.json"), fig_dir / "ed_group_variance.png"
    )
    plot_spec_compare(load_json(results / "spec_matched.json"), fig_dir / "eb_spec_matched.png")
    plot_dpo_compare(
        load_json(results / "dpo_compare.json"),
        fig_dir / "ed_dpo_reward.png",
        fig_dir / "ed_dpo_kl.png",
    )


if __name__ == "__main__":
    main()
