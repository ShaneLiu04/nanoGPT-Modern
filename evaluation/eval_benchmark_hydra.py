"""Hydra entry point for standardized benchmark evaluation."""

from __future__ import annotations

import os
from typing import Any

import hydra
from omegaconf import DictConfig


from data.openwebtext import get_dataloader
from evaluation.eval_benchmark import evaluate_ppl, run_lm_eval
from training.trainer_base import load_model_from_checkpoint
from utils.hydra_utils import to_namespace


@hydra.main(
    config_path="../config/hydra", config_name="eval_benchmark", version_base=None
)
def main(cfg: DictConfig) -> None:
    args: Any = to_namespace(cfg)
    device = args.device

    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    model.eval()

    block_size = getattr(model.config, "block_size", args.block_size)
    dataloader = get_dataloader(
        args.data_dir, args.split, args.batch_size, block_size, num_workers=0
    )

    print(f"[Eval] checkpoint={args.checkpoint}")
    ppl = evaluate_ppl(model, dataloader, device, num_batches=args.num_batches)
    print(f"[Eval] {args.split} perplexity: {ppl:.4f}")

    results = {"perplexity": ppl, "split": args.split}

    if args.tasks:
        lm_results = run_lm_eval(args.checkpoint, args.tasks, device)
        results["lm_eval"] = lm_results
        for task, metrics in lm_results.items():
            print(f"[Eval] {task}: {metrics}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            import json

            json.dump(results, f, indent=2)
        print(f"[Eval] results saved to {args.output_json}")


if __name__ == "__main__":
    main()
