"""Hydra entry point for iterative GRPO with rejection sampling."""
from __future__ import annotations

import os
from typing import Any

import hydra
from omegaconf import DictConfig


from training.iterative_grpo import IterativeGRPOTrainer
from utils.hydra_utils import to_namespace


@hydra.main(config_path="../config/hydra", config_name="grpo", version_base=None)
def main(cfg: DictConfig) -> None:
    args: Any = to_namespace(cfg)
    trainer = IterativeGRPOTrainer(args)
    try:
        trainer.train()
    finally:
        trainer.finalize()
        trainer.save_checkpoint(
            f"final_iterative_grpo_g{args.group_size}.pt",
            trainer.global_step,
            trainer.best_reward,
        )
        trainer.cleanup()


if __name__ == "__main__":
    main()
