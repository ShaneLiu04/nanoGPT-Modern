"""Hydra entry point for GRPO alignment."""

from __future__ import annotations

import os
from typing import Any

import hydra
from omegaconf import DictConfig


from model.attention_utils import set_attention_backend, print_attention_backend
from training.train_grpo import GRPOTrainer
from utils.hydra_utils import to_namespace


@hydra.main(config_path="../config/hydra", config_name="grpo", version_base=None)
def main(cfg: DictConfig) -> None:
    args: Any = to_namespace(cfg)
    set_attention_backend(getattr(args, "attn_backend", "auto"))
    print_attention_backend()
    trainer = GRPOTrainer(args)
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
