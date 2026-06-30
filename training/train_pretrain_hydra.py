"""Hydra entry point for pretraining.

This script mirrors ``train_pretrain.py`` but uses Hydra/OmegaConf for
configuration composition.  The legacy argparse-based entry point remains
available for backward compatibility.
"""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig


from model.attention_utils import set_attention_backend, print_attention_backend
from training.train_pretrain import PretrainTrainer
from utils.hydra_utils import to_namespace


@hydra.main(config_path="../config/hydra", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    args: Any = to_namespace(cfg)
    set_attention_backend(getattr(args, "attn_backend", "auto"))
    print_attention_backend()
    trainer = PretrainTrainer(args)
    try:
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
