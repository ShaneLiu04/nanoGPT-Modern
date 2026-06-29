"""Export a nanoGPT-Modern checkpoint to HuggingFace format."""
import argparse
import os

import torch

from model.hf_model import NanoGPTModernConfig, NanoGPTModernForCausalLM
from model.modern_gpt import ModernGPTConfig


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a nanoGPT-Modern checkpoint (.pt)")
    p.add_argument("--out_dir", type=str, required=True,
                   help="Destination directory for the HF-format checkpoint")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_config = ckpt.get("config", {})
    if isinstance(raw_config, dict):
        nano_config = ModernGPTConfig.from_dict(raw_config)
    else:
        nano_config = raw_config

    hf_config = NanoGPTModernConfig.from_nanogpt_config(nano_config)
    wrapper = NanoGPTModernForCausalLM(hf_config)
    wrapper.model.load_state_dict(ckpt["model"])
    wrapper.save_pretrained(args.out_dir, safe_serialization=True)
    print(f"Exported HF checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
