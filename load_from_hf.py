"""Load a HuggingFace-format nanoGPT-Modern checkpoint and save it back as .pt."""
import argparse
import os

import torch

from model.hf_model import NanoGPTModernForCausalLM


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dir", type=str, required=True,
                   help="Directory containing config.json and model.safetensors")
    p.add_argument("--out", type=str, required=True,
                   help="Output .pt checkpoint path")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    wrapper = NanoGPTModernForCausalLM.from_pretrained(args.hf_dir)
    nano_config = wrapper.config.to_nanogpt_config()
    ckpt = {
        "model": wrapper.model.state_dict(),
        "config": nano_config.to_dict(),
    }
    torch.save(ckpt, args.out)
    print(f"Loaded HF checkpoint from {args.hf_dir} and saved nanoGPT checkpoint to {args.out}")


if __name__ == "__main__":
    main()
