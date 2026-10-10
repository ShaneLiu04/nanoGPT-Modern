"""Export a nanoGPT-Modern checkpoint to HuggingFace format."""

import argparse
import os
import os as _os
import sys as _sys

# Repo root on sys.path so this script runs from any cwd after the
# FR10 move into tools/export/ (three levels up to the root).
_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

from model.hf_model import NanoGPTModernConfig, NanoGPTModernForCausalLM
from training.trainer_base import load_model_from_checkpoint


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a nanoGPT-Modern checkpoint (.pt)",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Destination directory for the HF-format checkpoint",
    )
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, _ckpt = load_model_from_checkpoint(
        args.checkpoint, device="cpu", model_type="modern", eval_mode=True
    )

    hf_config = NanoGPTModernConfig.from_nanogpt_config(model.config)
    wrapper = NanoGPTModernForCausalLM(hf_config)
    wrapper.model.load_state_dict(model.state_dict())
    wrapper.save_pretrained(args.out_dir, safe_serialization=True)
    print(f"Exported HF checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
