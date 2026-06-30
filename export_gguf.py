"""Export a nanoGPT-Modern checkpoint to GGUF format.

Uses the in-tree ``model.gguf_utils`` writer so the export works without the
external ``gguf`` package.  Supported output types are ``f32``, ``f16`` and
``q8_0``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from model.gguf_utils import GGMLQuantizationType, GGUFWriter
from model.modern_gpt import ModernGPT, ModernGPTConfig

_QUANT_MAP = {
    "f32": GGMLQuantizationType.F32,
    "f16": GGMLQuantizationType.F16,
    "q8_0": GGMLQuantizationType.Q8_0,
}


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export nanoGPT-Modern checkpoint to GGUF")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a nanoGPT-Modern checkpoint (.pt)",
    )
    p.add_argument("--out", type=str, required=True, help="Output .gguf file path")
    p.add_argument(
        "--quant",
        type=str,
        default="q8_0",
        choices=list(_QUANT_MAP.keys()),
        help="Tensor quantization type (default: q8_0)",
    )
    p.add_argument(
        "--name",
        type=str,
        default=None,
        help="Human-readable model name written to metadata",
    )
    p.add_argument(
        "--f16-for-1d",
        action="store_true",
        help="Store 1-D buffers/norms in F16 even when using Q8_0",
    )
    return p.parse_args()


def _tensor_quant_type(
    name: str, tensor: torch.Tensor, quant: GGMLQuantizationType, f16_for_1d: bool
) -> GGMLQuantizationType:
    """Decide which GGUF dtype a tensor should be stored as."""
    if quant in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        return quant
    if tensor.dim() == 1:
        return GGMLQuantizationType.F16 if f16_for_1d else GGMLQuantizationType.F32
    return GGMLQuantizationType.Q8_0


def main() -> None:
    args = get_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    raw_config = ckpt.get("config", {})
    if isinstance(raw_config, dict):
        config = ModernGPTConfig.from_dict(raw_config)
    else:
        config = raw_config

    model = ModernGPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    quant_type = _QUANT_MAP[args.quant]
    architecture = "nanogpt-modern"
    writer = GGUFWriter(out_path, architecture=architecture)

    name = args.name or (Path(args.checkpoint).stem + f"-{args.quant}")
    writer.add_name(name)
    writer.add_context_length(config.effective_block_size)
    writer.add_embedding_length(config.n_embd)
    writer.add_block_count(config.n_layer)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.n_head)
    writer.add_head_count_kv(config.n_kv_head)

    state_dict = model.state_dict()
    for key, tensor in state_dict.items():
        ttype = _tensor_quant_type(key, tensor, quant_type, args.f16_for_1d)
        writer.add_tensor(key, tensor, quant_type=ttype)

    writer.write()

    file_size = os.path.getsize(out_path)
    print(f"Exported {args.quant.upper()} GGUF to {out_path}")
    print(f"  tensors: {len(state_dict)}  size: {file_size / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
