"""Export nanoGPT-Modern checkpoints to ONNX format with dynamic axes.

Supports both ``BaselineGPT`` and ``ModernGPT`` architectures.  Exported ONNX
graphs use dynamic axes for ``batch_size`` and ``seq_len`` so that a single
``.onnx`` file can serve variable-length inputs at runtime.

Includes a validation helper that compares PyTorch and ONNX Runtime outputs
with the same random input.

Usage
-----
    python export_to_onnx.py --checkpoint out/pretrain/best_ckpt.pt --model modern
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.onnx import export as onnx_export

from model.baseline_gpt import BaselineGPT, BaselineGPTConfig
from model.modern_gpt import ModernGPT, ModernGPTConfig


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export nanoGPT-Modern checkpoint to ONNX"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to .pt checkpoint"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="modern",
        choices=["baseline", "modern"],
        help="Model architecture",
    )
    parser.add_argument(
        "--out_path", type=str, default=None, help="Output .onnx file path"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default 17)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Dummy batch size for tracing"
    )
    parser.add_argument(
        "--seq_len", type=int, default=8, help="Dummy sequence length for tracing"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run ONNX Runtime validation after export",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the model on",
    )
    return parser.parse_args()


def _load_model(ckpt_path: str, model_type: str, device: str) -> nn.Module:
    """Load a checkpoint and return the model on ``device``."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_config = ckpt.get("config", {})

    config: Union[BaselineGPTConfig, ModernGPTConfig]
    model: Union[BaselineGPT, ModernGPT]
    if isinstance(raw_config, dict):
        if model_type == "baseline":
            config = BaselineGPTConfig.from_dict(raw_config)
            model = BaselineGPT(config)
        else:
            config = ModernGPTConfig.from_dict(raw_config)
            model = ModernGPT(config)
    else:
        config = raw_config
        if model_type == "baseline":
            model = BaselineGPT(config)
        else:
            model = ModernGPT(config)  # type: ignore[arg-type]

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


class _OnnxWrapper(nn.Module):
    """Wrapper that strips auxiliary outputs and returns only logits.

    ONNX export works best when the forward signature returns a simple tensor.
    ModernGPT returns ``(logits, loss, past_kvs, ...)``; this wrapper returns
    just ``logits``.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        out = self.model(idx)
        if isinstance(out, tuple):
            # (logits, loss, ...) or (logits, loss, past_kvs, ...)
            return out[0]
        return out


class _OnnxCacheWrapper(nn.Module):
    """Wrapper for cache-enabled (KV-cache) forward pass.

    This exports a single-token decode step (batch=1, seq_len=1) with
    explicit past KV tensors.  Useful for ONNX Runtime inference engines
    that manage their own KV cache.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        config = model.config
        self.n_layers = config.n_layer
        self.n_head = config.n_head
        self.n_kv_head = getattr(config, "n_kv_head", config.n_head)
        self.head_dim = config.n_embd // config.n_head

    def forward(
        self,
        idx: torch.Tensor,
        *past_kvs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward with explicit past KV pairs.

        Parameters
        ----------
        idx : [B, 1]
        past_kvs : flattened list of (k, v) tensors per layer

        Returns
        -------
        logits : [B, 1, V]
        present_kvs : flattened list of (k, v) tensors per layer
        """
        # Reconstruct nested list from flat args
        n_layers = self.n_layers
        assert len(past_kvs) == 2 * n_layers
        past_kv_list = [
            (past_kvs[2 * i], past_kvs[2 * i + 1]) for i in range(n_layers)
        ]

        logits, _, new_past_kvs = self.model(
            idx, past_kvs=past_kv_list, use_cache=True, start_pos=0
        )
        # Flatten new KV pairs into a tuple
        flat = cast(Tuple[torch.Tensor, ...], tuple(t for pair in new_past_kvs for t in pair))
        return logits, flat


def export_to_onnx(
    model: nn.Module,
    out_path: str,
    batch_size: int = 2,
    seq_len: int = 8,
    opset: int = 17,
    use_cache: bool = False,
) -> str:
    """Export ``model`` to ONNX at ``out_path``.

    Parameters
    ----------
    model : nn.Module
        BaselineGPT or ModernGPT instance.
    out_path : str
        Destination ``.onnx`` file.
    batch_size, seq_len : int
        Dummy input dimensions for tracing.
    opset : int
        ONNX opset version.
    use_cache : bool
        If True, export the cache-enabled decode step (KV cache) instead
        of the standard training forward.

    Returns
    -------
    str
        The output path (same as ``out_path``).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size

    wrapper: nn.Module
    if use_cache:
        wrapper = _OnnxCacheWrapper(model)
        dummy_idx = torch.randint(
            0, vocab_size, (batch_size, 1), dtype=torch.long, device=device
        )
        dummy_past = []
        for _ in range(model.config.n_layer):
            past_k = torch.zeros(
                batch_size,
                wrapper.n_kv_head,
                seq_len,
                wrapper.head_dim,
                device=device,
                dtype=torch.float32,
            )
            past_v = torch.zeros_like(past_k)
            dummy_past.extend([past_k, past_v])
        dummy_input = (dummy_idx, *dummy_past)

        # Build dynamic axes for each input and output
        n_layers = model.config.n_layer
        input_names = ["input_ids"] + [
            f"past_k_{i}" for i in range(n_layers)
        ] + [f"past_v_{i}" for i in range(n_layers)]
        output_names = ["logits"] + [
            f"present_k_{i}" for i in range(n_layers)
        ] + [f"present_v_{i}" for i in range(n_layers)]
        dynamic_axes: Dict[str, Dict[int, str]] = {}
        dynamic_axes["input_ids"] = {0: "batch_size", 1: "seq_len"}
        for i in range(n_layers):
            dynamic_axes[f"past_k_{i}"] = {0: "batch_size", 2: "past_seq_len"}
            dynamic_axes[f"past_v_{i}"] = {0: "batch_size", 2: "past_seq_len"}
        dynamic_axes["logits"] = {0: "batch_size", 1: "seq_len"}
        for i in range(n_layers):
            dynamic_axes[f"present_k_{i}"] = {0: "batch_size", 2: "total_seq_len"}
            dynamic_axes[f"present_v_{i}"] = {0: "batch_size", 2: "total_seq_len"}

        onnx_export(
            wrapper,
            dummy_input,
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
    else:
        wrapper = _OnnxWrapper(model)
        dummy_idx = torch.randint(
            0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device
        )
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }
        onnx_export(
            wrapper,
            dummy_idx,
            out_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    print(f"ONNX model exported to {out_path}")
    return out_path


def validate_onnx(
    onnx_path: str,
    model: nn.Module,
    batch_size: int = 2,
    seq_len: int = 8,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> bool:
    """Compare PyTorch and ONNX Runtime outputs on random data.

    Returns ``True`` if outputs match within tolerance.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime not installed; skipping validation.")
        return False

    device = next(model.parameters()).device
    vocab_size = model.config.vocab_size
    dummy_idx = torch.randint(
        0, vocab_size, (batch_size, seq_len), dtype=torch.long, device=device
    )

    with torch.no_grad():
        pt_out = model(dummy_idx)
        if isinstance(pt_out, tuple):
            pt_logits = pt_out[0]
        else:
            pt_logits = pt_out

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_idx.cpu().numpy()}
    ort_outs = session.run(None, ort_inputs)
    ort_logits = torch.from_numpy(ort_outs[0]).to(device)

    match = torch.allclose(pt_logits, ort_logits, atol=atol, rtol=rtol)
    max_diff = (pt_logits - ort_logits).abs().max().item()
    print(
        f"[ONNX Validation] max_diff={max_diff:.6e} | atol={atol} | rtol={rtol} | "
        f"{'PASS' if match else 'FAIL'}"
    )
    return match


def main() -> None:
    args = get_args()
    model = _load_model(args.checkpoint, args.model, args.device)

    if args.out_path is None:
        base = os.path.splitext(args.checkpoint)[0]
        args.out_path = f"{base}.onnx"

    export_to_onnx(
        model,
        args.out_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        opset=args.opset,
        use_cache=False,
    )

    if args.validate:
        validate_onnx(
            args.out_path,
            model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )


if __name__ == "__main__":
    main()
