"""Light-weight post-training quantization helpers.

This module provides a pure-PyTorch per-channel INT8 weight quantizer plus
optional ``bitsandbytes`` wrappers.  The quantized modules are intended for
**inference only**: they keep weights in low precision and de-quantize on the
fly during ``forward``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class QuantConfig:
    """Configuration for ``quantize_model``.

    Parameters
    ----------
    method:
        One of ``"none"``, ``"int8"`` (static per-channel INT8),
        ``"bnb_8bit"`` or ``"bnb_4bit"``.
    skip_modules:
        Substrings that, when present in a module's full name, cause the module
        to be left untouched.  The defaults keep embeddings / norms / the LM
        head in full precision.
    layer_filter:
        Optional regex.  Only linear layers whose full name matches the regex
        are quantized.
    compute_dtype:
        Dtype used for the scale buffer and for on-the-fly dequantization.
    bnb_8bit_threshold:
        Outlier threshold for ``bnb.nn.Linear8bitLt`` (LLM.int8()).
    bnb_4bit_quant_type:
        ``"nf4"`` or ``"fp4"`` for ``bnb.nn.Linear4bit``.
    bnb_4bit_use_double_quant:
        Whether to use nested quantization for 4-bit scales.
    """

    method: str = "int8"
    skip_modules: Tuple[str, ...] = ("wte", "lm_head", "norm", "ln_")
    layer_filter: Optional[str] = None
    compute_dtype: torch.dtype = torch.float16
    bnb_8bit_threshold: float = 6.0
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


class QuantizedLinear(nn.Module):
    """Static per-channel INT8 linear layer.

    For each output channel the weight is quantized to ``int8`` with a single
    FP16/FP32 scale.  The forward pass dequantizes to the input dtype and then
    calls ``F.linear``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "qweight",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        self.register_buffer("scale", torch.ones(out_features, 1, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, module: nn.Linear, dtype: torch.dtype = torch.float16) -> "QuantizedLinear":
        """Create a quantized copy of a ``nn.Linear`` module."""
        qmodule = cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            dtype,
        )
        with torch.no_grad():
            w = module.weight.data.to(dtype)
            # Per-output-channel scale.
            amax = w.abs().amax(dim=1, keepdim=True)
            scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
            qweight = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
            qmodule.qweight.copy_(qweight)
            qmodule.scale.copy_(scale)
            if module.bias is not None:
                assert qmodule.bias is not None
                qmodule.bias.data.copy_(module.bias.data.to(dtype))
        return qmodule

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized weight, mainly for inspection / export."""
        return self.qweight.to(self.scale.dtype) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, self.bias)


def _should_quantize(name: str, module: nn.Module, config: QuantConfig) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if any(skip in name for skip in config.skip_modules):
        return False
    if config.layer_filter is not None and not re.search(config.layer_filter, name):
        return False
    return True


def _get_parent(model: nn.Module, name: str) -> Tuple[nn.Module, str]:
    parts = name.split(".")
    if len(parts) == 1:
        return model, parts[0]
    parent = model.get_submodule(".".join(parts[:-1]))
    return parent, parts[-1]


def _make_bnb_8bit(linear: nn.Linear, config: QuantConfig) -> nn.Module:
    import bitsandbytes as bnb  # type: ignore[import-untyped]

    device = linear.weight.device
    new = bnb.nn.Linear8bitLt(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        has_fp16_weights=False,
        threshold=config.bnb_8bit_threshold,
    )
    new = new.to(device)
    if linear.bias is not None:
        assert new.bias is not None
        new.bias.data.copy_(linear.bias.data)
    new.weight = bnb.nn.Int8Params(
        linear.weight.data,
        requires_grad=False,
        has_fp16_weights=False,
    ).to(device)  # type: ignore[misc]
    return new


def _make_bnb_4bit(linear: nn.Linear, config: QuantConfig) -> nn.Module:
    import bitsandbytes as bnb  # type: ignore[import-untyped]

    device = linear.weight.device
    new = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        compute_dtype=config.compute_dtype,
        compress_statistics=config.bnb_4bit_use_double_quant,
        quant_type=config.bnb_4bit_quant_type,
    )
    new = new.to(device)
    if linear.bias is not None:
        assert new.bias is not None
        new.bias.data.copy_(linear.bias.data)
    new.weight = bnb.nn.Params4bit(
        linear.weight.data,
        requires_grad=False,
    ).to(device)  # type: ignore[misc]
    return new


def quantize_model(
    model: nn.Module,
    config: Optional[QuantConfig] = None,
) -> Tuple[nn.Module, List[str]]:
    """Quantize eligible ``nn.Linear`` layers in ``model`` in-place.

    Returns the (possibly mutated) model and a list of replaced module names.
    On import or runtime errors with ``bitsandbytes`` backends the error is
    re-raised so callers can fall back to the pure-PyTorch ``int8`` path.
    """
    if config is None:
        config = QuantConfig()

    if config.method == "none":
        return model, []

    if config.layer_filter is not None:
        re.compile(config.layer_filter)  # validate early

    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not _should_quantize(name, module, config):
            continue
        parent, child_name = _get_parent(model, name)
        linear = module
        if config.method == "int8":
            new_module: nn.Module = QuantizedLinear.from_float(linear, config.compute_dtype)
        elif config.method == "bnb_8bit":
            new_module = _make_bnb_8bit(linear, config)
        elif config.method == "bnb_4bit":
            new_module = _make_bnb_4bit(linear, config)
        else:
            raise ValueError(f"Unsupported quantization method: {config.method}")
        setattr(parent, child_name, new_module)
        replaced.append(name)

    return model, replaced


def dequantize_model(model: nn.Module) -> nn.Module:
    """Undo ``quantize_model`` by restoring ``QuantizedLinear`` to ``nn.Linear``.

    ``bitsandbytes`` layers are left untouched because their dequantized weights
    are not directly exposed.
    """
    for name, module in list(model.named_modules()):
        if not isinstance(module, QuantizedLinear):
            continue
        parent, child_name = _get_parent(model, name)
        restored = nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
        )
        with torch.no_grad():
            restored.weight.data.copy_(module.weight)
            if module.bias is not None:
                assert restored.bias is not None
                restored.bias.data.copy_(module.bias.data)
        setattr(parent, child_name, restored)
    return model


def estimate_quantized_size(
    model: nn.Module,
    method: str = "int8",
    bitsandbytes_overhead: float = 1.05,
) -> float:
    """Estimate the compressed model size in bytes.

    Parameters
    ----------
    method:
        ``"int8"`` / ``"bnb_8bit"`` (1 byte/weight + fp16 scale per output
        channel) or ``"bnb_4bit"`` (0.5 byte/weight, approximate).
    bitsandbytes_overhead:
        Small multiplier for the BnB metadata.

    Returns
    -------
    Estimated size in bytes.
    """
    total_params = sum(p.numel() for p in model.parameters())
    if method in ("int8", "bnb_8bit"):
        bytes_per_weight = 1.0
        # per-output-channel fp16 scale
        scale_params = sum(
            m.out_features
            for m in model.modules()
            if isinstance(m, QuantizedLinear)
        )
        return total_params * bytes_per_weight + scale_params * 2
    if method == "bnb_4bit":
        return total_params * 0.5 * bitsandbytes_overhead
    return float("nan")


def compute_quantization_mse(
    model: nn.Module,
    config: Optional[QuantConfig] = None,
) -> float:
    """Compute the average per-element MSE introduced by quantizing weights.

    The model is quantized, each ``QuantizedLinear`` is compared against its
    original float weights, and the model is restored afterwards.
    """
    if config is None:
        config = QuantConfig(method="int8")

    original_sd = {k: v.clone() for k, v in model.state_dict().items()}
    _, replaced = quantize_model(model, config)
    mse_sum = 0.0
    n = 0
    for name in replaced:
        module = model.get_submodule(name)
        if not isinstance(module, QuantizedLinear):
            continue
        original = original_sd[name + ".weight"]
        diff = (module.weight - original).float()
        mse_sum += (diff * diff).sum().item()
        n += original.numel()

    dequantize_model(model)
    model.load_state_dict(original_sd, strict=True)
    if n == 0:
        return 0.0
    return mse_sum / n
