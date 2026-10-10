"""Pydantic validation models for nanoGPT-Modern configurations.

Provides structured, runtime-validated config schemas that mirror the
existing ``BaselineGPTConfig`` and ``ModernGPTConfig`` classes.  These
models are **optional** — the existing YAML + argparse pipeline remains
unchanged; validation is only applied when explicitly requested via
``validate_with_pydantic`` in :mod:`utils.config`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class ModelConfig(BaseModel):
    """Shared architecture hyperparameters.

    Parameters
    ----------
    vocab_size : int
    block_size : int
        Maximum sequence length (must be > 0).
    n_layer : int
    n_head : int
    n_embd : int
    dropout : float
    """

    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True

    @field_validator("block_size", "n_embd", "n_head", "n_layer", "vocab_size")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be > 0, got {v}")
        return v


class BaselineGPTModelConfig(ModelConfig):
    """BaselineGPT-specific model config."""

    attention_backend: Literal["sdpa", "manual"] = "sdpa"
    norm_position: Literal["pre", "post"] = "pre"
    gradient_checkpointing: bool = False


class ModernGPTModelConfig(ModelConfig):
    """ModernGPT-specific model config with GQA, RoPE, SwiGLU, MoE.

    Validates the GQA invariant ``n_head % n_kv_head == 0``.
    """

    n_kv_head: int | None = None
    intermediate_size: int | None = None
    norm_position: Literal["pre", "post"] = "pre"
    num_experts: int = 1
    position_encoding: Literal["rope", "learned"] = "rope"
    ffn_variant: Literal["swiglu", "gelu"] = "swiglu"
    gradient_checkpointing: bool = False
    qk_norm: bool = False
    attn_temperature: float = 1.0
    rmsnorm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None
    moe_aux_loss_factor: float = 0.01
    moe_capacity_factor: float = 1.25
    use_flash_attn: bool = False
    use_ring_attention: bool = False
    ring_block_size_q: int = 64
    ring_block_size_kv: int = 64
    n_future: int = 0
    mtp_weight: float = 1.0
    sliding_window_size: int | None = None
    use_paged_kv_cache: bool = False
    kv_cache_block_size: int = 16
    gqa_broadcast: Literal["auto", "raw", "grouped", "repeat"] = "auto"

    @field_validator("n_kv_head")
    @classmethod
    def _default_kv_head(cls, v: int | None, info) -> int:
        """Default to n_head when not provided."""
        if v is None:
            n_head = info.data.get("n_head", 8)
            return n_head
        return v

    @model_validator(mode="after")
    def _check_gqa(self) -> ModernGPTModelConfig:
        """Ensure n_head is divisible by n_kv_head.

        ``n_kv_head=None`` (the default; field validators do not run on
        defaults in pydantic v2) falls back to ``n_head`` here so that
        ``ModernGPTModelConfig()`` is constructible.
        """
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
            )
        if self.n_kv_head > self.n_head:
            raise ValueError(f"n_kv_head ({self.n_kv_head}) cannot exceed n_head ({self.n_head})")
        return self

    @model_validator(mode="after")
    def _check_ablation_combos(self) -> ModernGPTModelConfig:
        """Reject invalid ablation-switch combinations with readable messages."""
        if self.position_encoding == "learned" and self.rope_scaling is not None:
            raise ValueError(
                "learned 绝对位置编码不支持 NTK 缩放，请去掉 rope_scaling "
                "(position_encoding='learned' is incompatible with rope_scaling)"
            )
        if self.ffn_variant == "gelu" and self.num_experts > 1:
            raise ValueError(
                "ffn_variant='gelu' 不支持 MoE，请使用 ffn_variant='swiglu' " "或 num_experts=1"
            )
        return self

    @model_validator(mode="after")
    def _check_intermediate_size(self) -> ModernGPTModelConfig:
        """Auto-compute intermediate_size when None, aligned to multiple_of."""
        if self.intermediate_size is None:
            if self.ffn_variant == "gelu":
                self.intermediate_size = 4 * self.n_embd
            else:
                raw = int(8 / 3 * self.n_embd)
                multiple_of = 128
                self.intermediate_size = ((raw + multiple_of - 1) // multiple_of) * multiple_of
        return self


class TrainingConfig(BaseModel):
    """Training hyperparameters shared across pretrain / SFT / GRPO / DPO.

    Parameters
    ----------
    batch_size : int
        Must be > 0.
    learning_rate : float
    min_lr : float
    max_iters : int
    warmup_iters : int
    lr_decay_iters : int
    weight_decay : float
    grad_clip : float
    """

    batch_size: int = 12
    learning_rate: float = 6.0e-4
    min_lr: float = 6.0e-5
    max_iters: int = 18000
    warmup_iters: int = 2000
    lr_decay_iters: int = 18000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 1000
    eval_iters: int = 200
    log_interval: int = 10
    seed: int = 1337
    device: str = "cuda"
    compile: bool = False
    use_wandb: bool = False

    @field_validator("batch_size", "max_iters", "eval_iters", "eval_interval")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"must be > 0, got {v}")
        return v


class DataConfig(BaseModel):
    """Data pipeline configuration."""

    data_dir: str = "data/openwebtext"
    num_workers: int = 4
    shuffle_buffer: int | None = None
    use_packing: bool = False


class FullConfig(BaseModel):
    """Top-level configuration that merges model, training, and data settings.

    Used when validating a complete YAML config file before passing it to
    the argparse pipeline.
    """

    model_type: Literal["baseline", "modern"] = "modern"
    out_dir: str = "out"
    model: BaselineGPTModelConfig | ModernGPTModelConfig = ModernGPTModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()

    @model_validator(mode="after")
    def _model_type_matches(self) -> FullConfig:
        """Ensure model type matches the instantiated model config."""
        if self.model_type == "baseline" and not isinstance(self.model, BaselineGPTModelConfig):
            raise ValueError("model_type='baseline' but model config is not BaselineGPTModelConfig")
        if self.model_type == "modern" and not isinstance(self.model, ModernGPTModelConfig):
            raise ValueError("model_type='modern' but model config is not ModernGPTModelConfig")
        return self
