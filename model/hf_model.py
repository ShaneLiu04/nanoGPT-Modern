"""HuggingFace Transformers compatibility wrapper for ModernGPT.

This module provides a thin ``transformers.PreTrainedModel`` wrapper around
``ModernGPT`` so that checkpoints can be exported to / loaded from the
HuggingFace Hub format (``config.json`` + ``model.safetensors``) without
rewriting the underlying model.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .modern_gpt import ModernGPT, ModernGPTConfig


class NanoGPTModernConfig(PretrainedConfig):
    """HF-style config that mirrors ``ModernGPTConfig``.

    The wrapper intentionally keeps the same hyper-parameter names as
    ``ModernGPTConfig`` so that round-tripping is obvious and mechanical.
    """

    model_type = "nanogpt-modern"

    def __init__(
        self,
        vocab_size: int = 50257,
        block_size: int = 1024,
        n_layer: int = 12,
        n_head: int = 8,
        n_embd: int = 512,
        n_kv_head: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.0,
        norm_position: str = "pre",
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.norm_position = norm_position
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

    @classmethod
    def from_nanogpt_config(cls, config: ModernGPTConfig) -> "NanoGPTModernConfig":
        """Build an HF config from a ``ModernGPTConfig``."""
        d = config.to_dict()
        # ``n_rep`` is a computed property, not a constructor argument.
        d.pop("n_rep", None)
        # ``block_size`` in ModernGPT is ``max_position_embeddings`` in HF terms.
        return cls(**d)

    def to_nanogpt_config(self) -> ModernGPTConfig:
        """Recover the native ``ModernGPTConfig`` from this HF config."""
        allowed = set(ModernGPTConfig.__init__.__code__.co_varnames)
        allowed.discard("self")
        d = {k: v for k, v in self.to_dict().items() if k in allowed}
        return ModernGPTConfig(**d)


class NanoGPTModernForCausalLM(PreTrainedModel):
    """HuggingFace-compatible causal LM backed by ``ModernGPT``."""

    config_class = NanoGPTModernConfig  # type: ignore[assignment]
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: NanoGPTModernConfig):
        super().__init__(config)
        self.model = ModernGPT(config.to_nanogpt_config())
        self.post_init()
        # Inform HF that ``lm_head`` shares storage with ``transformer.wte``.
        self._tied_weights_keys = ["model.lm_head.weight"]  # type: ignore[assignment]

    def get_input_embeddings(self) -> Any:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: Any) -> None:
        self.model.transformer.wte = value

    def get_output_embeddings(self) -> Any:
        return self.model.lm_head

    def set_output_embeddings(self, value: Any) -> None:
        self.model.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithCrossAttentions:
        """Forward compatible with ``transformers`` training / evaluation."""
        logits, loss, _ = self.model(
            input_ids,
            targets=labels,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generation delegates to the native ``ModernGPT.generate``."""
        return self.model.generate(input_ids, **kwargs)

    def _init_weights(self, module: Any) -> None:
        """Keep the native ModernGPT initialization."""
        if hasattr(self.model, "_init_weights"):
            self.model._init_weights(module)
