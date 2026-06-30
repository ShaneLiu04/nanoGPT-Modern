"""nanoGPT-Modern model zoo.

Exports the two architecture variants (BaselineGPT / ModernGPT), their
configurations, KV-cache utilities, and the attention backend helpers so that
callers can simply do::

    from model import ModernGPT, ModernGPTConfig, KVCacheManager
"""

from .attention_utils import (
    print_attention_backend,
    probe_gqa_sdpa_support,
    reset_gqa_probe_cache,
    set_attention_backend,
)
from .baseline_gpt import BaselineGPT, BaselineGPTConfig
from .kv_cache_utils import KVCacheManager, build_sliding_window_mask
from .modern_gpt import ModernGPT, ModernGPTConfig
from .quantization import QuantConfig, QuantizedLinear, quantize_model
from .paged_kv_cache import PagedKVCacheManager

__all__ = [
    "BaselineGPT",
    "BaselineGPTConfig",
    "ModernGPT",
    "ModernGPTConfig",
    "KVCacheManager",
    "PagedKVCacheManager",
    "build_sliding_window_mask",
    "set_attention_backend",
    "print_attention_backend",
    "probe_gqa_sdpa_support",
    "reset_gqa_probe_cache",
    "QuantConfig",
    "QuantizedLinear",
    "quantize_model",
]
