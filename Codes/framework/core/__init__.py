"""
Efficient Transformer Toolkit – Core module.
Provides model‑agnostic architecture detection, attention injection,
compression pipeline, and base classes.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Map symbol names to their module filename within framework.core
_MODULE_MAP: dict[str, str] = {
    "ADAPTER_MAP": "adapters",
    "get_adapter": "adapters",
    "detect_architecture": "architecture",
    "get_attention_modules": "architecture",
    "get_classifier_head": "architecture",
    "get_embedding_module": "architecture",
    "get_ffn_modules": "architecture",
    "get_transformer_layers": "architecture",
    "apply_attention_mask": "attention_mask",
    "get_token_valid_mask": "attention_mask",
    "create_causal_mask": "attention_mask",
    "create_band_mask": "attention_mask",
    "available_attention_types": "attention_utils",
    "inject_attention": "attention_utils",
    "softpick": "attention_utils",
    "BaseAttention": "base",
    "BaseCompressor": "base",
    "CompressionPipeline": "compression_engine",
    "compress_model": "compression_engine",
    "GPUMemoryGuard": "memory",
    "cleanup_gpu_memory": "memory",
    "get_gpu_memory_stats": "memory",
    "activation_hooks": "model_utils",
    "copy_attention_weights": "model_utils",
    "count_params": "model_utils",
    "get_parent_and_child_name": "model_utils",
    "replace_module": "model_utils",
    "run_calibration_forward": "model_utils",
    "UnifiedTransformerConfig": "model_wrapper",
    "UnifiedTransformerModel": "model_wrapper",
    "ATTENTION_REGISTRY": "registry",
    "AUX_REGISTRY": "registry",
    "COMPRESSOR_REGISTRY": "registry",
    "DATASET_REGISTRY": "registry",
    "LOSS_REGISTRY": "registry",
    "METHOD_REGISTRY": "registry",
    "MODEL_REGISTRY": "registry",
    "TRAINER_REGISTRY": "registry",
    "VISUALIZER_REGISTRY": "registry",
    "list_all_components": "registry",
    "MultiModeBenchmarkRunner": "multi_mode_benchmark",
    "SUPPORTED_MODES": "multi_mode_benchmark",
    "format_number": "utils",
    "get_device": "utils",
    "selective_scan": "selective_scan",
    "selective_scan_chunked": "selective_scan",
    "get_calibration_tensors": "pruning_utils",
    "human_readable_time": "utils",
    "set_seed": "utils",
    "_cfg_get": "common",
    "repeat_kv": "common",
    "_detect_decoder": "common",
    "_apply_rope": "common",
    "LoRALinear": "common",
    "FactorizedLinear": "common",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loading for framework.core symbols."""
    if name in _MODULE_MAP:
        mod_name = _MODULE_MAP[name]
        try:
            module = importlib.import_module(f"framework.core.{mod_name}")
            val = getattr(module, name)
            globals()[name] = val
            return val
        except Exception as e:
            logger.error("Failed to lazily load %s from framework.core.%s: %s", name, mod_name, e)
            raise
    # Submodule lookup fallback
    try:
        module = importlib.import_module(f"framework.core.{name}")
        globals()[name] = module
        return module
    except ModuleNotFoundError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_MODULE_MAP.keys())

