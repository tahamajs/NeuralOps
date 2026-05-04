# framework/core/__init__.py
"""Core utilities for the Efficient Transformer Toolkit."""

from .utils import (
    get_device,
    set_seed,
    get_model_device,
    count_parameters,
    format_number,
    clear_memory,
    get_memory_usage,
    human_readable_time,
)
from .registry import (
    Registry,
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    TRAINER_REGISTRY,
    VISUALIZER_REGISTRY,
    list_all_components,
)
from .base import BaseAttention, BaseCompressor, BaseVisualizer
from .model_wrapper import UnifiedTransformerModel
from .architecture import detect_architecture, get_attention_modules, replace_attention_module
from .experiment_tracker import ExperimentTracker, create_tracker
from .config import load_config
from .attention_utils import replace_attention_in_model, get_attention_module_names, count_attention_modules

__all__ = [
    # utils
    "get_device",
    "set_seed",
    "get_model_device",
    "count_parameters",
    "format_number",
    "clear_memory",
    "get_memory_usage",
    "human_readable_time",
    # registry
    "Registry",
    "ATTENTION_REGISTRY",
    "COMPRESSOR_REGISTRY",
    "TRAINER_REGISTRY",
    "VISUALIZER_REGISTRY",
    "list_all_components",
    # base classes
    "BaseAttention",
    "BaseCompressor",
    "BaseVisualizer",
    # model wrapper
    "UnifiedTransformerModel",
    # architecture
    "detect_architecture",
    "get_attention_modules",
    "replace_attention_module",
    # experiment tracker
    "ExperimentTracker",
    "create_tracker",
    # config
    "load_config",
    # attention utils
    "replace_attention_in_model",
    "get_attention_module_names",
    "count_attention_modules",
]