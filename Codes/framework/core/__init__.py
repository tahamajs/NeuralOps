# framework/core/__init__.py (final version)
"""
Core framework modules for SVD-Lin.

This package provides base classes, registries, utilities, and experiment
tracking for the SVD compression framework.
"""


from .base import (
    BaseAttention, 
    BaseVisualizer, 
    BaseCompressor,
    SVDCompressor,
    PruningCompressor,
    QuantizationCompressor,
    CompressorFactory,
    compress_model,
)

from .registry import (
    Registry, 
    MODEL_REGISTRY, 
    ATTENTION_REGISTRY, 
    VISUALIZER_REGISTRY,
    list_registered_components,
    print_registry_summary,
)
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
from .experiment_tracker import ExperimentTracker, create_tracker
from .attention_utils import (
    replace_attention_in_model,
    get_attention_module_names,
    count_attention_modules,
)

__all__ = [
    # Base classes
    "BaseCompressor",
    "SVDCompressor",
    "PruningCompressor",
    "QuantizationCompressor",
    "CompressorFactory",
    "compress_model",
    
    # Registry
    "Registry",
    "MODEL_REGISTRY",
    "ATTENTION_REGISTRY",
    "VISUALIZER_REGISTRY",
    "list_registered_components",
    "print_registry_summary",
    
    # Utilities
    "get_device",
    "set_seed",
    "get_model_device",
    "count_parameters",
    "format_number",
    "clear_memory",
    "get_memory_usage",
    "human_readable_time",
    
    # Tracking
    "ExperimentTracker",
    "create_tracker",
    
    # Attention utilities
    "replace_attention_in_model",
    "get_attention_module_names",
    "count_attention_modules",
]

__version__ = "2.0.0"
__author__ = "SVD-Lin Team"
__description__ = "Core framework for SVD-based transformer compression"