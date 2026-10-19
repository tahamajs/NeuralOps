"""
Central Component Registry Interface

Decorators and registry catalogs managing discovery and instantiation of attention and compression modules.

Author: Taha Majlesi
"""

from __future__ import annotations

import sys
from typing import Any, Callable

from neuralops.core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    METHOD_REGISTRY,
    AUX_REGISTRY,
    MODEL_REGISTRY,
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    TRAINER_REGISTRY,
    VISUALIZER_REGISTRY,
    Registry,
    create_attention,
    create_compressor,
    is_helper_module,
    get_evaluatable_modules,
    get_module_capabilities,
    is_module_declared_compatible,
    list_all_components,
    register_attention,
    register_compressor,
    list_attention_mechanisms,
    list_attentions,
    list_compressors,
)

__version__ = "1.4.0"

if "neuralops" in sys.modules:
    _top = sys.modules["neuralops"]
    setattr(_top, "list_attention_mechanisms", list_attention_mechanisms)
    setattr(_top, "register_attention", register_attention)
    setattr(_top, "register_compressor", register_compressor)

__all__ = [
    "ATTENTION_REGISTRY",
    "COMPRESSOR_REGISTRY",
    "METHOD_REGISTRY",
    "AUX_REGISTRY",
    "MODEL_REGISTRY",
    "DATASET_REGISTRY",
    "LOSS_REGISTRY",
    "TRAINER_REGISTRY",
    "VISUALIZER_REGISTRY",
    "Registry",
    "create_attention",
    "create_compressor",
    "is_helper_module",
    "get_evaluatable_modules",
    "get_module_capabilities",
    "is_module_declared_compatible",
    "list_all_components",
    "register_attention",
    "register_compressor",
    "list_attention_mechanisms",
    "list_attentions",
    "list_compressors",
]
