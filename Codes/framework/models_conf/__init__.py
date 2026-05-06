"""
Complete model loading system – supports any HuggingFace model and variant.
"""

from .loaders import MODEL_LOADER_REGISTRY, load_model
from .attention import replace_attention_in_model
from .catalog import search_models, get_model_info, POPULAR_MODELS

__all__ = [
    "MODEL_LOADER_REGISTRY",
    "load_model",
    "replace_attention_in_model",
    "search_models",
    "get_model_info",
    "POPULAR_MODELS",
]