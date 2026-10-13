"""
NeuralOps Public Clean API.

Enables drop-in replacement and compression of attention layers in any
PyTorch or HuggingFace Transformer model with a single line of code:

    import neuralops

    # Drop-in replace attention mechanism
    model = neuralops.replace_attention(model, "cosformer")

    # Compress model using SVD or low-rank adapters
    model = neuralops.compress(model, method="adasvd", rank=32)

    # Inspect available modules
    modules = neuralops.list_attentions()
    compressors = neuralops.list_compressors()
"""
from __future__ import annotations

from typing import Any, List, Optional
import torch.nn as nn

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions  # noqa: F401 - ensure registrations
import neuralops.compression_engine  # noqa: F401 - ensure registrations
from neuralops.core.attention_utils import inject_attention
from neuralops.export.exporter import export_model, load_exported_model


def replace_attention(
    model: nn.Module,
    attention: Optional[str] = None,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    layers: Optional[List[int]] = None,
    attention_type: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    """Drop-in replace attention layers in any PyTorch / HuggingFace model.

    Args:
        model: HuggingFace or PyTorch Transformer model (GPT-2, LLaMA, Mistral, BERT, etc.)
        attention: Attention mechanism name from `neuralops.list_attentions()`
                   (e.g., "cosformer", "mamba_block", "gqa_attention", "svd_linear_attention")
        rank: Rank parameter for low-rank or SVD-based attention variants (default: None).
        copy_weights: Whether to transplant weights from the original model (default: True).
        layers: Optional list of layer indices to replace (default: None replaces all layers).
        **kwargs: Additional parameters forwarded to the attention module constructor.

    Returns:
        The model with attention layers replaced in-place.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> import neuralops
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> model = neuralops.replace_attention(model, "cosformer")
    """
    attn_name = attention or attention_type
    if not attn_name:
        raise ValueError("Must specify attention or attention_type.")
    if attn_name not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention module {attn_name}. "
            f"Use neuralops.list_attentions() to see all {len(ATTENTION_REGISTRY)} available modules."
        )

    if layers is not None:
        kwargs["layer_indices"] = layers

    return inject_attention(
        model=model,
        attention_type=attn_name,
        rank=rank,
        copy_weights=copy_weights,
        **kwargs,
    )


def compress(
    model: nn.Module,
    method: str = "svd",
    rank: int = 32,
    layers: Optional[List[int]] = None,
    **kwargs: Any,
) -> nn.Module:
    """Compress model weights using SVD, low-rank factorization, or pruning.

    Args:
        model: PyTorch or HuggingFace model to compress.
        method: Compression engine name from `neuralops.list_compressors()`.
        rank: Target rank for low-rank decomposition.
        layers: Optional list of layer indices to compress.
        **kwargs: Extra parameters passed to the compressor.

    Returns:
        The compressed model.
    """
    from neuralops.core.compression_pipeline import compress_model

    comp_step = {
        "method": method,
        "rank": rank,
        **kwargs,
    }
    if layers is not None:
        comp_step["layer_indices"] = layers

    config = {
        "compressors": [comp_step],
    }

    return compress_model(model=model, config=config)


def list_attentions() -> List[str]:
    """Return a sorted list of all available canonical attention module names."""
    return sorted(list(ATTENTION_REGISTRY.keys()))


def list_compressors() -> List[str]:
    """Return a sorted list of all available canonical compressor names."""
    return sorted(list(COMPRESSOR_REGISTRY.keys()))


def get_attention_class(name: str) -> type:
    """Retrieve the class for a registered attention module."""
    if name not in ATTENTION_REGISTRY:
        raise KeyError(f"Unknown attention {name}. Available: {list_attentions()}")
    return ATTENTION_REGISTRY[name]


def get_compressor_class(name: str) -> type:
    """Retrieve the class for a registered compressor engine."""
    if name not in COMPRESSOR_REGISTRY:
        raise KeyError(f"Unknown compressor {name}. Available: {list_compressors()}")
    return COMPRESSOR_REGISTRY[name]
