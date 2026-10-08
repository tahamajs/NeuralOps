"""Model loading and compression injection utilities for benchmarks."""
from __future__ import annotations

import logging
from typing import Any
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from neuralops.core.attention_utils import inject_attention
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

LOG = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name_or_path: str,
    num_labels: int | None = None,
    device: str | torch.device = "cpu",
    is_causal: bool = False,
    trust_remote_code: bool = True,
) -> tuple[nn.Module, Any]:
    """
    Robustly load HuggingFace model and associated tokenizer.

    Args:
        model_name_or_path: Model repo ID or local checkpoint path.
        num_labels: Classification labels count (for sequence classification).
        device: Target execution device.
        is_causal: Whether to instantiate a causal language model.
        trust_remote_code: HuggingFace remote code execution flag.

    Returns:
        Tuple of (model, tokenizer).
    """
    LOG.info("Loading tokenizer for %s...", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "[PAD]"

    LOG.info("Loading model for %s...", model_name_or_path)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if num_labels is not None:
        config.num_labels = num_labels

    if is_causal:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=trust_remote_code,
        )
    elif num_labels is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModel.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=trust_remote_code,
        )

    model = model.to(device)
    model.eval()
    return model, tokenizer


def apply_compression(
    model: nn.Module,
    module_name: str,
    rank: int = 64,
    copy_weights: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Apply an attention replacement or compression engine algorithm to a transformer model.

    Args:
        model: PyTorch model instance.
        module_name: Target module key registered in ATTENTION_REGISTRY or COMPRESSOR_REGISTRY.
        rank: Target bottleneck rank.
        copy_weights: SVD weight projection copying enabled.
        kwargs: Module-specific hyperparameter arguments.

    Returns:
        Modified model with injected architecture or compressed weights.
    """
    if module_name in ("full", "dense", "baseline"):
        LOG.info("Preserving uncompressed baseline architecture.")
        return model

    if module_name in ATTENTION_REGISTRY:
        LOG.info("Injecting attention replacement: '%s' (rank=%d)...", module_name, rank)
        return inject_attention(
            model,
            attention_type=module_name,
            rank=rank,
            copy_weights=copy_weights,
            **kwargs,
        )

    if module_name in COMPRESSOR_REGISTRY:
        LOG.info("Applying compression engine: '%s' (rank=%d)...", module_name, rank)
        compressor_cls = COMPRESSOR_REGISTRY[module_name]
        compressor = compressor_cls(rank=rank, **kwargs)
        return compressor.compress(model)

    raise ValueError(
        f"Unknown module '{module_name}'. Must be in ATTENTION_REGISTRY "
        f"({len(ATTENTION_REGISTRY)} keys) or COMPRESSOR_REGISTRY ({len(COMPRESSOR_REGISTRY)} keys)."
    )


def inject_attention_module(
    model: nn.Module,
    module_name: str,
    rank: int = 64,
    copy_weights: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Direct alias to apply_compression for attention surgery."""
    return apply_compression(
        model,
        module_name=module_name,
        rank=rank,
        copy_weights=copy_weights,
        **kwargs,
    )


__all__ = [
    "load_model_and_tokenizer",
    "apply_compression",
    "inject_attention_module",
]
