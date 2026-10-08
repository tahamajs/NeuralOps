"""Model loading, metadata inspection, and attention/compression surgery for benchmarks."""
from __future__ import annotations

import gc
import logging
from typing import Any
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from neuralops.benchmarks.result import ModelMetadata
from neuralops.core.attention_utils import inject_attention
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

LOG = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable and non-trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def cleanup_model(model: nn.Module | None) -> None:
    """Safely delete model and reclaim CPU/GPU VRAM."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_model_metadata(model: nn.Module, config: Any) -> ModelMetadata:
    """Extract standard ModelMetadata attributes from model instance and config."""
    name = getattr(config, "_name_or_path", getattr(model, "name", "unknown_model"))
    model_type = getattr(config, "model_type", "unknown")
    num_params = count_parameters(model)
    hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 0))
    num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 0))
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layers", 0))

    return ModelMetadata(
        name=name,
        family=model_type,
        num_params=num_params,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
    )


def load_model_and_tokenizer(
    model_name: str,
    device: str | torch.device = "cpu",
    dtype: str = "float32",
    num_labels: int | None = None,
    is_causal: bool = False,
    trust_remote_code: bool = True,
) -> tuple[nn.Module, Any, Any]:
    """
    Load HuggingFace model, tokenizer, and config with transparent device fallback.

    Returns:
        Tuple of (model, tokenizer, config)
    """
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype, torch.float32)

    LOG.info("Loading tokenizer for '%s'...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "[PAD]"

    LOG.info("Loading configuration for '%s'...", model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if num_labels is not None:
        config.num_labels = num_labels

    LOG.info("Instantiating model '%s' (is_causal=%s, dtype=%s)...", model_name, is_causal, dtype)
    try:
        if is_causal:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        elif num_labels is not None:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = AutoModel.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        model = model.to(device)
    except torch.cuda.OutOfMemoryError as e:
        LOG.warning("CUDA OOM loading model to %s. Falling back to CPU: %s", device, e)
        cleanup_model(None)
        if is_causal:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=trust_remote_code,
            )
        elif num_labels is not None:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=trust_remote_code,
            )
        else:
            model = AutoModel.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=trust_remote_code,
            )
        model = model.to("cpu")

    model.eval()
    return model, tokenizer, config


def apply_attention_module(
    model: nn.Module,
    module_name: str,
    rank: int = 64,
    copy_weights: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Inject attention module using universal attention surgery."""
    if module_name in ("full", "dense", "baseline"):
        LOG.info("Preserving original baseline attention structure.")
        return model

    if module_name not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Module '{module_name}' is not in ATTENTION_REGISTRY ({len(ATTENTION_REGISTRY)} modules available)."
        )

    LOG.info("Injecting attention replacement: '%s' (rank=%d)...", module_name, rank)
    return inject_attention(
        model,
        attention_type=module_name,
        rank=rank,
        copy_weights=copy_weights,
        **kwargs,
    )


def apply_compressor(
    model: nn.Module,
    module_name: str,
    rank: int = 64,
    calibration_loader: Any = None,
    **kwargs: Any,
) -> nn.Module:
    """Apply compression engine to model."""
    if module_name in ("full", "dense", "baseline"):
        return model

    if module_name not in COMPRESSOR_REGISTRY:
        raise ValueError(
            f"Compressor '{module_name}' is not in COMPRESSOR_REGISTRY ({len(COMPRESSOR_REGISTRY)} available)."
        )

    LOG.info("Applying compressor: '%s' (rank=%d)...", module_name, rank)
    compressor_cls = COMPRESSOR_REGISTRY[module_name]
    try:
        compressor = compressor_cls(rank=rank, **kwargs)
    except TypeError:
        compressor = compressor_cls(**kwargs)

    if hasattr(compressor, "compress"):
        if calibration_loader is not None and "dataloader" in compressor.compress.__code__.co_varnames:
            return compressor.compress(model, dataloader=calibration_loader)
        return compressor.compress(model)
    return model


# Backward compatibility aliases
inject_attention_module = apply_attention_module
apply_compression = apply_compressor

__all__ = [
    "count_parameters",
    "cleanup_model",
    "get_model_metadata",
    "load_model_and_tokenizer",
    "apply_attention_module",
    "inject_attention_module",
    "apply_compressor",
    "apply_compression",
]

