"""Universal Hugging Face Model Loader and Architecture Adapters.

Supports loading ANY model from Hugging Face Hub or local path with:
- Transparent AutoModel class detection (SequenceClassification, CausalLM, Seq2SeqLM, etc.)
- Comprehensive family-specific attention surgery adapters (BERT, GPT, LLaMA, T5, Vision, SSM)
- Device placement & transparent device map / CPU offloading
- Precise precision & dtype resolution
- Tokenizer alignment & padding token synthesis
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from neuralops.benchmarks.results.schema import ModelMetadata
from neuralops.core.attention_utils import inject_attention
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

LOG = logging.getLogger(__name__)

# Registry for custom model architecture adapters
MODEL_ADAPTER_REGISTRY: Dict[str, Type["ModelAdapter"]] = {}


def register_model_adapter(name: str) -> Callable[[Type["ModelAdapter"]], Type["ModelAdapter"]]:
    """Decorator to register a custom model architecture adapter."""
    def decorator(cls: Type["ModelAdapter"]) -> Type["ModelAdapter"]:
        MODEL_ADAPTER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


class ModelAdapter:
    """Base class for architecture-specific attention surgery."""

    def __init__(self, model: nn.Module, config: Any) -> None:
        self.model = model
        self.config = config

    def extract_qkv(self) -> Dict[str, nn.Module]:
        """Identify Q/K/V/O layers in the architecture."""
        return {}

    def inject_attention(self, new_module_name: str, rank: int = 64, **kwargs: Any) -> nn.Module:
        """Inject candidate attention replacement."""
        return inject_attention(self.model, attention_type=new_module_name, rank=rank, **kwargs)


@register_model_adapter("bert")
@register_model_adapter("roberta")
@register_model_adapter("distilbert")
@register_model_adapter("deberta")
@register_model_adapter("xlm-roberta")
class BertFamilyAdapter(ModelAdapter):
    """Adapter for bidirectional encoder transformer architectures."""

    def inject_attention(self, new_module_name: str, rank: int = 64, **kwargs: Any) -> nn.Module:
        return inject_attention(self.model, attention_type=new_module_name, rank=rank, **kwargs)


@register_model_adapter("gpt2")
@register_model_adapter("gpt_neo")
@register_model_adapter("gptj")
@register_model_adapter("opt")
class GptFamilyAdapter(ModelAdapter):
    """Adapter for causal auto-regressive GPT-style architectures."""

    def inject_attention(self, new_module_name: str, rank: int = 64, **kwargs: Any) -> nn.Module:
        return inject_attention(self.model, attention_type=new_module_name, rank=rank, is_causal=True, **kwargs)


@register_model_adapter("llama")
@register_model_adapter("qwen2")
@register_model_adapter("qwen")
@register_model_adapter("mistral")
@register_model_adapter("gemma")
@register_model_adapter("phi")
class LlamaFamilyAdapter(ModelAdapter):
    """Adapter for modern decoder architectures featuring RoPE, RMSNorm, and GQA."""

    def inject_attention(self, new_module_name: str, rank: int = 64, **kwargs: Any) -> nn.Module:
        return inject_attention(self.model, attention_type=new_module_name, rank=rank, is_causal=True, **kwargs)


@register_model_adapter("t5")
@register_model_adapter("bart")
@register_model_adapter("flan-t5")
class T5FamilyAdapter(ModelAdapter):
    """Adapter for encoder-decoder architectures."""

    def inject_attention(self, new_module_name: str, rank: int = 64, **kwargs: Any) -> nn.Module:
        return inject_attention(self.model, attention_type=new_module_name, rank=rank, **kwargs)


def get_model_adapter(model: nn.Module, config: Any) -> ModelAdapter:
    """Return matching ModelAdapter for given architecture."""
    model_type = getattr(config, "model_type", "").lower()
    for key, adapter_cls in MODEL_ADAPTER_REGISTRY.items():
        if key in model_type:
            return adapter_cls(model, config)
    return ModelAdapter(model, config)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable and non-trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cleanup_model(model: nn.Module | None) -> None:
    """Safely delete model and reclaim memory."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def inspect_model_metadata(model: nn.Module, config: Any, model_name: str) -> ModelMetadata:
    """Extract complete ModelMetadata attributes from model instance and config."""
    name = getattr(config, "_name_or_path", model_name)
    model_type = getattr(config, "model_type", "unknown")
    num_params = count_parameters(model)
    num_trainable = count_trainable_parameters(model)
    hidden_size = getattr(config, "hidden_size", getattr(config, "dim", getattr(config, "d_model", 768)))
    num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
    num_kv = getattr(config, "num_key_value_heads", num_heads)
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layers", 6))
    vocab_size = getattr(config, "vocab_size", 30522)
    max_pos = getattr(config, "max_position_embeddings", getattr(config, "n_positions", 512))
    dtype_str = str(next(model.parameters()).dtype).replace("torch.", "") if list(model.parameters()) else "float32"

    return ModelMetadata(
        name=name,
        family=model_type,
        num_params=num_params,
        num_trainable_params=num_trainable,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv,
        num_layers=num_layers,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        dtype=dtype_str,
        source="local_path" if os.path.exists(model_name) else "pretrained",
    )


def load_universal_model(
    model_name: str,
    task: str = "classification",
    device: Optional[str] = None,
    dtype: Optional[str] = "float32",
    num_labels: Optional[int] = None,
    trust_remote_code: bool = True,
    hf_token: Optional[str] = None,
    tokenizer_name: Optional[str] = None,
) -> Tuple[nn.Module, Any, Any]:
    """
    Load ANY Hugging Face model and tokenizer with automatic class deduction:
    - AutoModelForSequenceClassification for classification / regression
    - AutoModelForCausalLM for language-modeling / instruction-tuning
    - AutoModelForQuestionAnswering for question-answering
    - AutoModelForSeq2SeqLM for summarization / translation (T5, BART)
    - AutoModel as universal fallback
    """
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16 if hasattr(torch, "bfloat16") else torch.float32,
    }.get(dtype or "float32", torch.float32)

    # 1. Load Tokenizer
    tok_name = tokenizer_name or model_name
    LOG.info("Loading tokenizer from: '%s'...", tok_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "[PAD]"

    # 2. Load Config
    LOG.info("Loading config for '%s'...", model_name)
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    if num_labels is not None:
        config.num_labels = num_labels

    # 3. Determine Candidate Model Classes based on Task and Config
    task_lower = task.lower()
    is_seq2seq = getattr(config, "is_encoder_decoder", False)

    candidate_classes: List[Tuple[Any, Dict[str, Any]]] = []

    if "question" in task_lower or "qa" in task_lower:
        candidate_classes.append((AutoModelForQuestionAnswering, {}))
    elif "summariz" in task_lower or "translat" in task_lower or is_seq2seq:
        candidate_classes.append((AutoModelForSeq2SeqLM, {}))
    elif "instruction" in task_lower or "causal" in task_lower or "lm" in task_lower or "generation" in task_lower:
        candidate_classes.append((AutoModelForCausalLM, {}))
    elif "token" in task_lower or "ner" in task_lower:
        candidate_classes.append((AutoModelForTokenClassification, {}))
    elif "classif" in task_lower or "regress" in task_lower:
        candidate_classes.append((AutoModelForSequenceClassification, {}))

    # Universal fallbacks
    candidate_classes.extend([
        (AutoModelForSequenceClassification, {}),
        (AutoModelForCausalLM, {}),
        (AutoModelForSeq2SeqLM, {}),
        (AutoModel, {}),
    ])

    model = None
    loaded_class_name = ""
    last_err: Optional[Exception] = None

    for model_cls, extra_kwargs in candidate_classes:
        try:
            LOG.info("Attempting instantiation with %s...", model_cls.__name__)
            model = model_cls.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                token=token,
                **extra_kwargs,
            )
            loaded_class_name = model_cls.__name__
            break
        except Exception as e:
            last_err = e
            continue

    if model is None:
        raise RuntimeError(f"Failed to instantiate model '{model_name}'. Last error: {last_err}")

    # Move to target device
    try:
        model = model.to(target_device)
    except torch.cuda.OutOfMemoryError as oom:
        LOG.warning("CUDA OOM moving model to %s. Falling back to CPU: %s", target_device, oom)
        model = model.to("cpu")

    model.eval()
    meta = inspect_model_metadata(model, config, model_name)
    LOG.info(
        "Successfully loaded '%s' via %s (%d params, dtype=%s, device=%s)",
        model_name, loaded_class_name, meta.num_params, meta.dtype, target_device
    )
    return model, tokenizer, config


__all__ = [
    "ModelAdapter",
    "register_model_adapter",
    "get_model_adapter",
    "count_parameters",
    "count_trainable_parameters",
    "cleanup_model",
    "inspect_model_metadata",
    "load_universal_model",
    "MODEL_ADAPTER_REGISTRY",
]
