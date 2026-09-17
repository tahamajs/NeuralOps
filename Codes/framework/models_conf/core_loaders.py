"""
Unified model loader – works with any HuggingFace model, optionally injects
custom attention and wraps the result in a UnifiedTransformerModel.
"""
from typing import Optional

from framework.core.attention_utils import inject_attention
from framework.core.model_wrapper import UnifiedTransformerModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def _safe_load(loader_fn, *args, **kwargs):
    """Load model/config/tokenizer using local disk cache first to eliminate re-downloads."""
    try:
        return loader_fn(*args, local_files_only=True, **kwargs)
    except Exception:
        # Download once online if missing from local cache
        return loader_fn(*args, **kwargs)


def load_hf_model(
    model_name: str,
    num_labels: int = 2,
    cache_dir: str = None,
    **kwargs,
) -> tuple:
    """
    Load a standard HuggingFace classification model.

    Returns:
        model, tokenizer
    """
    config = _safe_load(
        AutoConfig.from_pretrained,
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
    )
    tokenizer = _safe_load(
        AutoTokenizer.from_pretrained, model_name, cache_dir=cache_dir
    )
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in {"task", "attention_type", "attention_rank", "copy_weights", "rank", "loader_type"}
    }
    model = _safe_load(
        AutoModelForSequenceClassification.from_pretrained,
        model_name,
        config=config,
        cache_dir=cache_dir,
        **model_kwargs,
    )
    return model, tokenizer


def load_compressed_model(
    model_name: str,
    num_labels: int = 2,
    rank: int = 64,
    cache_dir: str = None,
    **kwargs,
) -> tuple:
    """
    Load a CompressedBertModel (SVD on embeddings + simplified encoder).
    Only works for BERT‑like architectures.
    """
    from .custom_models import CompressedBertModel

    AutoConfig.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = CompressedBertModel(model_path=model_name, rank=rank, num_labels=num_labels)
    return model, tokenizer


def load_hybrid_svd_model(
    model_name: str,
    num_labels: int = 2,
    rank: int = 64,
    cache_dir: str = None,
    **kwargs,
) -> tuple:
    """
    Load a HybridBertForSequenceClassification (uses DirectSVDEncoderLayer).
    Only works for BERT‑like architectures.
    """
    from .custom_models import HybridBertForSequenceClassification

    config = AutoConfig.from_pretrained(
        model_name, num_labels=num_labels, cache_dir=cache_dir
    )
    config.svd_rank = rank
    model = HybridBertForSequenceClassification(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_unsloth_model(
    model_name: str,
    num_labels: int = 2,
    max_seq_length: int = 2048,
    **kwargs,
) -> tuple:
    """Load a model using Unsloth (requires unsloth)."""
    from .unsloth_loader import load_unsloth_model_with_attention

    return load_unsloth_model_with_attention(
        model_name=model_name,
        num_labels=num_labels,
        attention_type="full",  # default no replacement
        rank=64,
        max_seq_length=max_seq_length,
        **kwargs,
    )


# ----------------------------------------------------------------------
# Main dispatch function
# ----------------------------------------------------------------------
_VARIANT_LOADERS = {
    "standard": load_hf_model,
    "compressed": load_compressed_model,  # BERT only
    "hybrid_svd": load_hybrid_svd_model,  # BERT only
    "unsloth": load_unsloth_model,
}


def load_model(
    model_name: str,
    num_labels: int = 2,
    variant: str = "standard",
    attention_type: str = None,
    attention_rank: int = 32,
    wrap_unified: bool = True,
    cache_dir: str = None,
    **kwargs,
) -> tuple:
    """
    Main entry point: load any HuggingFace model + variant, optionally inject
    custom attention, and wrap in UnifiedTransformerModel.

    Args:
        model_name: HuggingFace model identifier or local path.
        num_labels: Number of output classes (for classification).
        variant: "standard", "compressed", "hybrid_svd", or "unsloth".
        attention_type: Optional attention registry key to inject.
        attention_rank: Compression rank for the attention module.
        wrap_unified: If True, wrap the model in UnifiedTransformerModel.
        cache_dir: HuggingFace cache directory.
        **kwargs: Additional arguments passed to the loader.

    Returns:
        (model, tokenizer) where model is a PyTorch module.
    """
    # 1. Load base model
    loader_type = kwargs.pop("loader_type", None)
    if variant == "standard" and loader_type and loader_type in _VARIANT_LOADERS:
        variant = loader_type
    loader = _VARIANT_LOADERS.get(variant, load_hf_model)
    model, tokenizer = loader(
        model_name,
        num_labels=num_labels,
        cache_dir=cache_dir,
        **kwargs,
    )

    # 2. Wrap with UnifiedTransformerModel (provides attention injection, Hub, etc.)
    if wrap_unified:
        model = UnifiedTransformerModel(model)

    # 3. Optionally inject a custom attention variant
    if attention_type is not None and attention_type != "full":
        if wrap_unified:
            model.replace_attention(attention_type, rank=attention_rank)
        else:
            # Fallback to direct injection if not wrapped
            inject_attention(
                model, attention_type, rank=attention_rank, copy_weights=True
            )

    return model, tokenizer
