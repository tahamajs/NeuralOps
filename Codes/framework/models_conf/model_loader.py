"""
Universal model loader – works with ANY HuggingFace model, plus variants.
"""
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

def load_hf_model(model_name: str, num_labels: int = 2, cache_dir: str = None):
    """Load a standard HuggingFace model and tokenizer."""
    if num_labels is None:
        num_labels = 2
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config, cache_dir=cache_dir
    )
    return model, tokenizer

def load_compressed_model(model_name: str, num_labels: int, rank: int = 64):
    """Compressed variant using SVD on embeddings (currently BERT‑based only)."""
    from .compressed_bert import CompressedBertModel
    model = CompressedBertModel(model_path=model_name, rank=rank, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_hybrid_svd_model(model_name: str, num_labels: int, rank: int = 64):
    """Hybrid SVD variant (BERT‑based only for now)."""
    from .hybrid_bert_svd import HybridBertForSequenceClassification
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    config.svd_rank = rank
    model = HybridBertForSequenceClassification(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_unsloth_model(model_name: str, num_labels: int, max_seq_length=2048):
    """Load using Unsloth for acceleration (Llama/Mistral models)."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth not installed. Run: pip install unsloth")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    if not hasattr(model, 'score'):
        model.score = torch.nn.Linear(model.config.hidden_size, num_labels)
        model.config.num_labels = num_labels
    return model, tokenizer

_VARIANT_LOADERS = {
    "standard":    load_hf_model,
    "compressed":  load_compressed_model,
    "hybrid_svd":  load_hybrid_svd_model,
    "unsloth":     load_unsloth_model,
}

def load_model(
    model_name: str,
    num_labels: int = 2,
    variant: str = "standard",
    cache_dir: str = None,
    **kwargs,
):
    """
    Main entry point: load any HuggingFace model + variant.
    Returns (model, tokenizer).
    """
    loader = _VARIANT_LOADERS.get(variant, load_hf_model)
    return loader(model_name, num_labels=num_labels, cache_dir=cache_dir, **kwargs)