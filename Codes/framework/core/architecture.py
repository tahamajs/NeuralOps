"""
Model‑agnostic architecture helpers.
Detects model type and provides unified access to layers, attention,
FFN, embeddings, and classification heads for any HuggingFace model.
"""

import torch.nn as nn
from typing import List, Tuple, Optional, Union


# ------------------------------------------------------------------ #
# Architecture detection
# ------------------------------------------------------------------ #
def detect_architecture(model: nn.Module) -> str:
    """Return a canonical architecture name for the model."""
    config = getattr(model, 'config', None)
    model_type = getattr(config, 'model_type', '') if config else ''
    name = model.__class__.__name__.lower()

    if 'distilbert' in model_type or 'distilbert' in name:
        return 'distilbert'
    if 'roberta' in model_type or 'roberta' in name:
        return 'roberta'
    if 'bert' in model_type or 'bert' in name:
        return 'bert'
    if 'gpt2' in model_type or 'gpt2' in name:
        return 'gpt2'
    if 'llama' in model_type:
        return 'llama'
    if 'mistral' in model_type:
        return 'mistral'
    if 'gemma' in model_type:
        return 'gemma'
    if 'phi' in model_type:
        return 'phi'
    if 't5' in model_type:
        return 't5'
    return 'unknown'


# ------------------------------------------------------------------ #
# Transformer block access
# ------------------------------------------------------------------ #
def get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """
    Return a flat list of the main transformer block modules.
    Covers all common HuggingFace architectures.
    """
    # Llama / Gemma / Phi / Mistral / Falcon
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)

    # GPT‑2 / OPT / GPT‑J
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return list(model.transformer.h)

    # BERT / RoBERTa / DistilBert / ALBERT
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return list(model.encoder.layer)
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        return list(model.bert.encoder.layer)
    if hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder'):
        return list(model.roberta.encoder.layer)
    if hasattr(model, 'distilbert') and hasattr(model.distilbert, 'transformer'):
        return list(model.distilbert.transformer.layer)

    # T5 encoder / decoder
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
        return list(model.encoder.block)
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
        return list(model.decoder.block)

    raise ValueError(
        "Could not locate transformer layers automatically. "
        "Please provide a custom layer getter."
    )


# ------------------------------------------------------------------ #
# Attention module access
# ------------------------------------------------------------------ #
def get_attention_modules(
    model: nn.Module
) -> List[Tuple[str, nn.Module]]:
    """
    Return a list of (name, module) for all attention sub‑modules.
    The heuristic looks for typical attention class names and attribute names.
    """
    attn_modules = []
    for name, module in model.named_modules():
        # Class‑based detection
        class_name = module.__class__.__name__.lower()
        if any(x in class_name for x in ['attention', 'selfattn', 'selfattention']):
            attn_modules.append((name, module))
            continue

        # Attribute‑based detection (fallback)
        if hasattr(module, 'q_proj') or hasattr(module, 'k_proj') or \
           hasattr(module, 'q_lin') or hasattr(module, 'query'):
            attn_modules.append((name, module))

    return attn_modules


# ------------------------------------------------------------------ #
# FFN / MLP module access
# ------------------------------------------------------------------ #
def get_ffn_modules(
    model: nn.Module
) -> List[Tuple[str, nn.Module]]:
    """Return (name, module) for all feed‑forward / MLP blocks."""
    ffn_modules = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if any(x in class_name for x in ['mlp', 'ffn', 'feedforward']):
            ffn_modules.append((name, module))
            continue
        # Heuristic: check for typical FFN structure
        if hasattr(module, 'gate_proj') or hasattr(module, 'up_proj') or \
           hasattr(module, 'intermediate'):
            ffn_modules.append((name, module))

    return ffn_modules


# ------------------------------------------------------------------ #
# Embedding module access
# ------------------------------------------------------------------ #
def get_embedding_module(model: nn.Module) -> Optional[nn.Module]:
    """
    Return the input word embedding module.
    Uses the standard HuggingFace method when available.
    """
    if hasattr(model, 'get_input_embeddings'):
        return model.get_input_embeddings()

    # Llama / Gemma / Phi
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens

    # GPT‑2 / OPT
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte

    # Fallback: search for nn.Embedding (may pick up position embeddings)
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return module
    return None


# ------------------------------------------------------------------ #
# Classifier head access (for classification models)
# ------------------------------------------------------------------ #
def get_classifier_head(model: nn.Module) -> Optional[nn.Module]:
    """Return the final classification layer (if present)."""
    # HuggingFace AutoModelForSequenceClassification stores head in 'classifier'
    if hasattr(model, 'classifier'):
        return model.classifier
    if hasattr(model, 'score'):
        return model.score
    # DistilBert stores pre_classifier + classifier
    if hasattr(model, 'pre_classifier'):
        return model.pre_classifier, model.classifier
    return None