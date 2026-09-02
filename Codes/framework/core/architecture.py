"""
Model‑agnostic architecture helpers.
Provides unified access to layers, attention, FFN, embeddings, classifiers
across all common HuggingFace architectures.
"""

from typing import List, Optional, Tuple

import torch.nn as nn


def detect_architecture(model: nn.Module) -> str:
    """Return canonical architecture name."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "") if config else ""
    name = model.__class__.__name__.lower()
    if "distilbert" in model_type or "distilbert" in name:
        return "distilbert"
    if "roberta" in model_type or "roberta" in name:
        return "roberta"
    if "bert" in model_type or "bert" in name:
        return "bert"
    if "gpt2" in model_type or "gpt2" in name:
        return "gpt2"
    if "qwen" in model_type or "qwen" in name:
        return "qwen"
    if "llama" in model_type:
        return "llama"
    if "mistral" in model_type:
        return "mistral"
    if "gemma" in model_type:
        return "gemma"
    if "phi" in model_type:
        return "phi"
    if "t5" in model_type:
        return "t5"
    return "unknown"


def get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """Return a flat list of transformer blocks."""
    # Llama / Gemma / Phi / Mistral / Falcon
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # GPT‑2 / OPT / GPT‑J
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # BERT / RoBERTa / DistilBert / ALBERT
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return list(model.encoder.layer)
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        return list(model.bert.encoder.layer)
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        return list(model.roberta.encoder.layer)
    if hasattr(model, "distilbert") and hasattr(model.distilbert, "transformer"):
        return list(model.distilbert.transformer.layer)
    # T5 encoder / decoder
    if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        return list(model.encoder.block)
    if hasattr(model, "decoder") and hasattr(model.decoder, "block"):
        return list(model.decoder.block)
    raise ValueError("Could not locate transformer layers automatically.")


def get_attention_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return (name, module) for all leaf attention sub-modules."""
    attn_modules = []
    for name, module in model.named_modules():
        if not name:
            continue
        class_name = module.__class__.__name__.lower()
        if any(x in class_name for x in ["attention", "selfattn", "selfattention"]):
            attn_modules.append((name, module))
            continue
        # Attribute-based detection
        if (
            hasattr(module, "q_proj")
            or hasattr(module, "k_proj")
            or hasattr(module, "q_lin")
            or hasattr(module, "query")
        ):
            attn_modules.append((name, module))

    # Crucial: Filter out parent composite containers whose children are also attention modules.
    # For example, in RoBERTa/BERT:
    # "roberta.encoder.layer.0.attention" is a container that includes "self" (RobertaSelfAttention)
    # and "output" (RobertaSelfOutput: dense, LayerNorm, residual connection).
    # Replacing the outer container destroys LayerNorm and the residual stream!
    # We must only return the innermost/leaf self-attention module.
    all_names = [name for name, _ in attn_modules]
    leaf_modules = []
    for name, module in attn_modules:
        is_parent_container = any(other != name and other.startswith(name + ".") for other in all_names)
        is_output_submodule = name.endswith(".output") or name.endswith("_output")
        if not is_parent_container and not is_output_submodule:
            leaf_modules.append((name, module))

    return leaf_modules


def get_ffn_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Return (name, module) for all feed‑forward blocks."""
    ffn_modules = []
    for name, module in model.named_modules():
        class_name = module.__class__.__name__.lower()
        if any(x in class_name for x in ["mlp", "ffn", "feedforward"]):
            ffn_modules.append((name, module))
            continue
        if (
            hasattr(module, "gate_proj")
            or hasattr(module, "up_proj")
            or hasattr(module, "intermediate")
        ):
            ffn_modules.append((name, module))
    return ffn_modules


def get_embedding_module(model: nn.Module) -> Optional[nn.Module]:
    """Return the input word embedding module."""
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return module
    return None


def get_classifier_head(model: nn.Module) -> Optional[nn.Module]:
    """Return the final classification layer."""
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "score"):
        return model.score
    if hasattr(model, "pre_classifier"):
        return model.pre_classifier, model.classifier
    return None
