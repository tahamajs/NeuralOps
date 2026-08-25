r"""
Model Attention Replacement and Dynamic Graph Surgery Utilities.

Provides unified dispatch interfaces to substitute attention sub-layers in pretrained
HuggingFace transformer architectures (BERT, RoBERTa, DistilBERT, GPT-2) with custom efficient attention mechanisms.

Mathematical Mechanism:
    Given a transformer computation graph $\mathcal{G} = (V, E)$ containing self-attention operators
    $f_{\text{attn}}^{(l)}: \mathbb{R}^{B \times L \times D} \to \mathbb{R}^{B \times L \times D}$ for layers $l \in \{1, \dots, L\}$,
    the replacement surgery executes an in-place operator mutation:
    $$\mathcal{G} \xmapsto{\text{replace}} \mathcal{G}' \quad \text{where } f_{\text{attn}}^{(l)} \leftarrow \widetilde{f}_{\text{attn}}^{(l)}$$
    transferring or re-initializing projection weights according to registry specifications.

Computational Complexity:
    - Graph traversal & module injection: $\mathcal{O}(L)$ operations over $L$ transformer layers.
"""

from typing import Any, Dict, Optional

import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


def replace_attention_in_model(
    model: nn.Module,
    attention_name: str,
    attention_params: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    r"""Replace all self-attention layers in a transformer model with target attention variant.

    Args:
        model: Pretrained transformer neural network module (`nn.Module`).
        attention_name: Identifier registered in `ATTENTION_REGISTRY`.
        attention_params: Optional dictionary of hyperparameter overrides.

    Returns:
        Mutated model instance with updated attention sub-layers.
    """
    if attention_params is None:
        attention_params = {}

    model_type: Optional[str] = getattr(getattr(model, "config", None), "model_type", None)

    if model_type in ("bert", "roberta"):
        _replace_bert_attention(model, attention_name, attention_params)
    elif model_type == "gpt2":
        _replace_gpt_attention(model, attention_name, attention_params)
    elif model_type == "distilbert":
        _replace_distilbert_attention(model, attention_name, attention_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def _replace_bert_attention(
    model: nn.Module,
    attention_name: str,
    attention_params: Dict[str, Any],
) -> None:
    r"""In-place replacement of BERT / RoBERTa attention sub-layers."""
    encoder = getattr(model, "encoder", None) or getattr(getattr(model, "bert", None), "encoder", None)
    if encoder is None:
        raise AttributeError("Could not locate encoder in model architecture.")
    inject_fn = ATTENTION_REGISTRY.get(attention_name)
    if inject_fn is None:
        raise ValueError(f"Attention '{attention_name}' not registered in ATTENTION_REGISTRY.")
        
    layers = encoder.layer
    total_layers = len(layers)
    for idx, layer in enumerate(layers):
        old_attn = layer.attention.self
        
        # Try to use the injection function directly first
        try:
            import inspect
            if hasattr(inject_fn, "__code__") and "layer_idx" in inspect.getfullargspec(inject_fn).args:
                new_attn = inject_fn(layer, getattr(model, "config", None), attention_params, idx, total_layers, old_attn)
                if getattr(layer, "attention", None) is not new_attn:
                    # Some injectors replace the whole layer.attention; others just return the module.
                    layer.attention.self = new_attn
                continue
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
            pass
            
        # Fallback to build_attention
        hidden_size = old_attn.query.in_features
        num_heads = old_attn.num_attention_heads
        new_attn = build_attention(
            attention_name, hidden_size, num_heads, **attention_params
        )
        layer.attention.self = new_attn


def _replace_gpt_attention(
    model: nn.Module,
    attention_name: str,
    attention_params: Dict[str, Any],
) -> None:
    r"""In-place replacement of GPT-2 block attention sub-layers."""
    transformer = getattr(model, "transformer", None)
    if transformer is None:
        raise AttributeError("Could not locate transformer in GPT-2 model architecture.")
    inject_fn = ATTENTION_REGISTRY.get(attention_name)
    if inject_fn is None:
        raise ValueError(f"Attention '{attention_name}' not registered in ATTENTION_REGISTRY.")

    layers = transformer.h
    total_layers = len(layers)
    for idx, block in enumerate(layers):
        old_attn = block.attn
        
        try:
            import inspect
            if hasattr(inject_fn, "__code__") and "layer_idx" in inspect.getfullargspec(inject_fn).args:
                new_attn = inject_fn(block, getattr(model, "config", None), attention_params, idx, total_layers, old_attn)
                if getattr(block, "attn", None) is not new_attn:
                    block.attn = new_attn
                continue
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
            pass
            
        hidden_size = old_attn.embed_dim
        num_heads = old_attn.num_heads
        new_attn = build_attention(
            attention_name, hidden_size, num_heads, **attention_params
        )
        block.attn = new_attn


def _replace_distilbert_attention(
    model: nn.Module,
    attention_name: str,
    attention_params: Dict[str, Any],
) -> None:
    r"""In-place replacement of DistilBERT transformer attention sub-layers."""
    distilbert = getattr(model, "distilbert", model)
    layers = distilbert.transformer.layer
    total_layers = len(layers)
    inject_fn = ATTENTION_REGISTRY.get(attention_name)
    if inject_fn is None:
        raise ValueError(f"Attention '{attention_name}' not registered in ATTENTION_REGISTRY.")

    for idx, layer in enumerate(layers):
        old_attn = layer.attention
        res = inject_fn(
            layer,
            model.config,
            attention_params,
            idx,
            total_layers,
            old_attn,
        )
        if isinstance(res, nn.Module) and getattr(layer, "attention", None) is not res:
            layer.attention = res


def build_attention(
    attention_name: str,
    hidden_size: int,
    num_heads: int,
    **kwargs: Any,
) -> nn.Module:
    r"""Instantiate an attention module from the registry."""
    attn_cls = ATTENTION_REGISTRY.get(attention_name)
    if attn_cls is None:
        raise ValueError(f"Attention '{attention_name}' not registered in ATTENTION_REGISTRY.")
    if hasattr(attn_cls, "__call__"):
        import inspect
        try:
            # First try standard signature
            return attn_cls(hidden_size=hidden_size, num_heads=num_heads, **kwargs)
        except TypeError:
            try:
                # Try with num_attention_heads
                return attn_cls(hidden_size=hidden_size, num_attention_heads=num_heads, **kwargs)
            except TypeError:
                try:
                    # Try with config object
                    from argparse import Namespace
                    config = Namespace(hidden_size=hidden_size, num_attention_heads=num_heads, **kwargs)
                    return attn_cls(config=config, **kwargs)
                except Exception as e:
                    raise ValueError(f"Failed to instantiate '{attention_name}': {e}")
        except Exception as e:
            raise ValueError(f"Failed to instantiate '{attention_name}': {e}")
            
    raise ValueError(f"Attention '{attention_name}' is not callable.")
