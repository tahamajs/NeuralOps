"""Model instantiation and dynamic attention layer replacement factory."""

from __future__ import annotations
import sys
import traceback
from pathlib import Path
from typing import Tuple, Optional, Any, Dict
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CODES_DIR = PROJECT_ROOT / "Codes"
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from .registry_utils import get_inject_function, has_inject_function
try:
    from framework.core.registry import ATTENTION_REGISTRY
    import framework.attentions
except ImportError:
    ATTENTION_REGISTRY = {}


def get_transformer_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """Extract the sequential list of transformer layer blocks from standard HuggingFace models."""
    if hasattr(model, "distilbert") and hasattr(model.distilbert, "transformer"):
        return model.distilbert.transformer.layer
    if hasattr(model, "bert") and hasattr(model.bert, "encoder"):
        return model.bert.encoder.layer
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return model.transformer.layer
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        return model.roberta.encoder.layer
    return None


def build_model_with_attention(
    model_name: str,
    base_model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    pretrained: bool = True,
    **kwargs: Any,
) -> Tuple[Optional[nn.Module], str]:
    """
    Construct a sequence classification model with standard attention layers replaced
    by the specified custom attention module from ATTENTION_REGISTRY.

    Args:
        model_name: Identifier registered in ATTENTION_REGISTRY (or 'full' for baseline).
        base_model_name: Base HuggingFace backbone identifier.
        num_labels: Target number of classification classes or 1 for regression.
        pretrained: Whether to load pre-trained backbone checkpoint weights.
        **kwargs: Additional parameters passed to attention constructors.

    Returns:
        Tuple of (instantiated_model, status_message).
    """
    model_key = model_name.strip().lower()

    try:
        config = AutoConfig.from_pretrained(base_model_name, num_labels=num_labels)
        if num_labels == 1:
            config.problem_type = "regression"

        if pretrained:
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                config=config,
            )
        else:
            model = AutoModelForSequenceClassification.from_config(config)

        if model_key == "full":
            return model, "OK"

        layers = get_transformer_layers(model)
        if layers is None:
            return None, f"FAIL_BUILD: Unsupported backbone architecture '{base_model_name}'."

        total_layers = len(layers)
        inject_fn = get_inject_function(model_key)

        for i, layer in enumerate(layers):
            old_attn = getattr(layer, "attention", None)

            if inject_fn is not None:
                new_attn = inject_fn(
                    layer=layer,
                    model_config=config,
                    layer_idx=i,
                    total_layers=total_layers,
                    old_attn=old_attn,
                    **kwargs,
                )
                layer.attention = new_attn
            else:
                candidate_keys = [model_key, f"{model_key}_attention", f"{model_key}_layer", f"{model_key}_module"]
                attn_cls = None
                for ck in candidate_keys:
                    candidate = ATTENTION_REGISTRY.get(ck)
                    if candidate is not None and isinstance(candidate, type):
                        attn_cls = candidate
                        break

                if attn_cls is None:
                    return None, f"FAIL_BUILD: Attention model '{model_name}' not found in ATTENTION_REGISTRY."

                try:
                    new_attn = attn_cls(config, **kwargs)
                except TypeError:
                    try:
                        new_attn = attn_cls(d_model=config.hidden_size, n_heads=config.num_attention_heads, **kwargs)
                    except Exception:
                        new_attn = attn_cls(**kwargs)

                layer.attention = new_attn

        return model, "OK"

    except Exception as e:
        err = traceback.format_exc()
        return None, f"FAIL_BUILD: {str(e)}\n{err}"


def extract_attention_outputs(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    output_attentions: bool = True,
) -> Any:
    """
    Execute forward pass with attention inspection enabled, returning layer-wise attention matrices.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=output_attentions)
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            return outputs.attentions
    return None
