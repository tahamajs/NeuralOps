"""
Architecture introspection and projection weight extractors for NeuralOps.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

import torch.nn as nn

from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

INJECT_TO_CLASS_MAP: dict[str, str] = {
    "inject_replace": "replace_class",
    "inject_replace_attention": "replace_class",
    "inject_gqa": "gqa_attention_class",
    "inject_moba": "moba_class",
}


def _resolve_class_name(inject_fn: Any, model_type: str = "") -> str:
    """Infer registered attention class name from the injector or class."""
    if inspect.isclass(inject_fn):
        for k, v in ATTENTION_REGISTRY.items():
            if v is inject_fn:
                return k
        return getattr(inject_fn, "__name__", "replace_class")
    name = getattr(inject_fn, "__name__", "")
    if name in INJECT_TO_CLASS_MAP:
        return INJECT_TO_CLASS_MAP[name]
    # Fallback: strip 'inject_' and append '_class'
    if name.startswith("inject_"):
        candidate = name[len("inject_") :] + "_class"
        if ATTENTION_REGISTRY.get(candidate) is not None:
            return candidate
        candidate2 = name[len("inject_") :]
        if ATTENTION_REGISTRY.get(candidate2) is not None:
            return candidate2
    return "replace_class"


def _infer_hidden_size(old_attn: nn.Module, attn_attr: str = "", model_type: str = "") -> int:
    for attr in ("hidden_size", "embed_dim", "d_model"):
        val = getattr(old_attn, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    cfg = getattr(old_attn, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "embed_dim", "d_model"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    for attr_path in ("query", "q_lin", "q_proj", "c_attn", "self.query"):
        obj = old_attn
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "in_features"):
            return int(obj.in_features)
        if obj is not None and hasattr(obj, "weight"):
            w = obj.weight
            if model_type == "gpt2" or obj.__class__.__name__ == "Conv1D":
                return int(w.shape[0])
            return int(w.shape[1])
    # Also try nested attention.self.query
    if hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
        return int(old_attn.self.query.in_features)
    raise ValueError(
        f"Cannot infer hidden_size from {type(old_attn).__name__} "
        f"(model_type={model_type}, attn_attr={attn_attr})"
    )


def _infer_num_heads(old_attn: nn.Module, attn_attr: str = "", model_type: str = "") -> int:
    for attr in ("num_attention_heads", "num_heads", "n_heads"):
        val = getattr(old_attn, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    # Check config attached to old_attn
    cfg = getattr(old_attn, "config", None)
    if cfg is not None:
        for attr in ("num_attention_heads", "num_heads", "n_heads"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    # Infer from q_proj and head_dim (Qwen, LLaMA)
    if hasattr(old_attn, "q_proj") and hasattr(old_attn, "head_dim"):
        q_proj = old_attn.q_proj
        if hasattr(q_proj, "out_features") and old_attn.head_dim > 0:
            return int(q_proj.out_features // old_attn.head_dim)
    raise ValueError(f"Cannot infer num_heads from {type(old_attn).__name__}")


def _replace_bert_attention(
    layer: nn.Module, inject_fn: Any, model_config: Any = None, **kwargs: Any
) -> None:
    from neuralops.core._attention_helpers import _replace_layers

    _replace_layers([layer], "attention", inject_fn, model_config, kwargs, "bert")


def _replace_gpt_attention(
    layer: nn.Module, inject_fn: Any, model_config: Any = None, **kwargs: Any
) -> None:
    from neuralops.core._attention_helpers import _replace_layers

    _replace_layers([layer], "attn", inject_fn, model_config, kwargs, "gpt2")


def _replace_distilbert_attention(
    layer: nn.Module, inject_fn: Any, model_config: Any = None, **kwargs: Any
) -> None:
    from neuralops.core._attention_helpers import _replace_layers

    _replace_layers([layer], "attention", inject_fn, model_config, kwargs, "distilbert")


__all__ = [
    "_resolve_class_name",
    "_infer_hidden_size",
    "_infer_num_heads",
    "_replace_bert_attention",
    "_replace_gpt_attention",
    "_replace_distilbert_attention",
]
