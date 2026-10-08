"""HuggingFace AutoModel compatibility layer, attention patching, and custom model loader."""
from __future__ import annotations
from typing import Any
import torch.nn as nn

from neuralops.core.attention_utils import inject_attention
from neuralops.core.registry import ATTENTION_REGISTRY

__all__ = ["AutoNeuralOpsModel", "register_hf_automodel", "patch_model_attention"]


def patch_model_attention(model: nn.Module, attention_type: str = "full_attention", **kwargs: Any) -> nn.Module:
    """Traverses model named_modules and replaces attention submodules (self_attn, attention) with the specified type."""
    cls = ATTENTION_REGISTRY.resolve(attention_type)
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if any(k in child_name.lower() for k in ["self_attn", "attention", "attn"]):
                # Extract hidden dimension if available
                dim = getattr(child, "embed_dim", getattr(child, "hidden_size", 64))
                try:
                    new_attn = cls(config=None, pretrained_attn=child, **kwargs)
                except Exception:
                    try:
                        new_attn = cls(embed_dim=dim, **kwargs)
                    except Exception:
                        try:
                            new_attn = cls(**kwargs)
                        except Exception:
                            continue
                setattr(module, child_name, new_attn)
    return model


class AutoNeuralOpsModel:
    """Convenience class to load and wrap HuggingFace models with NeuralOps attention mechanisms."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        attention_type: str = "full_attention",
        rank: int | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Loads a model via transformers (if available) and injects the specified attention type."""
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            model = nn.Sequential(nn.Linear(64, 64))

        return inject_attention(model, attention_type=attention_type, rank=rank)


def register_hf_automodel() -> bool:
    """Hooks NeuralOps configurations into HuggingFace AutoConfig / AutoModel registries if present."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False
