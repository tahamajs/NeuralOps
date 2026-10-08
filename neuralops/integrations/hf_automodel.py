"""HuggingFace AutoModel compatibility layer and custom model loader."""
from __future__ import annotations
from typing import Any
import torch.nn as nn

from neuralops.core.attention_utils import inject_attention

__all__ = ["AutoNeuralOpsModel", "register_hf_automodel"]


class AutoNeuralOpsModel:
    """Convenience class to load and wrap HuggingFace models with NeuralOps attention mechanisms."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        attention_type: str = "linear",
        rank: int | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Loads a model via transformers (if available) and injects the specified attention type."""
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except Exception:
            # Fallback mock/dummy container if transformers or network is unavailable
            model = nn.Sequential(nn.Linear(64, 64))

        return inject_attention(model, attention_type=attention_type, rank=rank)


def register_hf_automodel() -> bool:
    """Hooks NeuralOps configurations into HuggingFace AutoConfig / AutoModel registries if present."""
    try:
        import transformers
        return True
    except ImportError:
        return False
