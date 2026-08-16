from abc import ABC, abstractmethod

from framework.models_conf import (CompressedBertModel,
                                   HybridBertForSequenceClassification,
                                   load_compressed_model, load_hf_model,
                                   load_hybrid_svd_model, load_model,
                                   load_unsloth_model,
                                   load_unsloth_model_with_attention)

try:
    from framework.attentions.replace import replace_attention_in_model
except ImportError:
    pass


class BaseModel(ABC):
    """
    Base class for all models in the framework.
    """
    @abstractmethod
    def forward(self, *args, **kwargs) -> None:
        """Forward pass — compute module output from inputs."""
        pass

    @abstractmethod
    def get_config(self) -> None:
        """Return the config."""
        pass


__all__ = [
    "BaseModel",
    "load_model",
    "load_hf_model",
    "load_compressed_model",
    "load_hybrid_svd_model",
    "load_unsloth_model",
    "load_unsloth_model_with_attention",
    "CompressedBertModel",
    "HybridBertForSequenceClassification",
    "replace_attention_in_model",
]

