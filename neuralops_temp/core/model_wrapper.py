"""
UnifiedTransformerModel – wraps any HuggingFace model with Hub support,
attention replacement, and config serialization.

Usage:
    model = UnifiedTransformerModel.from_pretrained("distilbert-base-uncased")
    model.replace_attention("galerkin", rank=32)
    model.save_pretrained("./compressed_model")
    # Later:
    loaded = UnifiedTransformerModel.from_pretrained("./compressed_model")
"""

from typing import Any

import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoModelForSequenceClassification

from .architecture import detect_architecture, get_transformer_layers
from .attention_utils import inject_attention


# ----------------------------------------------------------------------
# 1. Custom config class to hold our additional parameters
# ----------------------------------------------------------------------
class UnifiedTransformerConfig:
    """Lightweight container for the custom attention configuration."""

    def __init__(
        self,
        attention_type: str = None,
        attention_rank: int | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.attention_type = attention_type
        self.attention_rank = attention_rank
        self.attention_kwargs = attention_kwargs or {}
        # Store any extra kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON‑compatible dict."""
        return {
            "attention_type": self.attention_type,
            "attention_rank": self.attention_rank,
            "attention_kwargs": self.attention_kwargs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UnifiedTransformerConfig":
        """From dict."""
        return cls(**d)


# ----------------------------------------------------------------------
# 2. The unified model class with HubMixin
# ----------------------------------------------------------------------
class UnifiedTransformerModel(nn.Module, PyTorchModelHubMixin):
    """
    A wrapper around a HuggingFace model that supports custom attention
    and can be saved/loaded via the HuggingFace Hub.
    """

    def __init__(
        self, model: nn.Module, custom_config: UnifiedTransformerConfig | None = None
    ) -> None:
        super().__init__()
        self.model = model
        self.config = getattr(model, "config", None)  # HF model config
        self.arch = detect_architecture(model)
        self._custom_config = custom_config or UnifiedTransformerConfig()

    # ------------------------------------------------------------------
    # Attention replacement
    # ------------------------------------------------------------------
    def replace_attention(
        self,
        attention_type: str,
        rank: int | None = None,
        copy_weights: bool = True,
        **kwargs,
    ) -> Any:
        """
        Inject a custom attention variant into all layers.
        This method also records the choice so it can be saved.
        """
        inject_attention(self.model, attention_type, rank=rank, copy_weights=copy_weights, **kwargs)
        self._custom_config = UnifiedTransformerConfig(
            attention_type=attention_type,
            attention_rank=rank,
            attention_kwargs=kwargs,
        )
        return self

    # ------------------------------------------------------------------
    # Freezing / parameter count
    # ------------------------------------------------------------------
    def freeze_layers(self, num_layers_to_freeze: int = 0) -> Any:
        """Freeze layers."""
        layers = get_transformer_layers(self.model)
        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = i >= num_layers_to_freeze
        return self

    def unfreeze_all(self) -> Any:
        """Unfreeze all."""
        for param in self.model.parameters():
            param.requires_grad = True
        return self

    def get_parameter_count(self) -> dict[str, int]:
        """Return the parameter count."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # ------------------------------------------------------------------
    # Hub serialization – these methods are called by PyTorchModelHubMixin
    # ------------------------------------------------------------------
    def _save_pretrained(self, save_directory: str) -> None:
        """
        Save the underlying model and our custom config to a directory.
        """
        # 1. Save the base HF model (attention layers are already inside)
        self.model.save_pretrained(save_directory)
        # 2. Save our custom attention config to a JSON file
        import json

        custom_cfg_path = f"{save_directory}/unified_attention_config.json"
        with open(custom_cfg_path, "w") as f:
            json.dump(self._custom_config.to_dict(), f, indent=2)

    @classmethod
    def _from_pretrained(cls, model_id: str, **kwargs) -> "UnifiedTransformerModel":
        """
        Load a pretrained model and reconstruct the custom attention variant.
        This method is invoked by `from_pretrained` class method of the mixin.
        """
        # 1. Load the custom config
        import json
        import os

        if os.path.isdir(model_id):
            cfg_path = os.path.join(model_id, "unified_attention_config.json")
        else:
            # If model_id is a Hub repo, we need to download the file
            from huggingface_hub import hf_hub_download

            cfg_path = hf_hub_download(repo_id=model_id, filename="unified_attention_config.json")

        with open(cfg_path) as f:
            custom_cfg_dict = json.load(f)
        custom_cfg = UnifiedTransformerConfig.from_dict(custom_cfg_dict)

        # 2. Load the base HF model
        base_model = AutoModel.from_pretrained(model_id, **kwargs)

        # 3. Re‑apply the attention replacement if a type was stored
        model = cls(base_model, custom_config=custom_cfg)
        if custom_cfg.attention_type:
            model.replace_attention(
                custom_cfg.attention_type,
                rank=custom_cfg.attention_rank,
                copy_weights=False,  # Weights already inside the saved base model
                **custom_cfg.attention_kwargs,
            )
        return model

    # ------------------------------------------------------------------
    # Other methods
    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass — compute module output from inputs."""
        return self.model(*args, **kwargs)

    def save_pretrained(self, save_directory: str) -> None:
        """Convenience wrapper – delegates to the mixin."""
        super().save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        task: str = "sequence_classification",
        num_labels: int | None = None,
        **kwargs,
    ) -> "UnifiedTransformerModel":
        """
        Create a new UnifiedTransformerModel from a pretrained identifier.
        Automatically applies the custom attention if the model was saved with one.
        This is the main entry point for users.
        """
        # If the model_id has a unified_attention_config.json, use the mixin's from_pretrained
        # which will call our _from_pretrained. Otherwise, create a fresh one.
        import os

        has_custom = False
        if os.path.isdir(model_id):
            has_custom = os.path.exists(os.path.join(model_id, "unified_attention_config.json"))

        if has_custom:
            return cls._from_pretrained(model_id, **kwargs)
        else:
            # Fresh model – no custom attention yet
            if task == "sequence_classification":
                sc_kwargs = dict(kwargs)
                if num_labels is not None:
                    sc_kwargs["num_labels"] = num_labels
                base = AutoModelForSequenceClassification.from_pretrained(model_id, **sc_kwargs)
            else:
                base = AutoModel.from_pretrained(model_id, **kwargs)
            return cls(base)


# ----------------------------------------------------------------------
# 3. Auto‑class registration – so that AutoModel works automatically
# ----------------------------------------------------------------------

# Register our wrapper so that `AutoModel.from_pretrained("path")` can load it
# if the config.json contains our custom keys. (Simplified: always return UnifiedTransformerModel)
# In practice, you'd only do this if the config indicates it's a unified model.
# For simplicity, we override the auto class behavior for "unified_transformer".
# However, to avoid conflicts, we'll register a separate auto class name.

AutoModel.register(UnifiedTransformerConfig, UnifiedTransformerModel)
# This allows AutoModel.from_pretrained("your-repo/your-model") to return a UnifiedTransformerModel
# if the config.json contains the appropriate auto_map. We'll set that up when saving.
# For now, users can simply use UnifiedTransformerModel.from_pretrained directly.
