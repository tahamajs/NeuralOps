"""
High‑level model wrapper that simplifies attention replacement and freezing.
"""
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoModel, AutoModelForSequenceClassification
from .attention_utils import inject_attention
from .architecture import detect_architecture, get_transformer_layers


class UnifiedTransformerModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.config = getattr(model, 'config', None)
        self.arch = detect_architecture(model)

    def replace_attention(self, attention_type: str, rank: Optional[int] = None,
                          copy_weights: bool = True, **kwargs):
        inject_attention(self.model, attention_type, rank=rank, copy_weights=copy_weights, **kwargs)
        return self

    def freeze_layers(self, num_layers_to_freeze: int = 0):
        layers = get_transformer_layers(self.model)
        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = i >= num_layers_to_freeze
        return self

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
        return self

    def get_parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_pretrained(self, path: str):
        self.model.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, model_name: str, task: str = "sequence_classification",
                        num_labels: Optional[int] = 2, **kwargs):
        if task == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels, **kwargs)
        else:
            model = AutoModel.from_pretrained(model_name, **kwargs)
        return cls(model)