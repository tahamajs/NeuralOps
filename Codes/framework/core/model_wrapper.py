# framework/core/model_wrapper.py
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoConfig,
)
from .registry import ATTENTION_REGISTRY
from .architecture import detect_architecture, get_attention_modules, replace_attention_module

class UnifiedTransformerModel(nn.Module):
    def __init__(self, model: nn.Module, config: Optional[Any] = None):
        super().__init__()
        self.model = model
        self.config = config or getattr(model, 'config', None)
        self.architecture = detect_architecture(model)

    def replace_attention(self, attention_type: str, rank: Optional[int] = None, **kwargs):
        attn_class = ATTENTION_REGISTRY.get(attention_type)
        if attn_class is None:
            raise ValueError(f"Unknown attention type: {attention_type}")
        replace_attention_module(self.model, attn_class, rank=rank, **kwargs)
        return self

    def freeze_layers(self, num_layers_to_freeze: int = 6):
        layers = self._get_transformer_layers()
        for i, layer in enumerate(layers):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        return self

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
        return self

    def get_attention_maps(self, input_ids, attention_mask=None, **kwargs):
        self.model.eval()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_attentions=True, **kwargs)
        return outputs.attentions if hasattr(outputs, 'attentions') else []

    def get_hidden_states(self, input_ids, attention_mask=None, **kwargs):
        self.model.eval()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True, **kwargs)
        return outputs.hidden_states if hasattr(outputs, 'hidden_states') else []

    def get_parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_name: str, task: str = "sequence_classification",
                        num_labels: Optional[int] = None, **kwargs):
        if task == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels or 2, **kwargs)
        elif task == "qa":
            model = AutoModelForQuestionAnswering.from_pretrained(model_name, **kwargs)
        elif task == "token_classification":
            model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels or 2, **kwargs)
        else:
            model = AutoModel.from_pretrained(model_name, **kwargs)
        config = AutoConfig.from_pretrained(model_name)
        return cls(model, config)

    def _get_transformer_layers(self):
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'encoder'):
            return self.model.bert.encoder.layer
        if hasattr(self.model, 'roberta') and hasattr(self.model.roberta, 'encoder'):
            return self.model.roberta.encoder.layer
        if hasattr(self.model, 'distilbert') and hasattr(self.model.distilbert, 'transformer'):
            return self.model.distilbert.transformer.layer
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            return self.model.encoder.layer
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        return []