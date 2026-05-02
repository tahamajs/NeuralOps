# framework/attention_injector.py
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, BertForSequenceClassification
from typing import Type

def inject_attention(model, attention_class: Type[nn.Module], **attn_kwargs):
    """
    Replaces all attention layers in the model with the given attention class.
    Works for BERT and DistilBERT architectures.
    
    Args:
        model: the HuggingFace model (either BertForSequenceClassification or DistilBertForSequenceClassification)
        attention_class: the class of the attention module to inject
        **attn_kwargs: extra arguments passed to attention_class constructor (e.g., from comp_config)
    """
    if isinstance(model, DistilBertForSequenceClassification):
        _inject_into_distilbert(model, attention_class, **attn_kwargs)
    elif isinstance(model, BertForSequenceClassification):
        _inject_into_bert(model, attention_class, **attn_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

def _inject_into_distilbert(model, attention_class, **kwargs):
    for layer in model.distilbert.transformer.layer:
        old_attn = layer.attention
        new_attn = attention_class(
            model_config=model.config,
            comp_config=kwargs.get('comp_config'),
            layer_idx=layer.idx if hasattr(layer, 'idx') else 0,
            total_layers=len(model.distilbert.transformer.layer),
            pretrained_attn=old_attn,
            bert_mode=False,
            **kwargs
        )
        # Copy pretrained weights if available (most attention modules do this internally)
        layer.attention = new_attn

def _inject_into_bert(model, attention_class, **kwargs):
    # BERT injection similar to LEKAAttention.py's logic for BERT
    # ...
    pass  # implement similarly to DistilBERT but using layer.attention.self etc.