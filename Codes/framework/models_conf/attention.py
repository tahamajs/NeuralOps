"""Universal attention injection that works on any HF model."""

import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

def _get_transformer_layers(model):
    for attr in ['transformer.h', 'encoder.layer', 'transformer.layer',
                 'distilbert.transformer.layer', 'bert.encoder.layer',
                 'roberta.encoder.layer', 'gpt_neox.layers']:
        obj = model
        for part in attr.split('.'):
            obj = getattr(obj, part, None)
            if obj is None: break
        if obj is not None: return obj
    for _, module in model.named_children():
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            return module
    raise ValueError("Could not locate transformer layers in model")

def _get_attention_module(layer):
    for attr in ['attention', 'self_attn', 'attn']:
        attn = getattr(layer, attr, None)
        if attn is not None: return attn
    return None

def replace_attention_in_model(model, attention_name, **kwargs):
    injection_fn = ATTENTION_REGISTRY.resolve(attention_name)
    layers = list(_get_transformer_layers(model))
    for idx, layer in enumerate(layers):
        old_attn = _get_attention_module(layer)
        if old_attn is not None:
            injection_fn(
                layer=layer,
                model_config=model.config,
                comp_config=kwargs.get('comp_config', type('CompConfig', (), {})()),
                layer_idx=idx,
                total_layers=len(layers),
                old_attn=old_attn,
            )
    return model