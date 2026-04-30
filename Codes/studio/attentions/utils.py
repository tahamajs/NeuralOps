# studio/attentions/utils.py
from ..core.registry import ATTENTION_REGISTRY

def replace_attention_in_model(model, attention_type, rank=None):
    attn_class = ATTENTION_REGISTRY.get(attention_type)
    if attn_class is None:
        raise ValueError(f"Unknown attention type: {attention_type}")
    for name, mod in model.named_modules():
        if "attention" in name.lower() and hasattr(mod, "query"):
            parent = model.get_submodule('.'.join(name.split('.')[:-1])) if '.' in name else model
            child = name.split('.')[-1]
            new_attn = attn_class(model.config, rank=rank)
            # copy weights
            if hasattr(new_attn, 'query') and hasattr(mod, 'query'):
                new_attn.query.weight.data = mod.query.weight.data
                if hasattr(new_attn, 'key'): new_attn.key.weight.data = mod.key.weight.data
                if hasattr(new_attn, 'value'): new_attn.value.weight.data = mod.value.weight.data
                if hasattr(new_attn, 'out_proj'): new_attn.out_proj = mod.out_proj
            setattr(parent, child, new_attn)
    return model