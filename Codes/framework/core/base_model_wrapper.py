from ..core.registry import ATTENTION_REGISTRY

def replace_attention_in_model(model, attention_type, rank):
    attention_class = ATTENTION_REGISTRY.get(attention_type)
    if attention_class is None:
        raise ValueError(f"Unknown attention type: {attention_type}")

    for module_name, module in model.named_modules():
        if "attention" in module_name.lower() and hasattr(module, "query"):
            parent_name, child_name = module_name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            new_attn = attention_class(model.config, rank=rank)
            # Copy weights from original Q/K/V projections
            new_attn.query.weight.data = module.query.weight.data
            new_attn.key.weight.data = module.key.weight.data
            new_attn.value.weight.data = module.value.weight.data
            if hasattr(module, "out_proj"):
                new_attn.out_proj = module.out_proj
            setattr(parent, child_name, new_attn)
    return model