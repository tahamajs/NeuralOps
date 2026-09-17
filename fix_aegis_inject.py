import re
with open("Codes/framework/attentions/aegis_hybrid.py", "r") as f:
    text = f.read()

new_sig = """def inject_aegis_hybrid(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[AegisHybridAttention, Any]]:
    r\"\"\"
    Injects AegisHybridAttention into a transformer layer with safe pre-test extraction.
    Returns None on failure to preserve the original layer.
    \"\"\"
    target = layer if layer is not None else model
    if target is None:
        return None
        
    if getattr(target, "_aegis_injected", False):
        return target
        
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_hybrid(
                layer=lyr, model_config=model_config, comp_config=comp_config,
                layer_idx=idx, total_layers=len(layer_list), **kwargs,
            )
            if res is not None:
                num_injected += 1
        if num_injected > 0:
            target._aegis_injected = True
        return target if num_injected > 0 else None
        
    try:"""

text = re.sub(
    r'def inject_aegis_hybrid\([\s\S]*?Returns None on failure to preserve the original layer.\n\s*\"\"\"\n\s*try:',
    new_sig,
    text
)

with open("Codes/framework/attentions/aegis_hybrid.py", "w") as f:
    f.write(text)
