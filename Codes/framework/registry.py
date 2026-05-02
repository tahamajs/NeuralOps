# framework/registry.py
"""
Registry for attention modules.
Maps string names to attention classes.
"""

from attentions.leka_v2 import LEKAv2Attention
from attentions.spectra_v2 import SpectraV2Attention
from attentions.orion import ORIONAttention
from attentions.aegis import AegisAttention          # pick your best Aegis
from attentions.prism import PRISMAttention          # or MinimalPRISM
from attentions.nova import NOVAAttention            # or MinimalNOVA
from attentions.aurora import AURORAAttention        # or MinimalAURORA
from attentions.svd_compressed import SVDCompressedAttention

ATTENTION_REGISTRY = {
    'leka_v2': LEKAv2Attention,
    'spectra_v2': SpectraV2Attention,
    'orion': ORIONAttention,
    'aegis': AegisAttention,
    'prism': PRISMAttention,
    'nova': NOVAAttention,
    'aurora': AURORAAttention,
    'svd_compressed': SVDCompressedAttention,
}


"""
Attention Registry – maps a string (e.g., 'leka_v2') to its injection function.
Import your attention modules here and add them to the dictionary.
"""

# Example: import your injection functions
# from attentions.leka_v2 import inject_leka_v2_into_distilbert
# from attentions.spectra_v2 import inject_spectra_v2_into_distilbert
# from attentions.orion import inject_orion_into_distilbert
# from attentions.aegis import inject_aegis_into_distilbert

ATTENTION_REGISTRY = {}

def register_attention(name: str):
    """Decorator to register an attention injection function."""
    def decorator(fn):
        ATTENTION_REGISTRY[name] = fn
        return fn
    return decorator

# -------------------------------------------------------------------
# Register your attention types below (examples)
# You can either use the decorator or manually add to the dictionary.

@register_attention("full")
def inject_full_attention(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """Full attention – do nothing (keep the original)."""
    pass

# @register_attention("leka_v2")
# def inject_leka_v2(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
#     from attentions.leka_v2 import LEKAv2Attention
#     new_attn = LEKAv2Attention(config=model_config, comp_config=comp_config,
#                                layer_idx=layer_idx, total_layers=total_layers,
#                                pretrained_attn=old_attn)
#     layer.attention = new_attn
#     return new_attn

# Add similar registrations for spectra_v2, orion, aegis, etc.