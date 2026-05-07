import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY
from .gated_deltanet_block import GatedDeltaNetBlock   # adjust import

@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet")
class HGRN2GatedDeltaNet(nn.Module):
    """Hybrid model with configurable linear‑to‑full ratio."""
    def __init__(self, d_model, n_heads, n_layers, ratio=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            use_full = (i % (ratio+1) == ratio)
            self.layers.append(GatedDeltaNetBlock(d_model, n_heads, use_full))

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x, _ = layer(x)
        return x, None

@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet")
def inject_hgrn2_gated_deltanet(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    d_model = model_config.hidden_size
    n_heads = model_config.num_attention_heads
    n_layers = total_layers
    ratio = getattr(comp_config, 'ratio', 3)
    new_net = HGRN2GatedDeltaNet(d_model, n_heads, n_layers, ratio)
    layer.attention = new_net
    return new_net