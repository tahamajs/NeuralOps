import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


class Mamba(nn.Module):
    """Trivial Mamba placeholder – replace with official implementation."""

    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, **kwargs):
        return self.out_proj(F.silu(self.in_proj(x)))


@ATTENTION_REGISTRY.register("mamba")
def inject_mamba(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    d_model = model_config.hidden_size
    new_module = Mamba(d_model)
    layer.attention = new_module
    return new_module
