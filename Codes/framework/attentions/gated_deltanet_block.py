import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

from .hgrn2_layer import HGRN2Layer  # adjust import as needed


@ATTENTION_REGISTRY.register("gated_deltanet_block")
class GatedDeltaNetBlock(nn.Module):
    """Block with HGRN2 and optional full attention."""

    def __init__(self, d_model, n_heads, use_full_attention=False):
        super().__init__()
        self.use_full_attention = use_full_attention
        if use_full_attention:
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.hgrn = HGRN2Layer(d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        if self.use_full_attention:
            attn_out, _ = self.attn(x, x, x)
            x = self.norm(x + attn_out)
        else:
            hgrn_out, _ = self.hgrn(x)
            x = self.norm(x + hgrn_out)
        return x, None


@ATTENTION_REGISTRY.register("gated_deltanet_block")
def inject_gated_deltanet_block(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    d_model = model_config.hidden_size
    n_heads = model_config.num_attention_heads
    use_full = getattr(comp_config, "use_full_attention", False)
    new_block = GatedDeltaNetBlock(d_model, n_heads, use_full)
    layer.attention = new_block
    return new_block
