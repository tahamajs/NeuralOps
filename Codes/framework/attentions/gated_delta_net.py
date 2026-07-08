import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


class HGRN2Layer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.forget_gate = nn.Linear(d_model, d_model)
        self.input_gate = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.res_clip = 1.0

    def forward(self, x, **kwargs):
        B, L, D = x.shape
        h = torch.zeros(B, D, device=x.device)
        outs = []
        for t in range(L):
            xt = x[:, t]
            f = torch.sigmoid(self.forget_gate(xt))
            i = torch.sigmoid(self.input_gate(xt))
            h = f * h + i * xt
            h = torch.clamp(h, -self.res_clip, self.res_clip)
            outs.append(self.output(h))
        return torch.stack(outs, dim=1)


class GatedDeltaNetBlock(nn.Module):
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
            x = self.norm(x + self.hgrn(x))
        return x, None  # return tuple for attention interface


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
