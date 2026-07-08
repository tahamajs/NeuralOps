import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("hgrn2_layer")
class HGRN2Layer(nn.Module):
    """HGRN‑2 linear recurrence with controlled forgetting."""

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
        out = torch.stack(outs, dim=1)
        return out, None  # return tuple for attention interface


@ATTENTION_REGISTRY.register("hgrn2_layer")
def inject_hgrn2_layer(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    d_model = model_config.hidden_size
    new_layer = HGRN2Layer(d_model)
    layer.attention = new_layer
    return new_layer
