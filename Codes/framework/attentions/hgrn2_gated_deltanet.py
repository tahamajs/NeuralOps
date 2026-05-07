import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("hgrn2_layer")
class HGRN2Layer(nn.Module):
    """HGRN‑2 linear recurrence with controlled forgetting."""
    def __init__(self, d_model):
        super().__init__()
        self.forget_gate = nn.Linear(d_model, d_model)
        self.input_gate = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.res_clip = 1.0

    def forward(self, x):
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

    def forward(self, x):
        if self.use_full_attention:
            attn_out, _ = self.attn(x, x, x)
            x = self.norm(x + attn_out)
        else:
            x = self.norm(x + self.hgrn(x))
        return x

@ATTENTION_REGISTRY.register("hgrn2_gated_deltanet")
class HGRN2GatedDeltaNet(nn.Module):
    """Hybrid model with configurable linear‑to‑full ratio."""
    def __init__(self, d_model, n_heads, n_layers, ratio=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            use_full = (i % (ratio+1) == ratio)
            self.layers.append(GatedDeltaNetBlock(d_model, n_heads, use_full_attention=use_full))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x