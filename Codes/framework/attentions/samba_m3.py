import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

from .jamba2 import Mamba  # import the stub or replace with official Mamba


@ATTENTION_REGISTRY.register("spectral_mamba")
class SpectralMamba(nn.Module):
    """Mamba with learnable spectral scaling."""

    def __init__(self, d_model):
        super().__init__()
        self.mamba = Mamba(d_model)
        self.dt_scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        out = self.mamba(x)
        out = out * torch.sigmoid(self.dt_scale).unsqueeze(0).unsqueeze(0)
        return out


@ATTENTION_REGISTRY.register("samba_m3")
class SAMBAM3Block(nn.Module):
    """Sliding window attention + spectral Mamba."""

    def __init__(self, d_model, n_heads, window=128):
        super().__init__()
        self.window_size = window
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mamba = SpectralMamba(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def sliding_window_mask(self, L):
        idx = torch.arange(L)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        window = (idx.unsqueeze(1) - idx.unsqueeze(0)) < self.window_size
        mask = ~(causal & window)
        return mask.float().masked_fill(mask, float("-inf")).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, L, D = x.shape
        mask = self.sliding_window_mask(L).to(x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mamba(x))
        return x


@ATTENTION_REGISTRY.register("mamba")
def inject_mamba(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    d_model = model_config.hidden_size
    new_module = Mamba(d_model=d_model)
    layer.attention = new_module
    return new_module
