import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralMamba(nn.Module):
    """Mamba with spectral normalisation (Simba‑inspired stability)."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(d_model=d_model)   # from jamba2 stub or official
        self.dt_scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        out = self.mamba(x)
        out = out * torch.sigmoid(self.dt_scale).unsqueeze(0).unsqueeze(0)
        return out

class SAMBAM3Block(nn.Module):
    """One SAMBA‑M3 layer: sliding window attention + spectral Mamba."""
    def __init__(self, d_model, n_heads, window=128, chunk_size=64):
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
        mask = ~(causal & window)   # True where to mask
        # Convert to additive mask: 0 for allowed, -inf for masked
        return mask.float().masked_fill(mask, float('-inf')).unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B, L, D = x.shape
        mask = self.sliding_window_mask(L).to(x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        mamba_out = self.mamba(x)
        x = self.norm2(x + mamba_out)
        return x