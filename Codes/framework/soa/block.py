import torch.nn as nn
from .attention import SOAFusedAttention

class SOABlock(nn.Module):
    def __init__(self, cfg, shared_basis):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.basis = shared_basis  # (H, R)
        self.attn = SOAFusedAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.k_rank)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        x_latent = h @ self.basis   # (B, S, R)
        attn_out = self.attn(x_latent, attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x