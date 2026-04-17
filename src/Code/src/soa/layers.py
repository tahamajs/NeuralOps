import torch.nn as nn
from .attention import SOALatentAttention
from .config import SOAConfig

class SOABlock(nn.Module):
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.attn = SOALatentAttention(cfg)
        self.ln1 = nn.LayerNorm(cfg.k_rank, eps=cfg.layer_norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(cfg.k_rank, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.k_rank),
            nn.Dropout(cfg.dropout)
        )
        self.ln2 = nn.LayerNorm(cfg.k_rank, eps=cfg.layer_norm_eps)

    def forward(self, x, attn_mask=None):
        attn_out = self.attn(x, attn_mask=attn_mask)
        x = self.ln1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x
