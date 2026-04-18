import torch.nn as nn
from .attention import SOAFusedAttention
from .config import SOAConfig

class SOABlock(nn.Module):
    def __init__(self, cfg: SOAConfig, shared_basis: nn.Parameter):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        
        # This is a fixed, shared parameter, not trained
        self.register_buffer("basis", shared_basis) # Shape: (H, R)
        
        self.attn = SOAFusedAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.k_rank)
        
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        # 1. Pre-LayerNorm
        h = self.ln1(x)
        
        # 2. Reduce to shared latent space: x_latent = h @ U  (B, S, H) @ (H, R) -> (B, S, R)
        x_latent = h @ self.basis 
        
        # 3. Attention in the latent space
        attn_out = self.attn(x_latent, attn_mask)
        
        # 4. Residual connection
        x = x + attn_out
        
        # 5. FFN block with residual connection
        h2 = self.ln2(x)
        x = x + self.ffn(h2)
        
        return x