# framework/soa/block.py
"""
SOA Transformer Block - Pre-normalization with SVD-optimized attention.
"""

import torch
import torch.nn as nn
from .attention import SOAFusedAttention


class SOABlock(nn.Module):
    """
    SOA Transformer Block with pre-normalization.
    
    Architecture: LN -> Attention -> Residual -> LN -> FFN -> Residual
    
    Args:
        cfg: SOAConfig instance
        shared_basis: Shared basis tensor for latent projection
    """
    
    def __init__(self, cfg, shared_basis: torch.Tensor):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.basis = shared_basis  # (hidden_size, k_rank)
        self.attn = SOAFusedAttention(
            cfg.hidden_size, 
            cfg.num_attention_heads, 
            cfg.k_rank,
            dropout=cfg.dropout_prob
        )
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout_prob),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
            nn.Dropout(cfg.dropout_prob),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through SOA block.
        
        Args:
            x: Input tensor (batch, seq, hidden_size)
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq, hidden_size)
        """
        # Pre-norm for attention
        h = self.ln1(x)
        # Project to latent space using shared basis
        x_latent = h @ self.basis  # (batch, seq, k_rank)
        # Apply attention in latent space
        attn_out = self.attn(x_latent, attn_mask)
        # Residual connection
        x = x + attn_out
        
        # Pre-norm for FFN
        x = x + self.ffn(self.ln2(x))
        
        return x