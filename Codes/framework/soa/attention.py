# framework/soa/attention.py
"""
SOA Fused Attention - Low-rank attention using shared basis.
"""

import torch
import torch.nn as nn
import math


class SOAFusedAttention(nn.Module):
    """
    SOA Fused Attention - Projects from low-rank latent space to full attention.
    
    This attention mechanism takes a low-rank latent representation and projects
    it to Q, K, V using learned projections, then computes standard attention.
    
    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        k_rank: Rank of the latent space
        dropout: Dropout probability
        
    Example:
        >>> attn = SOAFusedAttention(768, 12, 64)
        >>> x_latent = torch.randn(4, 128, 64)  # (batch, seq, rank)
        >>> output = attn(x_latent)
    """
    
    def __init__(self, hidden_size: int, num_heads: int, k_rank: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_rank = k_rank

        # Projections from latent space to Q, K, V
        self.q_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.k_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.v_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_latent: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of SOA fused attention.
        
        Args:
            x_latent: Input tensor of shape (batch_size, seq_len, k_rank)
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        B, S, _ = x_latent.shape
        
        # Project to Q, K, V in full hidden space
        q = self.q_proj(x_latent).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_latent).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_latent).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            scores = scores + attn_mask
            
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.hidden_size)
        
        return self.o_proj(out)
    
    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, k_rank={self.k_rank}"