import torch
import torch.nn as nn
import math

class SOAFusedAttention(nn.Module):
    """
    Shared Orthogonal Attention (SOA) mechanism.
    Applies Q/K/V projections from a low-rank latent space defined by a shared basis U.
    The formula is: W_orig ≈ U @ A, where A is this module's weight.
    """
    def __init__(self, hidden_size: int, num_heads: int, k_rank: int):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_rank = k_rank

        # These 'A' matrices project from latent space (k_rank) to Q/K/V space (hidden_size)
        self.q_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.k_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.v_proj = nn.Linear(k_rank, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x_latent: torch.Tensor, attn_mask: torch.Tensor = None):
        """
        Args:
            x_latent (torch.Tensor): Input in latent space, shape (B, S, k_rank)
            attn_mask (torch.Tensor, optional): Attention mask, shape (B, 1, 1, S)
        """
        B, S, _ = x_latent.shape

        # Project from latent space (k_rank) to Q, K, V (hidden_size)
        q = self.q_proj(x_latent)  # (B, S, H)
        k = self.k_proj(x_latent)  # (B, S, H)
        v = self.v_proj(x_latent)  # (B, S, H)

        # Reshape for multi-head attention
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, S, hd)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, S, hd)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, S, hd)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        ctx = torch.matmul(attn_weights, v) # (B, nh, S, hd)

        # Re-combine heads
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)

        # Final output projection
        return self.o_proj(ctx)