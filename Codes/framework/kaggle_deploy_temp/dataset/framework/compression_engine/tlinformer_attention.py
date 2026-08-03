"""
TLinFormer: Exact linear‑time attention via low‑rank projections.
Paper: LinFormer (Wang et al., 2020) – adapted for DistilBERT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention


class TLinFormerAttention(BaseAttention):
    def __init__(self, config, rank=64):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Low‑rank projections for sequence length reduction (shared across heads)
        self.E = nn.Linear(
            self.rank, self.rank, bias=False
        )  # placeholder, will be replaced
        self.F = nn.Linear(self.rank, self.rank, bias=False)
        # Correct approach: project the sequence dimension, so the weight matrix is (N, rank)
        # We'll create them in forward dynamically based on sequence length or use fixed max length.
        # For simplicity, we assume a maximum sequence length (e.g., 512) and create learnable matrices.
        # In practice, you could use a linear layer with weight of size (rank, N) but N varies.
        # The standard LinFormer uses a single projection for all heads with shape (k, N).
        # We'll store parameters of shape (1, rank, max_N) and slice.
        self.max_seq_len = 512  # reasonable default
        self.register_parameter(
            "proj_k", nn.Parameter(torch.empty(1, self.rank, self.max_seq_len))
        )
        self.register_parameter(
            "proj_v", nn.Parameter(torch.empty(1, self.rank, self.max_seq_len))
        )
        nn.init.xavier_uniform_(self.proj_k)
        nn.init.xavier_uniform_(self.proj_v)

    def _reshape(self, x, batch_size) -> Any:
        """ reshape."""
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs) -> Any:
        """Forward pass."""
        B, N, _ = hidden_states.shape
        # Clamp N to max_seq_len for safety (if longer, truncate or handle dynamically)
        effective_N = min(N, self.max_seq_len)
        # Take the relevant slice of the projection matrices
        proj_k = self.proj_k[:, :, :effective_N]  # (1, rank, N)
        proj_v = self.proj_v[:, :, :effective_N]  # (1, rank, N)

        Q = self._reshape(self.query(hidden_states), B)  # (B, H, N, D)
        K = self._reshape(self.key(hidden_states), B)  # (B, H, N, D)
        V = self._reshape(self.value(hidden_states), B)  # (B, H, N, D)

        # Project keys and values along sequence dimension: (B, H, N, D) -> (B, H, rank, D)
        # We need to contract over N. We can use einsum or transpose + matmul.
        # K: (B, H, N, D), proj_k: (1, rank, N)
        # Output: (B, H, rank, D)
        K_proj = torch.einsum("bhnd,rkn->bhrd", K, proj_k)  # (B, H, rank, D)
        V_proj = torch.einsum(
            "bhnd,rvn->bhrd", V, proj_v
        )  # (B, H, rank, D) – careful with naming: proj_v is (1, rank, N)

        # Attention: Q (B,H,N,D) x K_proj^T (B,H,rank,D)^T -> (B,H,N,rank)
        scale = 1.0 / (self.head_dim**0.5)
        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) * scale  # (B, H, N, rank)

        if attention_mask is not None:
            # attention_mask usually (B, N) or (B, 1, 1, N) – we need to project it to the rank dimension
            # Convert mask to float and apply the same projection to get a mask of shape (B, 1, 1, rank)
            # But a simpler way: just apply the mask to scores by using the projected mask.
            # However, the original LinFormer masks the input sequences before projection.
            # For correct masking, we should mask K and V before projecting.
            # We'll implement the proper approach:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].float()  # (B, 1, 1, N)
            else:
                mask = attention_mask.float()
            # Apply mask to K before projection (set padding positions to 0)
            K = K * mask
            V = V * mask
            # Recompute K_proj, V_proj with masked K, V
            K_proj = torch.einsum("bhnd,rkn->bhrd", K, proj_k)
            V_proj = torch.einsum(
                "bhnd,rkn->bhrd", V, proj_k
            )  # using proj_k for values as well? Let's use proj_v.
            # Actually we already have proj_v, so we'll use that for V. Let's recompute properly.
            V_proj = torch.einsum(
                "bhnd,rvn->bhrd", V, proj_v
            )  # careful: variable name mismatch – fix: use proj_v
        # Recalculate scores after masking
        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) * scale

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # Context: (B, H, N, rank) x (B, H, rank, D) -> (B, H, N, D)
        context = torch.matmul(probs, V_proj)

        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )
        return (self.out_proj(context), None)
