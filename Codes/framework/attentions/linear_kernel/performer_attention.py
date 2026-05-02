"""
Performer Attention: Fast Attention Via Positive Orthogonal Random features (FAVOR+)
Paper: https://arxiv.org/abs/2009.14794
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base import BaseAttention
from ..core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("performer")
class PerformerAttention(BaseAttention):
    """
    Performer approximates softmax attention using random features,
    reducing complexity to O(n).
    """
    def __init__(self, config, rank=256, redraw_projection=True):
        super().__init__(config, rank)   # rank is number of random features
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank
        self.redraw_projection = redraw_projection

        # Q, K, V projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Random projection matrix for features (one per head)
        self.register_buffer('projection', None)

    def _reshape(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)   # (B, H, N, D)

    def _get_projection_matrix(self, device):
        if self.projection is None or self.redraw_projection:
            # Sample orthogonal random matrix of shape (head_dim, rank)
            ortho = torch.randn(self.head_dim, self.rank, device=device)
            # QR decomposition to make orthogonal
            q, _ = torch.linalg.qr(ortho)
            self.projection = q
        return self.projection

    def _feature_map(self, x, proj):
        # x: (B, H, N, D)
        # proj: (D, R)
        # Compute features: φ(x) = 1/sqrt(R) * exp(x @ proj * scale)
        scale = 1.0 / (self.head_dim ** 0.25)   # typical scaling
        x_proj = torch.matmul(x, proj) * scale   # (B, H, N, R)
        # Positive random features (ReLU instead of exp for stability, as in some implementations)
        # The original paper uses exp, but ReLU works as well.
        phi = torch.relu(x_proj + 1e-6)
        # Normalize
        phi = phi / (phi.sum(dim=-1, keepdim=True) + 1e-8)
        return phi

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape

        Q = self._reshape(self.query(hidden_states), batch_size)   # (B, H, N, D)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        proj = self._get_projection_matrix(Q.device)  # (D, R)

        # Apply feature map to queries and keys
        Q_prime = self._feature_map(Q, proj)   # (B, H, N, R)
        K_prime = self._feature_map(K, proj)   # (B, H, N, R)

        # Attention via kernel trick: context = φ(Q) (φ(K)^T V) / normalization
        # Step 1: compute KV = φ(K)^T V
        KV = torch.matmul(K_prime.transpose(-2, -1), V)   # (B, H, R, D)

        # Step 2: normalize Q' * KV by sum of φ(Q) φ(K)^T
        # equivalent to row‑wise softmax normalization.
        # We compute numerator = Q' * KV, denominator = Q' * sum(φ(K)).
        num = torch.matmul(Q_prime, KV)   # (B, H, N, D)
        den = torch.matmul(Q_prime, K_prime.sum(dim=-2, keepdim=True).unsqueeze(-1))
        den = den + 1e-8

        context = num / den   # (B, H, N, D)

        if attention_mask is not None:
            # Mask out padding (simple multiplication)
            mask = attention_mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, seq_len, -1)
            context = context * mask.to(context.dtype)

        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)
        return (self.out_proj(context), None) if kwargs.get('output_attentions') else (self.out_proj(context),)