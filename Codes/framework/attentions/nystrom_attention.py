"""
Nyströmformer Attention: linear‑time attention via Nyström approximation.
Paper: https://arxiv.org/abs/2102.03902
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


def _compute_moore_penrose(A, iters=6):
    # A: (B, H, m, m)
    # Initialize X = (1/||A||) A^T (rough approximation)
    norm_A = torch.linalg.norm(A, dim=(-2, -1), keepdim=True)
    X = A.transpose(-2, -1) / (norm_A + 1e-8)
    I = torch.eye(A.size(-1), device=A.device).unsqueeze(0).unsqueeze(0)
    for _ in range(iters):
        X = torch.matmul(X, 2 * I - torch.matmul(A, X))
    return X


@ATTENTION_REGISTRY.register("nystrom")
class NystromAttention(BaseAttention):
    def __init__(self, config, rank=64, pinv_iterations=6, residual=True):
        super().__init__(config, rank)
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank
        self.pinv_iterations = pinv_iterations
        self.residual = residual

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _reshape(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states),   batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        num_landmarks = self.rank
        # Landmark selection (segmented mean)
        if seq_len % num_landmarks == 0:
            seg = seq_len // num_landmarks
            Q_land = Q.reshape(batch_size, self.num_heads, num_landmarks, seg, self.head_dim).mean(dim=3)
            K_land = K.reshape(batch_size, self.num_heads, num_landmarks, seg, self.head_dim).mean(dim=3)
        else:
            idx = torch.linspace(0, seq_len-1, num_landmarks).long().to(hidden_states.device)
            Q_land = Q[:, :, idx, :]
            K_land = K[:, :, idx, :]

        scale = 1.0 / (self.head_dim ** 0.5)
        F1 = torch.matmul(Q, K_land.transpose(-2, -1)) * scale
        F2 = torch.matmul(Q_land, K_land.transpose(-2, -1)) * scale
        F3 = torch.matmul(Q_land, K.transpose(-2, -1)) * scale

        F2_pinv = _compute_moore_penrose(F2, self.pinv_iterations)
        attn = torch.matmul(F1, torch.matmul(F2_pinv, F3))
        if attention_mask is not None:
            attn = attn + attention_mask.unsqueeze(1)

        probs = F.softmax(attn, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V)
        if self.residual:
            context = context + V
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)
        return (self.out_proj(context), None) if kwargs.get('output_attentions') else (self.out_proj(context),)