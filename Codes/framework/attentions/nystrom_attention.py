# framework/attentions/specialized/nystrom_attention.py
"""
Nyströmformer: linear‑time attention via Nyström approximation.
Paper: https://arxiv.org/abs/2102.03902
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def _compute_moore_penrose(A: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Iterative Newton–Schulz pseudoinverse."""
    norm_A = torch.linalg.norm(A, dim=(-2, -1), keepdim=True)
    X = A.transpose(-2, -1) / (norm_A + 1e-8)
    I = torch.eye(A.size(-1), device=A.device).unsqueeze(0).unsqueeze(0)
    for _ in range(iters):
        X = torch.matmul(X, 2 * I - torch.matmul(A, X))
    return X


@ATTENTION_REGISTRY.register("nystrom")
class NystromAttention(BaseAttention):
    """
    Nyströmformer Attention with landmark selection (segmented mean or uniform).
    """

    def __init__(self, config, rank: int = 64, pinv_iterations: int = 6,
                 residual: bool = True, pretrained_attn: nn.Module = None):
        if rank is None:
            rank = 64
        super().__init__(config, rank)
        self.pinv_iterations = pinv_iterations
        self.residual = residual

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.rank = rank

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src):
        with torch.no_grad():
            self.query.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.query.bias.copy_(src.q_lin.bias)
            self.key.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.key.bias.copy_(src.k_lin.bias)
            self.value.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.value.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def _reshape(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), batch_size)   # (B, H, N, Dh)
        K = self._reshape(self.key(hidden_states),   batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)

        # Landmark selection
        num_landmarks = self.rank
        if seq_len % num_landmarks == 0:
            seg = seq_len // num_landmarks
            Q_land = Q.reshape(batch_size, self.num_heads, num_landmarks, seg, self.head_dim).mean(dim=3)
            K_land = K.reshape(batch_size, self.num_heads, num_landmarks, seg, self.head_dim).mean(dim=3)
        else:
            idx = torch.linspace(0, seq_len-1, num_landmarks, dtype=torch.long, device=hidden_states.device)
            Q_land = Q[:, :, idx, :]
            K_land = K[:, :, idx, :]

        scale = 1.0 / (self.head_dim ** 0.5)
        F1 = torch.matmul(Q, K_land.transpose(-2, -1)) * scale
        F2 = torch.matmul(Q_land, K_land.transpose(-2, -1)) * scale
        F3 = torch.matmul(Q_land, K.transpose(-2, -1)) * scale

        F2_pinv = _compute_moore_penrose(F2, self.pinv_iterations)
        attn = torch.matmul(F1, torch.matmul(F2_pinv, F3))   # (B, H, N, N)

        # Mask handling
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (B, N) -> additive mask
                mask = attention_mask[:, None, None, :].to(dtype=attn.dtype)
                attn = attn + (1.0 - mask) * -1e9
            else:
                attn = attn + attention_mask   # assume additive

        probs = F.softmax(attn, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V)
        if self.residual:
            context = context + V
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.all_head_size)
        out = self.out_proj(context)

        if output_attentions:
            return out, attn
        return out, None


def inject_nystrom(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rank = getattr(comp_config, 'rank', 64)
    residual = getattr(comp_config, 'residual', True)
    new_attn = NystromAttention(config=model_config, rank=rank, residual=residual, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn