"""
Rank‑Augmented Linear Attention (RALA) – ICLR 2024.
Paper: https://arxiv.org/abs/2311.17374
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


@ATTENTION_REGISTRY.register("rala")
class RALALinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"hidden_size must be divisible by num_attention_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.rank_r = getattr(config, "rala_rank", 8)
        self.U = nn.Parameter(torch.randn(self.num_heads, self.head_dim, self.rank_r) * 0.02)
        self.V = nn.Parameter(torch.randn(self.num_heads, self.head_dim, self.rank_r) * 0.02)

        # Learnable alpha to balance the augmentation
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.activation = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )

    @staticmethod
    def _apply_linear_mask(k, attention_mask):
        if attention_mask is None:
            return k
        if attention_mask.dim() == 2:      # (B, L)
            mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:    # (B, 1, L)
            mask = attention_mask[:, None, :, :]
        else:
            mask = attention_mask
        mask = (1.0 - mask.to(torch.float32)).clamp(min=0) * -1e9
        return k + mask.to(k.dtype)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        q_prime = F.elu(q) + 1.0
        k_prime = self._apply_linear_mask(F.elu(k) + 1.0, attention_mask)

        kv = torch.einsum("b h n d, b h n e -> b h d e", k_prime, v)
        z = k_prime.sum(dim=2)  # (B, H, D)

        # Linear attention output
        lin_out = (
            torch.einsum("b h n d, b h d e -> b h n e", q_prime, kv)
            / (torch.einsum("b h n d, b h d -> b h n", q_prime, z).unsqueeze(-1) + 1e-8)
        )

        # Rank augmentation: (U V^T) per head
        rank_mat = torch.einsum("h d r, h e r -> h d e", self.U, self.V)   # (H, D, E)
        aug_out = (
            torch.einsum("b h n d, h d e -> b h n e", q_prime, rank_mat)
            / (torch.einsum("b h n d, b h d -> b h n", q_prime, z).unsqueeze(-1) + 1e-8)
        )

        output = lin_out + self.alpha * aug_out
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)

        if output_attentions:
            return (output, None)
        return output