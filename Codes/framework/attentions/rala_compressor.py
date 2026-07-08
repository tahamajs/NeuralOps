import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("rala")
class RALALinearAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.rank_r = getattr(config, "rala_rank", 8)
        self.U = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.rank_r) * 0.02
        )
        self.V = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.rank_r) * 0.02
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.activation = nn.GELU()  # fixed

    def forward(
        self, hidden_states, attention_mask=None, output_attentions=False, **kwargs
    ):
        batch_size, seq_len, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        q_prime = F.elu(q) + 1.0
        k_prime = F.elu(k) + 1.0
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            else:
                mask = attention_mask
            k_prime = k_prime + mask.unsqueeze(-1)

        kv = torch.einsum("b h n d, b h n e -> b h d e", k_prime, v)
        z = k_prime.sum(dim=2)
        lin_out = torch.einsum("b h n d, b h d e -> b h n e", q_prime, kv) / (
            torch.einsum("b h n d, b h d -> b h n", q_prime, z).unsqueeze(-1) + 1e-8
        )

        rank_mat = torch.einsum("h d r, h e r -> h d e", self.U, self.V)
        aug_out = torch.einsum("b h n d, h d e -> b h n e", q_prime, rank_mat) / (
            torch.einsum("b h n d, b h d -> b h n", q_prime, z).unsqueeze(-1) + 1e-8
        )

        output = lin_out + self.alpha * aug_out
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        output = self.out_proj(output)
        return output, None
