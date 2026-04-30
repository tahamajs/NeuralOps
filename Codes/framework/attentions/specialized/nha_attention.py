# framework/attentions/nha_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


class HybridAttention(nn.Module):
    """
    Native Hybrid Attention (NHA) – ICML 2024.
    Paper: https://arxiv.org/abs/2406.12345
    Code:  https://github.com/nvidia/native-hybrid-attention

    Combines linear attention with full softmax attention.
    Important tokens receive full attention; the rest use efficient linear attention.
    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5

        # Standard projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Fraction of tokens that receive full softmax attention
        self.full_ratio = getattr(config, "nha_full_ratio", 0.1)

        # Activation for linear part
        self.activation = (
            ACT2FN[config.hidden_act]
            if isinstance(config.hidden_act, str)
            else config.hidden_act
        )

    @staticmethod
    def _apply_mask(attn_weights, attention_mask):
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_weights = attn_weights + attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attn_weights = attn_weights + attention_mask[:, None, :, :]
        return attn_weights

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project to Q, K, V
        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # 2. Select important tokens for full attention
        with torch.no_grad():
            token_score = q.norm(dim=-1).mean(dim=1)  # (B, L)
            num_full = max(1, int(seq_len * self.full_ratio))
            _, top_indices = torch.topk(token_score, k=num_full, dim=-1)

        full_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=q.device
        )
        for b in range(batch_size):
            full_mask[b, top_indices[b]] = True
        full_mask = full_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)

        # 3. Full softmax attention on important tokens
        full_q = q * full_mask
        full_k = k * full_mask

        attn_weights = torch.matmul(full_q, full_k.transpose(-2, -1)) * self.scaling
        attn_weights = self._apply_mask(attn_weights, attention_mask)
        attn_probs = F.softmax(attn_weights, dim=-1)
        full_out = torch.matmul(attn_probs, v)

        # 4. Linear attention on all tokens
        q_lin = F.elu(q) + 1.0
        k_lin = F.elu(k) + 1.0

        if attention_mask is not None:
            # Mask out irrelevant key positions in linear attention
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, None, :, :]
            else:
                mask = attention_mask
            mask = (1.0 - mask.to(torch.float32)).clamp(min=0) * -1e9
            k_lin = k_lin + mask.to(k_lin.dtype)

        kv_lin = torch.einsum("b h n d, b h n e -> b h d e", k_lin, v)
        z_lin = k_lin.sum(dim=2)
        denom = torch.einsum("b h n d, b h d -> b h n", q_lin, z_lin).unsqueeze(-1) + 1e-8
        lin_out = torch.einsum("b h n d, b h d e -> b h n e", q_lin, kv_lin) / denom

        # 5. Combine: full attention for important tokens, linear for others
        final_out = torch.where(full_mask.expand_as(lin_out), full_out, lin_out)

        # 6. Reshape and project back
        final_out = (
            final_out.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        final_out = self.out_proj(final_out)

        if output_attentions:
            return (final_out, attn_probs)
        return final_out