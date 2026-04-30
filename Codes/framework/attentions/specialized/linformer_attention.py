import torch
import torch.nn as nn
import math


class LinformerAttention(nn.Module):
    """
    Linformer: Self-Attention with Linear Complexity.
    Paper: https://arxiv.org/abs/2006.04768
    Projects the key and value matrices to a fixed low‑dimensional space (seq_len -> k).
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Low‑rank projection dimension (fixed length, e.g. 128)
        self.k_dim = getattr(config, "linformer_k", 128)
        self.proj_k = nn.Linear(self.k_dim, config.hidden_size, bias=False)  # actually E·K
        self.proj_v = nn.Linear(self.k_dim, config.hidden_size, bias=False)  # actually F·V
        # We'll use a learned projection matrix E, F
        self.E = nn.Parameter(torch.randn(self.k_dim, config.max_position_embeddings) * 0.02)
        self.F = nn.Parameter(torch.randn(self.k_dim, config.max_position_embeddings) * 0.02)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        B, L, D = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply low‑rank projections (E and F) to reduce sequence length
        # k_proj = E @ K, shape (B, H, k_dim, d)
        E_eff = self.E[:, :L]  # (k_dim, L)
        F_eff = self.F[:, :L]
        k_proj = torch.einsum("kl,bhld->bhkd", E_eff, k)   # (B, H, k_dim, d)
        v_proj = torch.einsum("kl,bhld->bhkd", F_eff, v)

        # Attention: Q @ K_proj^T
        attn = torch.matmul(q, k_proj.transpose(-2, -1)) * self.scaling  # (B, H, L, k_dim)
        if attention_mask is not None:
            # Linformer typically ignores mask on the projected side,
            # but we can apply the mask to the Q side.
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                mask = attention_mask[:, None, :, :]
            else:
                mask = attention_mask
            attn = attn + (1.0 - mask.to(torch.float32)) * -1e9[:, :, :, None]   # need to adjust dimensions
        attn_weights = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn_weights, v_proj)  # (B, H, L, d)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if output_attentions:
            return (out, attn_weights)
        return out