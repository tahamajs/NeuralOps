"""
SOA Fused Attention: low‑rank latent space attention with zero‑shot init.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from framework.core.registry import ATTENTION_REGISTRY


class SOAFusedAttention(nn.Module):
    """
    SOA Fused: separate latent spaces for Q/K/V, full output projection.
    """

    def __init__(self, model_config, k_rank: int = 32, dropout: float = 0.1,
                 pretrained_attn=None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.hidden_size = model_config.hidden_size
        self.k_rank = k_rank

        self.q_down = nn.Linear(self.hidden_size, k_rank, bias=False)
        self.k_down = nn.Linear(self.hidden_size, k_rank, bias=False)
        self.v_down = nn.Linear(self.hidden_size, k_rank, bias=False)

        self.q_up = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.k_up = nn.Linear(k_rank, self.hidden_size, bias=False)
        self.v_up = nn.Linear(k_rank, self.hidden_size, bias=False)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.dropout = nn.Dropout(dropout)
        self._init_from_pretrained(pretrained_attn)

    def _init_from_pretrained(self, pretrained_attn):
        with torch.no_grad():
            for orig, down, up in [
                (pretrained_attn.q_lin, self.q_down, self.q_up),
                (pretrained_attn.k_lin, self.k_down, self.k_up),
                (pretrained_attn.v_lin, self.v_down, self.v_up),
            ]:
                W = orig.weight.data.float()
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(self.k_rank, U.size(1), Vt.size(0))
                down.weight.data[:r, :] = Vt[:r, :].to(down.weight.dtype)
                up.weight.data[:, :r] = (U[:, :r] * S[:r]).to(up.weight.dtype)

            self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight)
            if pretrained_attn.out_lin.bias is not None:
                self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape

        z_q = self.q_down(hidden_states)
        z_k = self.k_down(hidden_states)
        z_v = self.v_down(hidden_states)

        q = self.q_up(z_q)
        k = self.k_up(z_k)
        v = self.v_up(z_v)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            scores = scores + (1.0 - mask) * -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)

        output = self.out_proj(context)

        if output_attentions:
            return output, attn_weights
        return output, None


@ATTENTION_REGISTRY.register("soa_fused_attn")
def inject_soa_fused_attn(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    k_rank = getattr(comp_config, 'soa_rank', 32)
    dropout = getattr(comp_config, 'dropout', 0.1)
    new_attn = SOAFusedAttention(model_config, k_rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn