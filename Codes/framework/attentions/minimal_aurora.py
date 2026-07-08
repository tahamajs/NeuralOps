"""
MinimalAURORA – stable, zero‑shot, drop‑in replacement for DistilBERT.
SVD‑factorised Q/K/V/Out projections only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("minimal_aurora")
class FactorisedLinear(nn.Module):
    """SVD‑factorised linear layer."""

    def __init__(self, pretrained_weight, pretrained_bias, energy=0.95):
        super().__init__()
        out_dim, in_dim = pretrained_weight.shape
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(pretrained_weight.float(), full_matrices=False)
            total = (S**2).sum()
            cum = torch.cumsum(S**2, dim=0) / total
            rank = int(torch.searchsorted(cum, energy).item()) + 1
            rank = max(4, min(rank, min(out_dim, in_dim), 64))
            self.rank = rank
            A = U[:, :rank] * S[:rank].unsqueeze(0)
            B = Vt[:rank, :]
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=(pretrained_bias is not None))
        self.down.weight.data = B.float()
        self.up.weight.data = A.float()
        if pretrained_bias is not None:
            self.up.bias.data = pretrained_bias.float()
        else:
            self.up.bias.data.zero_()

    def forward(self, x):
        return self.up(self.down(x))


class MinimalAURORAAttention(nn.Module):
    def __init__(self, model_config, energy=0.95, dropout=0.1, pretrained_attn=None):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size

        self.q_proj = FactorisedLinear(
            pretrained_attn.q_lin.weight.data, pretrained_attn.q_lin.bias.data, energy
        )
        self.k_proj = FactorisedLinear(
            pretrained_attn.k_lin.weight.data, pretrained_attn.k_lin.bias.data, energy
        )
        self.v_proj = FactorisedLinear(
            pretrained_attn.v_lin.weight.data, pretrained_attn.v_lin.bias.data, energy
        )
        self.out_proj = FactorisedLinear(
            pretrained_attn.out_lin.weight.data,
            pretrained_attn.out_lin.bias.data,
            energy,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        Q = Q.view(B, N, H, D).transpose(1, 2)
        K = K.view(B, N, H, D).transpose(1, 2)
        V = V.view(B, N, H, D).transpose(1, 2)

        scale = D**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        mask = (
            attention_mask
            if attention_mask is not None
            else kwargs.get("attn_mask", None)
        )
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -1e9
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -1e9
            elif mask.dim() == 3:
                token_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                token_valid = mask.bool()
            else:
                token_valid = None
            if token_valid is not None:
                scores = scores.masked_fill(
                    ~token_valid.unsqueeze(1).unsqueeze(2), -1e5
                )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        output = self.out_proj(context)
        # Always return a 2‑tuple: (output, attn_weights) – second element is None if not requested
        if output_attentions:
            return output, attn_weights
        else:
            return output, None

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("minimal_aurora")
def inject_minimal_aurora(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    new_attn = MinimalAURORAAttention(
        model_config=model_config,
        energy=getattr(comp_config, "svd_energy", 0.95),
        dropout=getattr(comp_config, "dropout", 0.1),
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
