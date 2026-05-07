"""
ORION — Optimized Rectified Input-aware Orthogonal Network
Hybrid attention: SVD‑factorised projections + head‑wise kernel/full attention + Softpick.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from framework.core.registry import ATTENTION_REGISTRY


@dataclass
class ORIONConfig:
    svd_energy: float = 0.95
    svd_min_rank: int = 4
    svd_max_rank: int = 64
    num_kernel_heads: int = 8
    num_full_heads: int = 4
    rff_features: int = 256
    softpick_temperature: float = 1.0
    dropout: float = 0.1


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)


class FactorisedLinear(nn.Module):
    def __init__(self, pretrained_weight: torch.Tensor, pretrained_bias: Optional[torch.Tensor],
                 energy: float, min_rank: int = 4, max_rank: int = 64):
        super().__init__()
        out_dim, in_dim = pretrained_weight.shape
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(pretrained_weight.float(), full_matrices=False)
            total = (S ** 2).sum()
            cum = torch.cumsum(S ** 2, dim=0) / total
            rank = int(torch.searchsorted(cum, energy).item()) + 1
            rank = max(min_rank, min(rank, min(out_dim, in_dim), max_rank))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class ORIONAttention(nn.Module):
    def __init__(self, model_config, orion_cfg: ORIONConfig, pretrained_attn=None, bert_mode: bool = False):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("ORION requires pretrained_attn for zero‑shot SVD init")
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.cfg = orion_cfg
        self.bert_mode = bert_mode

        if orion_cfg.num_kernel_heads + orion_cfg.num_full_heads != self.num_heads:
            raise ValueError(
                f"num_kernel_heads ({orion_cfg.num_kernel_heads}) + "
                f"num_full_heads ({orion_cfg.num_full_heads}) must equal "
                f"total heads ({self.num_heads})"
            )
        self.num_kernel_heads = orion_cfg.num_kernel_heads
        self.num_full_heads = orion_cfg.num_full_heads

        self.q_proj = FactorisedLinear(
            pretrained_attn.q_lin.weight.data, pretrained_attn.q_lin.bias.data,
            orion_cfg.svd_energy, orion_cfg.svd_min_rank, orion_cfg.svd_max_rank
        )
        self.k_proj = FactorisedLinear(
            pretrained_attn.k_lin.weight.data, pretrained_attn.k_lin.bias.data,
            orion_cfg.svd_energy, orion_cfg.svd_min_rank, orion_cfg.svd_max_rank
        )
        self.v_proj = FactorisedLinear(
            pretrained_attn.v_lin.weight.data, pretrained_attn.v_lin.bias.data,
            orion_cfg.svd_energy, orion_cfg.svd_min_rank, orion_cfg.svd_max_rank
        )
        self.out_proj = FactorisedLinear(
            pretrained_attn.out_lin.weight.data, pretrained_attn.out_lin.bias.data,
            orion_cfg.svd_energy, orion_cfg.svd_min_rank, orion_cfg.svd_max_rank
        )
        self.rff_proj = nn.Parameter(
            torch.randn(self.num_kernel_heads, self.head_dim, orion_cfg.rff_features) * 0.02
        )
        self.dropout = nn.Dropout(orion_cfg.dropout)

    def _kernel_linear_attention(self, Q, K, V):
        B, Hk, N, D = Q.shape
        F_dim = self.cfg.rff_features
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        proj = self.rff_proj.to(Q.device)
        Q_proj = torch.einsum('bhnd,hdf->bhnf', Q_norm, proj)
        K_proj = torch.einsum('bhnd,hdf->bhnf', K_norm, proj)
        Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
        K_feat = torch.exp(K_proj) / math.sqrt(F_dim)
        KV = torch.einsum('bhnf,bhnd->bhfd', K_feat, V)
        K_sum = K_feat.sum(dim=2)
        denom = torch.einsum('bhnf,bhf->bhn', Q_feat, K_sum).unsqueeze(-1)
        ctx = torch.einsum('bhnf,bhfd->bhnd', Q_feat, KV) / (denom + 1e-8)
        return ctx

    def _full_softpick_attention(self, Q, K, V, token_valid=None):
        scale = self.head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if token_valid is not None:
            scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e5)
        attn_w = softpick(scores, dim=-1, temperature=self.cfg.softpick_temperature)
        attn_w = self.dropout(attn_w)
        return torch.matmul(attn_w, V)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        B, N, _ = hidden_states.shape
        Hk = self.num_kernel_heads
        Hf = self.num_full_heads

        Q_all = self.q_proj(hidden_states)
        K_all = self.k_proj(hidden_states)
        V_all = self.v_proj(hidden_states)
        Q_all = Q_all.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K_all = K_all.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V_all = V_all.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q_kernel, Q_full = Q_all[:, :Hk], Q_all[:, Hk:]
        K_kernel, K_full = K_all[:, :Hk], K_all[:, Hk:]
        V_kernel, V_full = V_all[:, :Hk], V_all[:, Hk:]

        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        token_valid = None
        if mask is not None:
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                token_valid = mask[:, 0, 0, :] > -1e9
            elif mask.dim() == 4:
                token_valid = mask.squeeze(1).squeeze(1) > -1e9
            elif mask.dim() == 3:
                token_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                token_valid = mask.bool()

        ctx_kernel = self._kernel_linear_attention(Q_kernel, K_kernel, V_kernel)
        ctx_full = self._full_softpick_attention(Q_full, K_full, V_full, token_valid)

        context = torch.cat([ctx_kernel, ctx_full], dim=1)
        if token_valid is not None:
            context = context * token_valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(context)
        return output, None

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("orion")
def inject_orion(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    cfg = ORIONConfig(
        svd_energy=getattr(comp_config, 'svd_energy', 0.95),
        svd_min_rank=getattr(comp_config, 'min_rank', 4),
        svd_max_rank=getattr(comp_config, 'max_rank', 64),
        num_kernel_heads=getattr(comp_config, 'num_kernel_heads', 8),
        num_full_heads=getattr(comp_config, 'num_full_heads', 4),
        rff_features=getattr(comp_config, 'rff_features', 256),
        softpick_temperature=getattr(comp_config, 'softpick_temperature', 1.0),
        dropout=getattr(comp_config, 'dropout', 0.1),
    )
    new_attn = ORIONAttention(model_config, cfg, pretrained_attn=old_attn)
    layer.attention = new_attn
    return new_attn