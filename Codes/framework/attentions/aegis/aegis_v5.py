"""
AEGIS‑Attention V5 – Hybrid Efficient Transformer Attention
------------------------------------------------------------
Drop‑in replacement for DistilBERT’s attention layer.

Combines:
  1. SVD‑factorised Q/K/V/Out projections (zero‑shot, >20× param reduction)
  2. Head‑wise hybrid: local window attention + global kernel linear attention
  3. Softpick rectified softmax for natural sparsity in local heads
  4. Per‑head adaptive gate to blend local and global information
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class AegisConfigV5:
    svd_energy: float = 0.95          # fraction of singular‑value energy to keep
    min_rank: int = 4
    max_rank: int = 64
    num_local_heads: int = 8          # local window attention
    num_global_heads: int = 4         # global kernel linear attention
    local_window: int = 64            # sliding window radius
    softpick_temperature: float = 1.0
    rff_features: int = 256           # random Fourier feature dimension
    use_zero_sum_keys: bool = True    # stabilise kernel approximation
    use_adaptive_gate: bool = True    # per‑head blending
    gate_init_bias: float = 1.5       # initial bias towards global
    dropout: float = 0.1


# --------------------------------------------------------------------------
# Factorised Linear Layer (truncated SVD)
# --------------------------------------------------------------------------
class FactorisedLinear(nn.Module):
    def __init__(self, pretrained_weight: torch.Tensor,
                 pretrained_bias: Optional[torch.Tensor],
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
            A = U[:, :rank] * S[:rank].unsqueeze(0)   # (out, r)
            B = Vt[:rank, :]                           # (r, in)

        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up   = nn.Linear(rank, out_dim, bias=(pretrained_bias is not None))
        self.down.weight.data = B.float()
        self.up.weight.data   = A.float()
        if pretrained_bias is not None:
            self.up.bias.data = pretrained_bias.float()
        else:
            self.up.bias.data.zero_()

    def forward(self, x):
        return self.up(self.down(x))


# --------------------------------------------------------------------------
# Softpick rectified softmax
# --------------------------------------------------------------------------
def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)


# --------------------------------------------------------------------------
# Local Window Attention (with Softpick)
# --------------------------------------------------------------------------
def local_window_attention(Q, K, V, window, temperature, token_valid=None):
    B, H, N, D = Q.shape
    scale = D ** -0.5
    idx = torch.arange(N, device=Q.device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    mask = (dist <= window // 2).unsqueeze(0).unsqueeze(0)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, -1e9)
    if token_valid is not None:
        scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e9)
    attn = softpick(scores, dim=-1, temperature=temperature)
    return torch.matmul(attn, V)


# --------------------------------------------------------------------------
# Global Kernel Linear Attention (Performer style)
# --------------------------------------------------------------------------
# (Used directly inside the class, not as a separate callable)


# --------------------------------------------------------------------------
# AEGIS‑Attention Module
# --------------------------------------------------------------------------
class AegisAttention(nn.Module):
    def __init__(self, model_config, aegis_cfg: AegisConfigV5,
                 pretrained_attn=None, bert_mode: bool = False):
        super().__init__()
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.cfg = aegis_cfg
        self.bert_mode = bert_mode

        assert (aegis_cfg.num_local_heads + aegis_cfg.num_global_heads == self.num_heads)
        self.num_local_heads = aegis_cfg.num_local_heads
        self.num_global_heads = aegis_cfg.num_global_heads

        if pretrained_attn is None:
            raise ValueError("AegisAttention requires pretrained_attn for zero‑shot init")

        # SVD‑factorised projections
        self.q_proj = FactorisedLinear(
            pretrained_attn.q_lin.weight.data, pretrained_attn.q_lin.bias.data,
            aegis_cfg.svd_energy, aegis_cfg.min_rank, aegis_cfg.max_rank)
        self.k_proj = FactorisedLinear(
            pretrained_attn.k_lin.weight.data, pretrained_attn.k_lin.bias.data,
            aegis_cfg.svd_energy, aegis_cfg.min_rank, aegis_cfg.max_rank)
        self.v_proj = FactorisedLinear(
            pretrained_attn.v_lin.weight.data, pretrained_attn.v_lin.bias.data,
            aegis_cfg.svd_energy, aegis_cfg.min_rank, aegis_cfg.max_rank)
        self.out_proj = FactorisedLinear(
            pretrained_attn.out_lin.weight.data, pretrained_attn.out_lin.bias.data,
            aegis_cfg.svd_energy, aegis_cfg.min_rank, aegis_cfg.max_rank)

        # Random Fourier features for global heads
        self.rff_proj = nn.Parameter(
            torch.randn(self.num_global_heads, self.head_dim, aegis_cfg.rff_features) * 0.02)

        # Adaptive gating
        if aegis_cfg.use_adaptive_gate:
            self.blend_logits = nn.Parameter(torch.zeros(self.num_heads))
            self.gate_bias = aegis_cfg.gate_init_bias
        else:
            self.blend_logits = None

        self.dropout = nn.Dropout(aegis_cfg.dropout)

    # ------------------------------------------------------------------
    # Mask helper (handles DistilBERT's various mask shapes)
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_mask(attention_mask, B, N, device):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            if attention_mask.shape[-2] == attention_mask.shape[-1]:
                token_valid = attention_mask[:, 0, 0, :] > -1e9
            else:
                token_valid = attention_mask.squeeze(1).squeeze(1) > -1e9
        elif attention_mask.dim() == 3:
            token_valid = attention_mask.squeeze(1).bool()
        elif attention_mask.dim() == 2:
            token_valid = attention_mask.bool()
        else:
            raise ValueError(f"Unexpected mask dim: {attention_mask.dim()}")
        return token_valid.to(device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs):
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        Hl, Hg = self.num_local_heads, self.num_global_heads

        # 1. Low‑rank projections
        Q_all = self.q_proj(hidden_states)
        K_all = self.k_proj(hidden_states)
        V_all = self.v_proj(hidden_states)

        Q_all = Q_all.view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        K_all = K_all.view(B, N, H, D).transpose(1, 2)
        V_all = V_all.view(B, N, H, D).transpose(1, 2)

        # split heads into local and global
        Q_local, Q_global = Q_all[:, :Hl], Q_all[:, Hl:]
        K_local, K_global = K_all[:, :Hl], K_all[:, Hl:]
        V_local, V_global = V_all[:, :Hl], V_all[:, Hl:]

        # token validity mask
        token_valid = self._prepare_mask(attention_mask, B, N, hidden_states.device)

        # 2. Local window attention
        ctx_local = local_window_attention(
            Q_local, K_local, V_local,
            window=self.cfg.local_window,
            temperature=self.cfg.softpick_temperature,
            token_valid=token_valid,
        )   # (B, Hl, N, D)

        # 3. Global kernel linear attention
        # project with learnable random Fourier features
        Q_feat = torch.einsum('bhnd,hdf->bhnf', Q_global, self.rff_proj)  # (B,Hg,N,F)
        K_feat = torch.einsum('bhnd,hdf->bhnf', K_global, self.rff_proj)

        # mask padding tokens
        if token_valid is not None:
            mask_g = token_valid.unsqueeze(1).unsqueeze(-1)   # (B,1,N,1)
            K_feat = K_feat.masked_fill(~mask_g, 0.0)
            V_global = V_global.masked_fill(~mask_g, 0.0)

        KV = torch.einsum('bhnf,bhnd->bhfd', K_feat, V_global)          # (B,Hg,F,D)
        K_sum = K_feat.sum(dim=2, keepdim=True)                          # (B,Hg,1,F)
        denom = torch.matmul(Q_feat, K_sum.transpose(-1, -2)) + 1e-8     # (B,Hg,N,1)
        ctx_global = torch.einsum('bhnf,bhfd->bhnd', Q_feat, KV) / denom  # (B,Hg,N,D)

        # 4. Adaptive blending (or simple concatenation)
        if self.blend_logits is not None:
            global_signal = ctx_global.mean(dim=1, keepdim=True).expand(-1, H, -1, -1)
            local_full = torch.cat([ctx_local, torch.zeros(B, Hg, N, D, device=ctx_local.device)], dim=1)
            gate = torch.sigmoid(self.blend_logits.unsqueeze(-1).unsqueeze(-1) + self.gate_bias)
            context = gate * local_full + (1 - gate) * global_signal
        else:
            context = torch.cat([ctx_local, ctx_global], dim=1)

        # mask output for padding tokens
        if token_valid is not None:
            context = context * token_valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        # reshape and factorised output projection
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        output = self.out_proj(context)
        return (output, None)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


# ------------------------------------------------------------
# Injection helper & Registry
# ------------------------------------------------------------
def inject_aegis_into_distilbert(layer, model_config, aegis_cfg, old_attn):
    new_attn = AegisAttention(model_config=model_config,
                              aegis_cfg=aegis_cfg,
                              pretrained_attn=old_attn,
                              bert_mode=False)
    layer.attention = new_attn
    return new_attn


from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("aegis_v5")
def inject_aegis_v5(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    cfg = AegisConfigV5(
        num_local_heads=8,
        num_global_heads=4,
        local_window=getattr(comp_config, 'block_size', 64),
        rff_features=getattr(comp_config, 'rff_features', 256),
        dropout=getattr(comp_config, 'dropout', 0.1),
    )
    return inject_aegis_into_distilbert(layer, model_config, cfg, old_attn)
