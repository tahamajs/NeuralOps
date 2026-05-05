"""
AEGIS‑Attention V1 – Adaptive Efficient Global‑Local Integrated Sparse Attention
CUR‑factorised projections, local block attention (Softpick), global kernel linear attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from framework.attentions.core.utils import softpick


@dataclass
class AEGISConfigV1:
    cur_rank: int = 16
    cur_method: str = "interpolative"
    use_tile_sharing: bool = True
    tile_size: int = 32
    num_shared_tiles: int = 64
    num_local_heads: int = 8
    num_global_heads: int = 4
    local_window: int = 64
    softpick_enabled: bool = True
    softpick_temperature: float = 1.0
    rff_features: int = 256
    use_adaptive_gate: bool = True
    gate_init_bias: float = 1.5
    dropout: float = 0.1


# ----------------------------------------------------------------------
# CUR‑Factorised Linear Layer
# ----------------------------------------------------------------------
class CURLinear(nn.Module):
    def __init__(self, pretrained_weight, pretrained_bias, rank, method="interpolative",
                 tile_size=32, num_shared_tiles=64, use_tile_sharing=True):
        super().__init__()
        out_dim, in_dim = pretrained_weight.shape
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.rank = rank
        self.tile_size = tile_size
        self.num_tiles = num_shared_tiles
        self.use_tile_sharing = use_tile_sharing

        W = pretrained_weight.float()
        if method == "interpolative":
            from scipy.linalg import interpolative
            W_np = W.cpu().numpy()
            idx, proj = interpolative.interp_decomp(W_np, rank)
            C = W[:, idx[:rank]].clone()
            R = torch.tensor(proj[:rank, :], dtype=torch.float32)
        else:  # leverage
            col_norms = W.norm(dim=0)
            probs = col_norms / col_norms.sum()
            selected = torch.multinomial(probs, rank, replacement=False)
            C = W[:, selected].clone()
            R = torch.linalg.lstsq(C, W).solution

        if use_tile_sharing:
            self.register_buffer("C_tiles", self._tile_encode(C))
            self.register_buffer("R_tiles", self._tile_encode(R))
        else:
            self.register_buffer("C", C)
            self.register_buffer("R", R)

        if pretrained_bias is not None:
            self.bias = nn.Parameter(pretrained_bias.clone().float())
        else:
            self.register_buffer('bias', None)

    def _tile_encode(self, M):
        rows, cols = M.shape
        ts = self.tile_size
        tiles = []
        for i in range(0, rows, ts):
            for j in range(0, cols, ts):
                tile = M[i:i+ts, j:j+ts].flatten()
                tiles.append(tile)
        tiles = torch.stack(tiles)
        idx = torch.randperm(len(tiles))[:self.num_tiles]
        return tiles[idx]

    def _tile_decode(self, proto, out_rows, out_cols):
        ts = self.tile_size
        K = proto.size(0)
        mat = torch.zeros(out_rows, out_cols, device=proto.device, dtype=proto.dtype)
        for i in range(0, out_rows, ts):
            hi = min(i+ts, out_rows)
            for j in range(0, out_cols, ts):
                proto_idx = (i // ts + j // ts) % K
                tile_val = proto[proto_idx].view(ts, ts)
                wj = min(j+ts, out_cols)
                mat[i:hi, j:wj] = tile_val[:hi-i, :wj-j]
        return mat

    def _get_matrices(self):
        if self.use_tile_sharing:
            C = self._tile_decode(self.C_tiles, self.out_dim, self.rank)
            R = self._tile_decode(self.R_tiles, self.rank, self.in_dim)
        else:
            C = self.C
            R = self.R
        return C, R

    def forward(self, x):
        C, R = self._get_matrices()
        y = F.linear(x, R)
        y = F.linear(y, C.t())
        if self.bias is not None:
            y = y + self.bias
        return y


# ----------------------------------------------------------------------
# Local block attention (with Softpick)
# ----------------------------------------------------------------------
def local_block_attention(Q, K, V, window, temperature=1.0,
                          softpick_enabled=True, token_valid=None):
    B, H, N, D = Q.shape
    scale = D ** -0.5
    idx = torch.arange(N, device=Q.device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    mask = (dist <= window // 2).unsqueeze(0).unsqueeze(0)

    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, -1e9)

    if token_valid is not None:
        scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e9)

    if softpick_enabled:
        attn = softpick(scores, dim=-1, temperature=temperature)
    else:
        attn = F.softmax(scores, dim=-1)

    return torch.matmul(attn, V)


# ----------------------------------------------------------------------
# Global kernel linear attention (Performer)
# ----------------------------------------------------------------------
def kernel_linear_attention(Q, K, V, rff_proj):
    B, H, N, D = Q.shape
    F_dim = rff_proj.size(-1)

    Q_norm = F.normalize(Q, p=2, dim=-1)
    K_norm = F.normalize(K, p=2, dim=-1)

    Q_proj = torch.einsum('bhnd,hdf->bhnf', Q_norm, rff_proj)
    K_proj = torch.einsum('bhnd,hdf->bhnf', K_norm, rff_proj)

    Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
    K_feat = torch.exp(K_proj) / math.sqrt(F_dim)

    KV = torch.einsum('bhnf,bhnd->bhfd', K_feat, V)
    Z = K_feat.sum(dim=2, keepdim=True)
    ctx = torch.einsum('bhnf,bhfd->bhnd', Q_feat, KV)
    ctx = ctx / (torch.einsum('bhnf,bhfd->bhnd', Q_feat, Z) + 1e-8)
    return ctx


# ----------------------------------------------------------------------
# AEGIS Attention Module
# ----------------------------------------------------------------------
class AEGISAttention(nn.Module):
    def __init__(self, model_config, aegis_cfg: AEGISConfigV1,
                 pretrained_attn=None, bert_mode=False):
        super().__init__()
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.cfg = aegis_cfg
        self.bert_mode = bert_mode

        assert aegis_cfg.num_local_heads + aegis_cfg.num_global_heads == self.num_heads
        self.num_local_heads = aegis_cfg.num_local_heads
        self.num_global_heads = aegis_cfg.num_global_heads

        if pretrained_attn is None:
            raise ValueError("AEGIS requires pretrained_attn for zero‑shot init")

        q_w = pretrained_attn.q_lin.weight.data
        k_w = pretrained_attn.k_lin.weight.data
        v_w = pretrained_attn.v_lin.weight.data
        o_w = pretrained_attn.out_lin.weight.data

        self.q_proj = CURLinear(q_w, pretrained_attn.q_lin.bias,
                                aegis_cfg.cur_rank, aegis_cfg.cur_method,
                                aegis_cfg.tile_size, aegis_cfg.num_shared_tiles,
                                aegis_cfg.use_tile_sharing)
        self.k_proj = CURLinear(k_w, pretrained_attn.k_lin.bias,
                                aegis_cfg.cur_rank, aegis_cfg.cur_method,
                                aegis_cfg.tile_size, aegis_cfg.num_shared_tiles,
                                aegis_cfg.use_tile_sharing)
        self.v_proj = CURLinear(v_w, pretrained_attn.v_lin.bias,
                                aegis_cfg.cur_rank, aegis_cfg.cur_method,
                                aegis_cfg.tile_size, aegis_cfg.num_shared_tiles,
                                aegis_cfg.use_tile_sharing)
        self.out_proj = CURLinear(o_w, pretrained_attn.out_lin.bias,
                                  aegis_cfg.cur_rank, aegis_cfg.cur_method,
                                  aegis_cfg.tile_size, aegis_cfg.num_shared_tiles,
                                  aegis_cfg.use_tile_sharing)

        self.rff_proj = nn.Parameter(
            torch.randn(self.num_global_heads, self.head_dim, aegis_cfg.rff_features) * 0.02)

        if aegis_cfg.use_adaptive_gate:
            self.local_gate_logits = nn.Parameter(torch.zeros(self.num_heads))
            self.local_gate_bias = aegis_cfg.gate_init_bias
        else:
            self.local_gate_logits = None

        self.dropout = nn.Dropout(aegis_cfg.dropout)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs):
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        Hl = self.num_local_heads
        Hg = self.num_global_heads

        Q_all = self.q_proj(hidden_states)
        K_all = self.k_proj(hidden_states)
        V_all = self.v_proj(hidden_states)

        Q_all = Q_all.view(B, N, H, D).transpose(1, 2)
        K_all = K_all.view(B, N, H, D).transpose(1, 2)
        V_all = V_all.view(B, N, H, D).transpose(1, 2)

        Q_local, Q_global = Q_all[:, :Hl], Q_all[:, Hl:]
        K_local, K_global = K_all[:, :Hl], K_all[:, Hl:]
        V_local, V_global = V_all[:, :Hl], V_all[:, Hl:]

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

        ctx_local = local_block_attention(
            Q_local, K_local, V_local,
            window=self.cfg.local_window,
            temperature=self.cfg.softpick_temperature,
            softpick_enabled=self.cfg.softpick_enabled,
            token_valid=token_valid,
        )

        ctx_global = kernel_linear_attention(Q_global, K_global, V_global, self.rff_proj)

        context = torch.cat([ctx_local, ctx_global], dim=1)

        if token_valid is not None:
            context = context * token_valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        output = self.out_proj(context)
        return (output, None)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.bias.device)


# ------------------------------------------------------------
# Framework integration
# ------------------------------------------------------------
from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("aegis_v1")
def inject_aegis_v1(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    cfg = AEGISConfigV1(
        cur_rank=getattr(comp_config, 'cur_rank', 16),
        num_local_heads=8,
        num_global_heads=4,
        local_window=getattr(comp_config, 'block_size', 64),
        rff_features=getattr(comp_config, 'rff_features', 256),
        dropout=getattr(comp_config, 'dropout', 0.1),
    )
    return inject_aegis_into_distilbert(layer, model_config, cfg, old_attn)

def inject_aegis_into_distilbert(layer, model_config, aegis_cfg, old_attn):
    new_attn = AEGISAttention(
        model_config=model_config,
        aegis_cfg=aegis_cfg,
        pretrained_attn=old_attn,
        bert_mode=False,
    )
    layer.attention = new_attn
    return new_attn
