"""
PRISM — Pruning‑aware Residual Integrated Sparse Memory Attention
Full version with six compression techniques.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

@dataclass
class PRISMConfig:
    energy_threshold: float = 0.95
    min_rank: int = 4
    max_rank: int = 64
    armor_enabled: bool = True
    armor_block_size: int = 8
    ast_enabled: bool = True
    ast_sparsity: float = 0.5
    ast_mask_lr: float = 0.01
    deltakv_enabled: bool = True
    deltakv_max_refs: int = 128
    deltakv_similarity_topk: int = 4
    kvt_enabled: bool = True
    kvt_quant_bits: int = 4
    kvt_pca_components: float = 0.95
    softpick_enabled: bool = True
    softpick_temperature: float = 1.0
    dropout: float = 0.1

def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    x = x / temperature
    p = F.softmax(x, dim=dim)
    threshold = p.mean(dim=dim, keepdim=True)
    p_rect = torch.where(p > threshold, p, torch.zeros_like(p))
    denom = p_rect.sum(dim=dim, keepdim=True) + 1e-9
    return p_rect / denom

class BlockDiagonal(nn.Module):
    """Block‑diagonal matrix for ARMOR structured sparsity."""
    def __init__(self, dim: int, block_size: int, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.bs = block_size
        self.padded_dim = ((dim + block_size - 1) // block_size) * block_size
        n_blocks = self.padded_dim // block_size
        self.weight = nn.Parameter(torch.empty(n_blocks, block_size, block_size))
        nn.init.normal_(self.weight, std=0.02)
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x):
        *batch, D = x.shape
        if self.padded_dim > D:
            pad = torch.zeros(*batch, self.padded_dim - D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        n_blocks = self.padded_dim // self.bs
        x = x.view(*batch, n_blocks, self.bs)
        x = torch.einsum('...nb,nbc->...nc', x, self.weight)
        x = x.reshape(*batch, self.padded_dim)
        x = x[..., :D]
        if self.bias is not None:
            x = x + self.bias
        return x

class KVTQuantiser:
    """PCA‑based adaptive quantisation for K/V."""
    def __init__(self, n_bits: int = 4, pca_components: float = 0.95):
        self.n_bits = n_bits
        self.pca_components = pca_components
        self.pca_basis: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None
        self.fitted = False

    @torch.no_grad()
    def fit(self, k: torch.Tensor, v: torch.Tensor):
        B, H, N, D = k.shape
        X = torch.cat([k.reshape(-1, D), v.reshape(-1, D)], dim=0)
        X = X.float()
        mean = X.mean(dim=0, keepdim=True)
        X_centered = X - mean
        U, S, _ = torch.linalg.svd(X_centered, full_matrices=False)
        total_var = S.pow(2).sum()
        cum_var = torch.cumsum(S.pow(2), dim=0) / total_var
        rank = torch.searchsorted(cum_var, self.pca_components).item() + 1
        rank = min(rank, D)
        self.pca_basis = U[:, :rank].contiguous()
        projected = X_centered @ self.pca_basis
        self.scale = (projected.max(dim=0).values - projected.min(dim=0).values) / (2 ** self.n_bits - 1)
        self.scale = self.scale.clamp(min=1e-8)
        self.zero_point = projected.min(dim=0).values
        self.fitted = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted or self.pca_basis is None:
            return x
        *B, D = x.shape
        xc = x.float() - x.float().mean(dim=-1, keepdim=True)
        proj = xc @ self.pca_basis.to(x.device)
        q = ((proj - self.zero_point.to(x.device)) / self.scale.to(x.device)).round()
        dq = q * self.scale.to(x.device) + self.zero_point.to(x.device)
        return (dq @ self.pca_basis.T.to(x.device)).to(x.dtype)

class PRISMAttention(nn.Module):
    """
    PRISM attention: 6 integrated compression techniques.
    - SVD rank reduction on projections
    - ARMOR block‑diagonal structured sparsity
    - AST learned sparse masking
    - DeltaKV residual compression via reference tokens
    - KVT PCA quantisation of keys/values
    - Softpick sparse activation
    """
    def __init__(self, model_config, prism_cfg: PRISMConfig, layer_idx=0, total_layers=6,
                 pretrained_attn=None, bert_mode=False):
        super().__init__()
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.cfg = prism_cfg
        self.layer_idx = layer_idx
        self.bert_mode = bert_mode

        if pretrained_attn is not None:
            q_w = pretrained_attn.q_lin.weight.data
            k_w = pretrained_attn.k_lin.weight.data
            v_w = pretrained_attn.v_lin.weight.data
            out_w = pretrained_attn.out_lin.weight.data

            self.q_rank = self._select_rank(q_w)
            self.k_rank = self._select_rank(k_w)
            self.v_rank = self._select_rank(v_w)
            self.out_rank = self._select_rank(out_w)

            self.q_down, self.q_up = self._svd_factorise(q_w, pretrained_attn.q_lin.bias, self.q_rank, self.hidden_size, self.all_head_size)
            self.k_down, self.k_up = self._svd_factorise(k_w, pretrained_attn.k_lin.bias, self.k_rank, self.hidden_size, self.all_head_size)
            self.v_down, self.v_up = self._svd_factorise(v_w, pretrained_attn.v_lin.bias, self.v_rank, self.hidden_size, self.all_head_size)
            self.out_down, self.out_up = self._svd_factorise(out_w, pretrained_attn.out_lin.bias, self.out_rank, self.all_head_size, self.hidden_size)
        else:
            self.q_rank = self.k_rank = self.v_rank = self.out_rank = self.head_dim
            self.q_down = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.q_up = nn.Linear(self.head_dim, self.all_head_size, bias=True)
            self.k_down = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.k_up = nn.Linear(self.head_dim, self.all_head_size, bias=True)
            self.v_down = nn.Linear(self.hidden_size, self.head_dim, bias=False)
            self.v_up = nn.Linear(self.head_dim, self.all_head_size, bias=True)
            self.out_down = nn.Linear(self.all_head_size, self.head_dim, bias=False)
            self.out_up = nn.Linear(self.head_dim, self.hidden_size, bias=True)

        self.armor_q = self.armor_k = self.armor_v = self.armor_out = None
        if self.cfg.armor_enabled:
            self.armor_q = BlockDiagonal(self.all_head_size, self.cfg.armor_block_size)
            self.armor_k = BlockDiagonal(self.all_head_size, self.cfg.armor_block_size)
            self.armor_v = BlockDiagonal(self.all_head_size, self.cfg.armor_block_size)
            self.armor_out = BlockDiagonal(self.hidden_size, self.cfg.armor_block_size)

        self.ast_q_logits = self.ast_k_logits = self.ast_v_logits = self.ast_out_logits = None
        if self.cfg.ast_enabled:
            self.ast_q_logits = nn.Parameter(torch.zeros(self.q_rank))
            self.ast_k_logits = nn.Parameter(torch.zeros(self.k_rank))
            self.ast_v_logits = nn.Parameter(torch.zeros(self.v_rank))
            self.ast_out_logits = nn.Parameter(torch.zeros(self.out_rank))

        self.ref_k: Optional[torch.Tensor] = None
        self.ref_v: Optional[torch.Tensor] = None
        self.ref_count: int = 0
        self.kvt: Optional[KVTQuantiser] = None
        if self.cfg.kvt_enabled:
            self.kvt = KVTQuantiser(self.cfg.kvt_quant_bits, self.cfg.kvt_pca_components)

        self.dropout = nn.Dropout(self.cfg.dropout)

    def _select_rank(self, W):
        with torch.no_grad():
            _, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
            total_e = S.pow(2).sum()
            cum_e = torch.cumsum(S.pow(2), dim=0) / total_e
            idx = torch.searchsorted(cum_e, self.cfg.energy_threshold)
            rank = int(idx.item()) + 1
            max_r = min(W.shape[0], W.shape[1])
            return max(self.cfg.min_rank, min(rank, max_r, self.cfg.max_rank))

    @staticmethod
    def _svd_factorise(W, bias, rank, in_dim, out_dim):
        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        down_w = Vt[:rank, :].float()
        up_w = (U[:, :rank] * S[:rank].unsqueeze(0)).float()
        down = nn.Linear(in_dim, rank, bias=False)
        up = nn.Linear(rank, out_dim, bias=True)
        down.weight.data = down_w
        up.weight.data = up_w
        if bias is not None:
            up.bias.data = bias.float()
        else:
            up.bias.data.zero_()
        return down, up

    def _get_ast_mask(self, logits):
        if not self.training or not self.cfg.ast_enabled:
            R = logits.size(0)
            keep = max(1, int(R * (1.0 - self.cfg.ast_sparsity)))
            _, idx = logits.topk(keep)
            mask = torch.zeros(R, device=logits.device)
            mask[idx] = 1.0
            return mask.view(1, R, 1)
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        y_soft = torch.sigmoid((logits + gumbel) * 5.0)
        mask_hard = (y_soft > 0.5).float()
        return ((mask_hard - y_soft).detach() + y_soft).view(1, -1, 1)

    @torch.no_grad()
    def _deltakv_update(self, k, v):
        if not self.cfg.deltakv_enabled: return
        B, H, N, D = k.shape
        if B == 0: return
        k_flat = k[0]
        v_flat = v[0]
        if self.ref_k is None or self.ref_count < self.cfg.deltakv_max_refs:
            self.ref_k = k_flat if self.ref_k is None else torch.cat([self.ref_k, k_flat[:, -1:, :]], dim=1)
            self.ref_v = v_flat if self.ref_v is None else torch.cat([self.ref_v, v_flat[:, -1:, :]], dim=1)
            self.ref_count += 1
            if self.ref_count > self.cfg.deltakv_max_refs:
                self.ref_k = self.ref_k[:, -self.cfg.deltakv_max_refs:, :]
                self.ref_v = self.ref_v[:, -self.cfg.deltakv_max_refs:, :]
                self.ref_count = self.cfg.deltakv_max_refs

    @torch.no_grad()
    def _deltakv_compress(self, k, v):
        if not self.cfg.deltakv_enabled or self.ref_k is None: return k, v
        B, H, N, D = k.shape
        topk = min(self.cfg.deltakv_similarity_topk, self.ref_k.size(1))
        sim = torch.matmul(k, self.ref_k.transpose(-2, -1))
        _, idx = sim.topk(topk, dim=-1)
        idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        rk = self.ref_k.unsqueeze(0).unsqueeze(2).expand(B, H, N, -1, -1).gather(3, idx_e).mean(dim=3)
        rv = self.ref_v.unsqueeze(0).unsqueeze(2).expand(B, H, N, -1, -1).gather(3, idx_e).mean(dim=3)
        return k - rk, v - rv

    def _project(self, x, down, up, ast_logits=None, armor=None):
        h = down(x)
        if ast_logits is not None and self.cfg.ast_enabled:
            mask = self._get_ast_mask(ast_logits)
            h = h * mask.squeeze(-1)
        out = up(h)
        if armor is not None and self.cfg.armor_enabled:
            out = armor(out)
        return out

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self._project(hidden_states, self.q_down, self.q_up, self.ast_q_logits, self.armor_q)
        K = self._project(hidden_states, self.k_down, self.k_up, self.ast_k_logits, self.armor_k)
        V = self._project(hidden_states, self.v_down, self.v_up, self.ast_v_logits, self.armor_v)

        Q = Q.view(B, N, H, D).transpose(1, 2)
        K = K.view(B, N, H, D).transpose(1, 2)
        V = V.view(B, N, H, D).transpose(1, 2)

        K_orig, V_orig = K, V
        if self.training or self.ref_k is not None:
            K, V = self._deltakv_compress(K, V)

        if self.cfg.kvt_enabled and self.kvt is not None:
            if self.training and not self.kvt.fitted:
                self.kvt.fit(K, V)
            elif self.kvt.fitted:
                K = self.kvt.compress(K.view(B * H, N, D)).view(B, H, N, D)
                V = self.kvt.compress(V.view(B * H, N, D)).view(B, H, N, D)

        scale = D ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
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
                scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e5)

        if self.cfg.softpick_enabled:
            attn_w = softpick(scores, dim=-1, temperature=self.cfg.softpick_temperature)
        else:
            attn_w = F.softmax(scores, dim=-1)
        attn_w = self.dropout(attn_w)

        context = torch.matmul(attn_w, V)
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)

        out = self._project(context, self.out_down, self.out_up, self.ast_out_logits, self.armor_out)

        if self.training and self.cfg.deltakv_enabled:
            self._deltakv_update(K_orig, V_orig)

        return (out, attn_w) if output_attentions else (out,)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.q_up.weight.device)


# ------------------------------------------------------------
# Framework injection & registration
# ------------------------------------------------------------
from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("prism_full")
def inject_prism_full(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    cfg = PRISMConfig(
        dropout=getattr(comp_config, 'dropout', 0.1),
        energy_threshold=getattr(comp_config, 'svd_energy', 0.95),
        ast_enabled=getattr(comp_config, 'ast_enabled', True),
        armor_enabled=getattr(comp_config, 'armor_enabled', True),
        deltakv_enabled=getattr(comp_config, 'deltakv_enabled', True),
        kvt_enabled=getattr(comp_config, 'kvt_enabled', True),
        softpick_enabled=getattr(comp_config, 'softpick_enabled', True),
    )
    new_attn = PRISMAttention(
        model_config=model_config,
        prism_cfg=cfg,
        layer_idx=layer_idx,
        total_layers=total_layers,
        pretrained_attn=old_attn,
        bert_mode=False,
    )
    layer.attention = new_attn
    return new_attn