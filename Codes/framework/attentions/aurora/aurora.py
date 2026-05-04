"""
AURORA — Adaptive Unified Robust Orthogonal Rank‑Aware Attention
Ten‑in‑one compression for DistilBERT.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class AURORAConfig:
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    dropout: float = 0.1

    # OATS
    oats_enabled: bool = True
    oats_critical_ratio: float = 0.25
    oats_marginal_ratio: float = 0.50

    # SLiM
    slim_enabled: bool = True
    slim_energy: float = 0.90
    slim_quant_bits: int = 4
    slim_use_2of4: bool = True

    # RPCA
    rpca_enabled: bool = True
    rpca_rank: int = 16

    # SliceGPT
    slicegpt_enabled: bool = True
    slicegpt_reduce_ratio: float = 0.5

    # ReplaceMe
    replaceme_enabled: bool = True
    replaceme_skip_threshold: float = 0.1

    # NMF
    nmf_enabled: bool = True
    nmf_rank: int = 16

    # ProxSparse
    proxsparse_enabled: bool = True
    proxsparse_sparsity: float = 0.5

    # Dual pathway (Scatterbrain/SLA)
    dual_pathway_enabled: bool = True
    kernel_rff_features: int = 128

    # DeltaKV + KVTC
    deltakv_enabled: bool = True
    deltakv_max_refs: int = 128
    deltakv_topk: int = 4
    kvt_bits: int = 4
    kvt_pca_ratio: float = 0.95

    # Softpick
    softpick_enabled: bool = True
    softpick_temp: float = 1.0

    max_seq_len: int = 128


# --------------------------------------------------------------------------
# Utility components
# --------------------------------------------------------------------------
def _select_rank_svd(W: torch.Tensor, energy: float, min_r: int = 4) -> int:
    """Energy‑thresholded rank from singular values."""
    with torch.no_grad():
        _, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
        total = (S ** 2).sum()
        cum = torch.cumsum(S ** 2, dim=0) / total
        r = int(torch.searchsorted(cum, energy).item()) + 1
        return max(min_r, min(r, min(W.shape[0], W.shape[1])))


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Rectified softmax — ~47% sparse, no attention sink."""
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)


class DeltaKVStore:
    """Residual KV reference store (DeltaKV, arXiv 2602.08005)."""
    def __init__(self, max_refs: int = 128, topk: int = 4):
        self.max_refs = max_refs
        self.topk = topk
        self.ref_k: Optional[torch.Tensor] = None
        self.ref_v: Optional[torch.Tensor] = None

    @torch.no_grad()
    def update(self, k: torch.Tensor, v: torch.Tensor):
        B, H, N, D = k.shape
        if B == 0:
            return
        ks = k[0]; vs = v[0]
        if self.ref_k is None:
            self.ref_k, self.ref_v = ks, vs
        else:
            self.ref_k = torch.cat([self.ref_k, ks], dim=1)
            self.ref_v = torch.cat([self.ref_v, vs], dim=1)
        if self.ref_k.size(1) > self.max_refs:
            self.ref_k = self.ref_k[:, -self.max_refs:, :]
            self.ref_v = self.ref_v[:, -self.max_refs:, :]

    @torch.no_grad()
    def compress(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ref_k is None:
            return k, v
        B, H, N, D = k.shape
        tk = min(self.topk, self.ref_k.size(1))
        sim = torch.matmul(k, self.ref_k.transpose(-2, -1))
        _, idx = sim.topk(tk, dim=-1)
        idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        rk = self.ref_k.unsqueeze(0).unsqueeze(2).expand(B, H, N, -1, -1).gather(3, idx_e).mean(dim=3)
        rv = self.ref_v.unsqueeze(0).unsqueeze(2).expand(B, H, N, -1, -1).gather(3, idx_e).mean(dim=3)
        return k - rk, v - rv


class KVTQuantiser:
    """PCA‑decorrelated adaptive quantisation (KVTC, ICLR 2026)."""
    def __init__(self, n_bits: int = 4, pca_ratio: float = 0.95):
        self.n_bits = n_bits
        self.pca_ratio = pca_ratio
        self.basis: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        self.zp: Optional[torch.Tensor] = None
        self.fitted = False

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        X = x.float()
        m = X.mean(dim=0, keepdim=True)
        Xc = X - m
        U, S, _ = torch.linalg.svd(Xc, full_matrices=False)
        total = (S ** 2).sum()
        cum = torch.cumsum(S ** 2, dim=0) / total
        r = min(int(torch.searchsorted(cum, self.pca_ratio).item()) + 1, X.size(1))
        self.basis = U[:, :r].contiguous()
        proj = Xc @ self.basis
        self.scale = (proj.max(dim=0).values - proj.min(dim=0).values) / (2 ** self.n_bits - 1)
        self.scale = self.scale.clamp(min=1e-8)
        self.zp = proj.min(dim=0).values
        self.fitted = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted or self.basis is None:
            return x
        *B, D = x.shape
        xc = x.float() - x.float().mean(dim=-1, keepdim=True)
        proj = xc @ self.basis.to(x.device)
        q = ((proj - self.zp.to(x.device)) / self.scale.to(x.device)).round()
        dq = q * self.scale.to(x.device) + self.zp.to(x.device)
        return (dq @ self.basis.T.to(x.device)).to(x.dtype)


# --------------------------------------------------------------------------
# SLiM‑style QSL projection
# --------------------------------------------------------------------------
class QSLProjection(nn.Module):
    """Quantized‑Sparse‑Low‑rank decomposition (SLiM, ICML 2025)."""
    def __init__(self, W_pretrained: torch.Tensor, bias: Optional[torch.Tensor],
                 energy: float, bits: int, use_2of4: bool):
        super().__init__()
        out_dim, in_dim = W_pretrained.shape
        self.out_dim, self.in_dim = out_dim, in_dim

        U, S, Vt = torch.linalg.svd(W_pretrained.float(), full_matrices=False)
        self.lr_rank = _select_rank_svd(W_pretrained, energy)
        self.A = nn.Parameter((U[:, :self.lr_rank] * S[:self.lr_rank].unsqueeze(0)).float())
        self.B = nn.Parameter(Vt[:self.lr_rank, :].float())

        lr_approx = self.A @ self.B
        residual = W_pretrained.float() - lr_approx
        if use_2of4:
            residual_abs = residual.abs()
            for i in range(out_dim):
                for j in range(0, in_dim, 4):
                    blk = residual_abs[i, j:j+4]
                    _, idx = blk.topk(2)
                    mask_2d = torch.zeros(4, device=residual.device)
                    mask_2d[idx] = 1.0
                    residual[i, j:j+4] *= mask_2d
        r_min, r_max = residual.min(), residual.max()
        if r_max == r_min:
            r_max = r_min + 1e-6
        scale_q = (r_max - r_min) / (2 ** bits - 1)
        self.register_buffer('sparse_q', torch.round((residual - r_min) / scale_q).to(torch.int8))
        self.register_buffer('r_min', r_min)
        self.register_buffer('r_scale', scale_q)

        if bias is not None:
            self.bias = nn.Parameter(bias.float())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_lr = F.linear(F.linear(x, self.B), self.A.t())
        out_sp = F.linear(x, self.sparse_q.float() * self.r_scale + self.r_min)
        out = out_lr + out_sp
        if self.bias is not None:
            out = out + self.bias
        return out


# --------------------------------------------------------------------------
# NMF projection
# --------------------------------------------------------------------------
class NMFProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.rank = rank
        self.H = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.W_mat = nn.Parameter(torch.randn(rank, in_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_nn = F.softplus(self.H)
        W_nn = F.softplus(self.W_mat)
        return F.linear(F.linear(x, W_nn), H_nn.t())


# --------------------------------------------------------------------------
# ProxSparse 2:4 mask
# --------------------------------------------------------------------------
class ProxSparseMask(nn.Module):
    def __init__(self, rank: int, sparsity: float = 0.5):
        super().__init__()
        self.rank = rank
        self.keep = max(1, int(rank * (1.0 - sparsity)))
        self.logits = nn.Parameter(torch.zeros(rank))

    def forward(self) -> torch.Tensor:
        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(self.logits) + 1e-9) + 1e-9)
            y_soft = torch.sigmoid((self.logits + g) * 5.0)
            y_hard = (y_soft > 0.5).float()
            return ((y_hard - y_soft).detach() + y_soft).view(1, self.rank, 1)
        else:
            _, idx = self.logits.topk(self.keep)
            mask = torch.zeros(self.rank, device=self.logits.device)
            mask[idx] = 1.0
            return mask.view(1, self.rank, 1)


# --------------------------------------------------------------------------
# Main AURORA Attention Module
# --------------------------------------------------------------------------
class AURORAAttention(nn.Module):
    def __init__(self, model_config, aurora_cfg: AURORAConfig,
                 layer_idx: int = 0, total_layers: int = 6,
                 pretrained_attn=None, bert_mode: bool = False):
        super().__init__()
        self.cfg = aurora_cfg
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.layer_idx = layer_idx
        self.bert_mode = bert_mode

        # Build projections
        if pretrained_attn is not None:
            q_w = pretrained_attn.q_lin.weight.data
            k_w = pretrained_attn.k_lin.weight.data
            v_w = pretrained_attn.v_lin.weight.data
            o_w = pretrained_attn.out_lin.weight.data

            if aurora_cfg.slim_enabled:
                self.q_proj = QSLProjection(q_w, pretrained_attn.q_lin.bias,
                                            aurora_cfg.slim_energy, aurora_cfg.slim_quant_bits,
                                            aurora_cfg.slim_use_2of4)
                self.k_proj = QSLProjection(k_w, pretrained_attn.k_lin.bias,
                                            aurora_cfg.slim_energy, aurora_cfg.slim_quant_bits,
                                            aurora_cfg.slim_use_2of4)
                self.v_proj = QSLProjection(v_w, pretrained_attn.v_lin.bias,
                                            aurora_cfg.slim_energy, aurora_cfg.slim_quant_bits,
                                            aurora_cfg.slim_use_2of4)
                self.out_proj = QSLProjection(o_w, pretrained_attn.out_lin.bias,
                                              aurora_cfg.slim_energy, aurora_cfg.slim_quant_bits,
                                              aurora_cfg.slim_use_2of4)
            else:
                self.q_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
                self.k_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
                self.v_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
                self.out_proj = nn.Linear(self.all_head_size, self.hidden_size, bias=True)
                self._copy_weights(q_w, k_w, v_w, o_w, pretrained_attn)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.k_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.v_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.out_proj = nn.Linear(self.all_head_size, self.hidden_size, bias=True)

        # NMF parallel path
        self.nmf_proj = None
        if aurora_cfg.nmf_enabled:
            self.nmf_proj = NMFProjection(self.hidden_size, self.all_head_size, aurora_cfg.nmf_rank)

        # ProxSparse masks (attach to low‑rank branches if SLiM is used)
        self.ps_q = self.ps_k = self.ps_v = self.ps_out = None
        if aurora_cfg.proxsparse_enabled and hasattr(self.q_proj, 'lr_rank'):
            self.ps_q = ProxSparseMask(self.q_proj.lr_rank, aurora_cfg.proxsparse_sparsity)
            self.ps_k = ProxSparseMask(self.k_proj.lr_rank, aurora_cfg.proxsparse_sparsity)
            self.ps_v = ProxSparseMask(self.v_proj.lr_rank, aurora_cfg.proxsparse_sparsity)
            self.ps_out = ProxSparseMask(self.out_proj.lr_rank, aurora_cfg.proxsparse_sparsity)

        # SliceGPT basis
        self.slicegpt_basis: Optional[torch.Tensor] = None
        if aurora_cfg.slicegpt_enabled:
            self.effective_dim = max(8, int(self.head_dim * (1.0 - aurora_cfg.slicegpt_reduce_ratio)))
        else:
            self.effective_dim = self.head_dim

        # Dual pathway kernel
        if aurora_cfg.dual_pathway_enabled:
            self.rff_proj = nn.Parameter(
                torch.randn(self.head_dim, aurora_cfg.kernel_rff_features) * 0.02)

        # DeltaKV & KVTC
        self.deltakv: Optional[DeltaKVStore] = None
        self.kvt: Optional[KVTQuantiser] = None
        if aurora_cfg.deltakv_enabled:
            self.deltakv = DeltaKVStore(aurora_cfg.deltakv_max_refs, aurora_cfg.deltakv_topk)
        if aurora_cfg.deltakv_enabled:
            self.kvt = KVTQuantiser(aurora_cfg.kvt_bits, aurora_cfg.kvt_pca_ratio)

        self.dropout = nn.Dropout(aurora_cfg.dropout)
        self.register_buffer('oats_mean', None)
        self.register_buffer('oats_cov', None)
        self.oats_calibrated = False

    def _copy_weights(self, q_w, k_w, v_w, o_w, pretrained_attn):
        self.q_proj.weight.data = q_w.clone()
        self.q_proj.bias.data = (pretrained_attn.q_lin.bias.data.clone()
                                 if pretrained_attn.q_lin.bias is not None
                                 else torch.zeros(self.all_head_size))
        self.k_proj.weight.data = k_w.clone()
        self.k_proj.bias.data = (pretrained_attn.k_lin.bias.data.clone()
                                 if pretrained_attn.k_lin.bias is not None
                                 else torch.zeros(self.all_head_size))
        self.v_proj.weight.data = v_w.clone()
        self.v_proj.bias.data = (pretrained_attn.v_lin.bias.data.clone()
                                 if pretrained_attn.v_lin.bias is not None
                                 else torch.zeros(self.all_head_size))
        self.out_proj.weight.data = o_w.clone()
        self.out_proj.bias.data = (pretrained_attn.out_lin.bias.data.clone()
                                   if pretrained_attn.out_lin.bias is not None
                                   else torch.zeros(self.hidden_size))

    @torch.no_grad()
    def _oats_score_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, D = hidden_states.shape
        flat = hidden_states.view(-1, D)
        if self.oats_mean is None or not self.oats_calibrated:
            self.oats_mean = flat.mean(dim=0)
            Xc = flat - self.oats_mean
            self.oats_cov = (Xc.T @ Xc) / len(Xc)
            self.oats_calibrated = True
        Xc = flat - self.oats_mean.to(hidden_states.device)
        diag_cov = self.oats_cov.diag().to(hidden_states.device).clamp(min=1e-8)
        scores = (Xc ** 2 / diag_cov.unsqueeze(0)).sum(dim=-1)
        return scores.view(B, N)

    def _rpca_decompose_attention(self, scores_full: torch.Tensor,
                                  rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, N, _ = scores_full.shape
        k = max(1, int(N * self.cfg.oats_critical_ratio))
        _, top_idx = scores_full.topk(k, dim=-1)
        sparse_mask = torch.zeros_like(scores_full).scatter_(-1, top_idx, 1.0)

        scores_flat = scores_full.reshape(B * H, N, N)
        U, S, Vt = torch.linalg.svd(scores_flat.float(), full_matrices=False)
        U_r = U[:, :, :rank]; S_r = S[:, :rank]; Vt_r = Vt[:, :rank, :]
        scores_lr = (U_r * S_r.unsqueeze(1)) @ Vt_r
        scores_lr = scores_lr.reshape(B, H, N, N).to(scores_full.dtype)
        return sparse_mask, scores_lr

    def _kernel_linear_attention(self, Q: torch.Tensor, K: torch.Tensor,
                                 V: torch.Tensor) -> torch.Tensor:
        B, H, N, D = Q.shape
        proj = torch.matmul(K, self.rff_proj.to(K.device))
        phi_k = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / (self.rff_proj.size(1) ** 0.25)
        q_proj = torch.matmul(Q, self.rff_proj.to(Q.device))
        phi_q = torch.cat([torch.cos(q_proj), torch.sin(q_proj)], dim=-1) / (self.rff_proj.size(1) ** 0.25)
        KV = torch.matmul(phi_k.transpose(-2, -1), V)
        Z = phi_k.sum(dim=2, keepdim=True)
        lin_ctx = torch.matmul(phi_q, KV)
        lin_ctx = lin_ctx / (torch.matmul(phi_q, Z.transpose(-2, -1)) + 1e-8)
        return lin_ctx

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs):
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        cfg = self.cfg

        # ── OATS token triage ──
        if cfg.oats_enabled:
            token_scores = self._oats_score_tokens(hidden_states)
            n_critical = max(1, int(N * cfg.oats_critical_ratio))
            n_marginal = max(1, int(N * cfg.oats_marginal_ratio))
            _, sorted_idx = token_scores.sort(dim=-1, descending=True)
            idx_critical = sorted_idx[:, :n_critical]
            idx_marginal = sorted_idx[:, n_critical:n_critical + n_marginal]
            mask_critical = torch.zeros(B, N, device=hidden_states.device).scatter_(1, idx_critical, 1.0)
            mask_marginal = torch.zeros(B, N, device=hidden_states.device).scatter_(1, idx_marginal, 1.0)
        else:
            mask_critical = torch.ones(B, N, device=hidden_states.device)
            mask_marginal = torch.zeros(B, N, device=hidden_states.device)

        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        if self.nmf_proj is not None:
            Q = Q + 0.1 * self.nmf_proj(hidden_states)
            K = K + 0.1 * self.nmf_proj(hidden_states)

        Q = Q.view(B, N, H, D).transpose(1, 2)
        K = K.view(B, N, H, D).transpose(1, 2)
        V = V.view(B, N, H, D).transpose(1, 2)

        # ── SliceGPT PCA reduction ──
        if cfg.slicegpt_enabled and self.effective_dim < D:
            if self.slicegpt_basis is None or self.slicegpt_basis.size(0) != D:
                with torch.no_grad():
                    Qf = Q.transpose(1, 2).reshape(B * N, H * D).float()
                    _, _, Vt = torch.linalg.svd(Qf, full_matrices=False)
                    self.slicegpt_basis = Vt[:self.effective_dim, :].to(Q.device)
            basis_mat = self.slicegpt_basis
            Q2 = Q.transpose(1, 2).reshape(B, N, H * D)
            K2 = K.transpose(1, 2).reshape(B, N, H * D)
            V2 = V.transpose(1, 2).reshape(B, N, H * D)
            Q_red = torch.matmul(Q2, basis_mat.T)
            K_red = torch.matmul(K2, basis_mat.T)
            V_red = torch.matmul(V2, basis_mat.T)
            eff_D = self.effective_dim
            if eff_D % H == 0:
                Q = Q_red.view(B, N, H, eff_D // H).transpose(1, 2)
                K = K_red.view(B, N, H, eff_D // H).transpose(1, 2)
                V = V_red.view(B, N, H, eff_D // H).transpose(1, 2)
                D = eff_D // H

        K_orig, V_orig = K, V

        if cfg.deltakv_enabled and self.deltakv is not None:
            K, V = self.deltakv.compress(K, V)
        if cfg.deltakv_enabled and self.kvt is not None:
            if self.training and not self.kvt.fitted:
                self.kvt.fit(K.reshape(B * H * N, D))
            elif self.kvt.fitted:
                K = self.kvt.compress(K.reshape(B * H, N, D)).view(B, H, N, D)
                V = self.kvt.compress(V.reshape(B * H, N, D)).view(B, H, N, D)

        scale = D ** -0.5
        scores_crit = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if cfg.dual_pathway_enabled and mask_marginal.sum() > 0:
            lin_ctx = self._kernel_linear_attention(
                Q * mask_marginal.unsqueeze(1).unsqueeze(-1), K, V)
        else:
            lin_ctx = torch.zeros_like(V)

        if cfg.rpca_enabled and mask_critical.sum() > 0:
            sparse_mask, scores_lr = self._rpca_decompose_attention(scores_crit, cfg.rpca_rank)
            scores_combined = sparse_mask * scores_crit + (1.0 - sparse_mask) * scores_lr
        else:
            scores_combined = scores_crit

        mask_attn = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        token_valid = None
        if mask_attn is not None:
            if mask_attn.dim() == 4 and mask_attn.shape[-2] == mask_attn.shape[-1]:
                token_valid = mask_attn[:, 0, 0, :] > -1e9
            elif mask_attn.dim() == 4:
                token_valid = mask_attn.squeeze(1).squeeze(1) > -1e9
            elif mask_attn.dim() == 3:
                token_valid = mask_attn.squeeze(1).bool()
            elif mask_attn.dim() == 2:
                token_valid = mask_attn.bool()
            if token_valid is not None:
                scores_combined = scores_combined.masked_fill(
                    ~token_valid.unsqueeze(1).unsqueeze(2), -1e5)

        attn_w = (softpick(scores_combined, dim=-1, temperature=cfg.softpick_temp)
                  if cfg.softpick_enabled else F.softmax(scores_combined, dim=-1))
        attn_w = self.dropout(attn_w)

        context_full = torch.matmul(attn_w, V)
        weight_critical = mask_critical.unsqueeze(1).unsqueeze(-1).to(context_full.dtype)
        weight_marginal = mask_marginal.unsqueeze(1).unsqueeze(-1).to(lin_ctx.dtype)
        context = weight_critical * context_full + weight_marginal * lin_ctx

        context = context.transpose(1, 2).contiguous().view(B, N, H * D)

        if (cfg.slicegpt_enabled and self.slicegpt_basis is not None
                and self.effective_dim < self.head_dim):
            context = torch.matmul(context, self.slicegpt_basis)

        out = self.out_proj(context)

        if self.training and cfg.deltakv_enabled and self.deltakv is not None:
            self.deltakv.update(K_orig, V_orig)

        return (out, attn_w) if output_attentions else (out,)

    def get_ortho_loss(self) -> torch.Tensor:
        if isinstance(self.out_proj, nn.Linear):
            return torch.tensor(0.0, device=self.out_proj.bias.device)
        return torch.tensor(0.0, device=self.out_proj.B.device)


from framework.attentions.core.registry import ATTENTION_REGISTRY

def inject_aurora_into_distilbert(layer, model_config, aurora_cfg, layer_idx, total_layers, old_attn):
    new_attn = AURORAAttention(model_config=model_config, aurora_cfg=aurora_cfg,
                               layer_idx=layer_idx, total_layers=total_layers,
                               pretrained_attn=old_attn, bert_mode=False)
    layer.attention = new_attn
    return new_attn

@ATTENTION_REGISTRY.register("aurora")
def inject_aurora(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """Registry bridge that builds AURORAConfig from CompressionConfig."""
    cfg = AURORAConfig(
        dropout=getattr(comp_config, 'dropout', 0.1),
        slim_enabled=getattr(comp_config, 'slim_enabled', True),
        dual_pathway_enabled=getattr(comp_config, 'aks_enabled', True),
        softpick_enabled=True,
        # add other mappings as needed
    )
    return inject_aurora_into_distilbert(layer, model_config, cfg, layer_idx, total_layers, old_attn)
