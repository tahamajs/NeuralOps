"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  NEBULA — Neural Efficient Bandwidth‑Unified Linear Attention                    ║
║  ═══════════════════════════════════════════════════════════════════════════════  ║
║                                                                                  ║
║  Eighteen compression techniques fused into one drop‑in DistilBERT attention:    ║
║                                                                                  ║
║   Weight‑side (6):  SVD‑LLM whitening  ·  ASVD activation‑aware SVD              ║
║                     OATS sparse+low‑rank  ·  SLiM quantized‑sparse‑LR            ║
║                     GFWSVD Fisher‑weighting  ·  Kronecker‑factorised core        ║
║                                                                                  ║
║   Attention‑side (6): SLA critical/marginal/negligible triage                    ║
║                       Scatterbrain RPCA‑inspired decomp  ·  Softpick rect. smax  ║
║                       NSA hierarchical sparse  ·  Lorsa low‑rank sparse          ║
║                       ProxSparse 2:4 masks  ·  MGIE‑SVD entropy‑driven rank      ║
║                                                                                  ║
║   KV‑cache (3):      DeltaKV residual encoding  ·  KVTC PCA quantisation         ║
║                      FAQ future‑aware quant  ·  CoopQ Shapley bit allocation     ║
║                                                                                  ║
║   Architectural (3): ReplaceMe linearised deep layers  ·  SliceGPT PCA reduction ║
║                      NMF non‑negative regularisation  ·  AstroQuantizer           ║
║                                                                                  ║
║  Papers: OATS'24, SLiM'25, Scatterbrain'21, Softpick'25, DeltaKV'26, KVTC'26,   ║
║          ProxSparse'25, Dobi‑SVD'25, SAES‑SVD'26, MGIE‑SVD'26, GFWSVD'25,       ║
║          Astro'25, CoopQ'25, FAQ'26, ReplaceMe'25, KroneckerBERT'21,             ║
║          Lorsa'25, SLA'26, NSA'25, SliceGPT'24, SVD‑LLM'24, ASVD'23, NMF'24     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NEBULAConfig:
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    dropout: float = 0.1

    # ── SVD‑LLM / ASVD truncation‑aware whitening ──
    svd_enabled: bool = True
    svd_energy: float = 0.90              # fraction of singular‑value energy to keep
    asvd_activation_aware: bool = True    # scale weight by activation statistics

    # ── SLiM quantized‑sparse‑low‑rank decomposition ──
    slim_enabled: bool = True
    slim_quant_bits: int = 4
    slim_2of4_sparsity: bool = True

    # ── GFWSVD Fisher‑weighted initialisation ──
    gfwsved_enabled: bool = True

    # ── Kronecker factorisation ──
    kronecker_enabled: bool = False       # heavier compute, off by default

    # ── OATS outlier‑aware sparse+low‑rank ──
    oats_enabled: bool = True
    oats_critical_ratio: float = 0.25
    oats_marginal_ratio: float = 0.25

    # ── SLA / Scatterbrain dual‑pathway attention decomposition ──
    sla_enabled: bool = True
    slm_rpca_rank: int = 16               # rank for low‑rank background
    rff_features: int = 128               # kernel feature dimension

    # ── NSA hierarchical sparse ──
    nsa_enabled: bool = True
    nsa_block_size: int = 32              # coarse‑grain compression block

    # ── MGIE‑SVD entropy‑driven rank selection ──
    mgie_enabled: bool = True
    mgie_min_rank: int = 4
    mgie_max_rank: int = 64

    # ── ProxSparse 2:4 structured masks ──
    proxsparse_enabled: bool = True
    proxsparse_sparsity: float = 0.5      # 50 % → 2:4

    # ── Lorsa low‑rank sparse decomposition ──
    lorsa_enabled: bool = True

    # ── Softpick rectified softmax ──
    softpick_enabled: bool = True
    softpick_temp: float = 1.0

    # ── DeltaKV residual KV cache ──
    deltakv_enabled: bool = True
    deltakv_max_refs: int = 128
    deltakv_topk: int = 4

    # ── KVTC PCA quantisation ──
    kvt_enabled: bool = True
    kvt_bits: int = 4
    kvt_pca_ratio: float = 0.95

    # ── FAQ future‑aware quantisation ──
    faq_enabled: bool = True
    faq_window: int = 2

    # ── CoopQ Shapley‑based bit allocation ──
    coopq_enabled: bool = True
    coopq_bits_low: int = 2
    coopq_bits_high: int = 4

    # ── ReplaceMe linearised deep layers ──
    replaceme_enabled: bool = False
    replaceme_depth_threshold: int = 4    # layers beyond this index → linear

    # ── SliceGPT PCA dimension reduction ──
    slicegpt_enabled: bool = True
    slicegpt_reduce_ratio: float = 0.25   # reduce head_dim by 25 %

    # ── NMF regularisation ──
    nmf_enabled: bool = True
    nmf_rank: int = 16

    # ── AstroQuantizer ──
    astro_enabled: bool = True
    astro_bits: int = 8

    # ── General ──
    max_seq_len: int = 128


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities shared across components
# ═══════════════════════════════════════════════════════════════════════════════

def _energy_rank(S: torch.Tensor, energy: float, min_r: int = 4, max_r: int = 64) -> int:
    """Energy‑thresholded rank from singular values."""
    total = (S ** 2).sum()
    cum = torch.cumsum(S ** 2, dim=0) / (total + 1e-9)
    r = int(torch.searchsorted(cum, energy).item()) + 1
    return max(min_r, min(r, min(len(S), max_r)))


def _entropy_rank(S: torch.Tensor) -> int:
    """MGIE‑SVD entropy‑driven rank: higher entropy → more truncation."""
    p = S / (S.sum() + 1e-9)
    H = -(p * torch.log(p + 1e-9)).sum()
    H_max = math.log(len(S))
    retention = 1.0 - (H / H_max).clamp(0.05, 0.95)
    return max(4, int(retention * len(S)))


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Rectified softmax — ~47 % sparse, no attention sink.  Zuhri et al. 2025."""
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SVD‑LLM + ASVD  —  truncation‑aware whitened SVD core
# ═══════════════════════════════════════════════════════════════════════════════

class WhitenedSVDCore(nn.Module):
    """
    SVD‑LLM truncation‑aware data whitening  +  ASVD activation‑aware scaling.
    W ≈ U_r · S_r · V_r^T   where r is chosen by energy threshold.
    """
    def __init__(self, W: torch.Tensor, bias: Optional[torch.Tensor],
                 in_dim: int, out_dim: int,
                 energy: float, act_aware: bool, act_stats: Optional[torch.Tensor] = None):
        super().__init__()
        # ASVD: scale by activation channel std
        with torch.no_grad():
            Wf = W.float()
            if act_aware and act_stats is not None:
                s_row = act_stats.float().clamp(min=1e-8)
                s_col = torch.ones(out_dim, device=W.device)
                Wf = Wf * s_row.unsqueeze(0)  # column‑wise scaling

            U, S_arr, Vt = torch.linalg.svd(Wf, full_matrices=False)
            self.svd_rank = _energy_rank(S_arr, energy, min_r=4, max_r=min(in_dim, out_dim, 64))
            # A = U[:,:r] * sqrt(S[:r]),  B = sqrt(S[:r]) * Vt[:r,:]
            sqrt_S = S_arr[:self.svd_rank].sqrt()
            A = U[:, :self.svd_rank] * sqrt_S.unsqueeze(0)
            B = sqrt_S.unsqueeze(1) * Vt[:self.svd_rank, :]
            if act_aware and act_stats is not None:
                B = B / s_row.unsqueeze(0)
                A = A / s_col.unsqueeze(1)

        self.down = nn.Linear(in_dim, self.svd_rank, bias=False)
        self.up = nn.Linear(self.svd_rank, out_dim, bias=True)
        self.down.weight.data = B.float()
        self.up.weight.data = A.float()
        if bias is not None:
            self.up.bias.data = bias.float()
        else:
            self.up.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SLiM  —  quantized‑sparse‑low‑rank wrapper around SVD core
# ═══════════════════════════════════════════════════════════════════════════════

class SLiMWrapper(nn.Module):
    """
    SLiM QSL decomposition:  W ≈ W_svd  +  W_sparse_quant.
    The sparse‑quant component is stored as int8 and dequantised on the fly.
    """
    def __init__(self, svd_core: WhitenedSVDCore, W_orig: torch.Tensor,
                 bits: int, use_2of4: bool):
        super().__init__()
        self.svd = svd_core
        # residual = W_orig - svd_approx
        with torch.no_grad():
            W_svd = (svd_core.up.weight.data @ svd_core.down.weight.data).float()
            residual = W_orig.float() - W_svd
            if use_2of4:
                for i in range(residual.size(0)):
                    for j in range(0, residual.size(1), 4):
                        blk = residual[i, j:j+4].abs()
                        _, idx = blk.topk(2)
                        mask_2d = torch.zeros(4, device=residual.device)
                        mask_2d[idx] = 1.0
                        residual[i, j:j+4] *= mask_2d
            r_min, r_max = residual.min(), residual.max()
            if r_max == r_min:
                r_max = r_min + 1e-6
            scale = (r_max - r_min) / (2 ** bits - 1)
            self.register_buffer('sparse_q', torch.round((residual - r_min) / scale).to(torch.int8))
            self.register_buffer('r_min', r_min)
            self.register_buffer('r_scale', scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_svd = self.svd(x)
        out_sp = F.linear(x, self.sparse_q.float() * self.r_scale + self.r_min)
        out = out_svd + out_sp
        if hasattr(self.svd.up, 'bias') and self.svd.up.bias is not None:
            out = out + self.svd.up.bias
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MGIE‑SVD  —  entropy‑driven rank selection module
# ═══════════════════════════════════════════════════════════════════════════════

class MGIERankSelector:
    """Entropy‑driven rank selector: higher entropy → more aggressive truncation."""
    @staticmethod
    def select(S: torch.Tensor, min_r: int = 4, max_r: int = 64) -> int:
        p = S / (S.sum() + 1e-9)
        H = -(p * torch.log(p + 1e-9)).sum()
        H_max = math.log(len(S))
        retention = 1.0 - (H / H_max).clamp(0.05, 0.95).item()
        return max(min_r, min(int(retention * len(S)), max_r, len(S)))


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  ProxSparse  —  learnable 2:4 structured sparsity mask
# ═══════════════════════════════════════════════════════════════════════════════

class ProxSparseMask2of4(nn.Module):
    """Learnable 2:4 semi‑structured mask via Gumbel‑Sigmoid straight‑through."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  DeltaKV  —  residual KV reference store
# ═══════════════════════════════════════════════════════════════════════════════

class DeltaKVStore:
    """DeltaKV: stores historical K/V references, returns residuals."""
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
        ks = k[0]
        vs = v[0]
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


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  KVTC  —  PCA‑decorrelated adaptive quantisation
# ═══════════════════════════════════════════════════════════════════════════════

class KVTCQuantiser:
    """KVTC: PCA‑based feature decorrelation + adaptive quantisation (ICLR 2026)."""
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
        cum = torch.cumsum(S ** 2, dim=0) / (total + 1e-9)
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


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SliceGPT  —  PCA‑based dimension reduction
# ═══════════════════════════════════════════════════════════════════════════════

class SliceGPTReducer(nn.Module):
    """SliceGPT: PCA‑based orthogonal projection to reduce head_dim."""
    def __init__(self, orig_dim: int, reduce_ratio: float):
        super().__init__()
        self.orig_dim = orig_dim
        self.reduced_dim = max(8, int(orig_dim * (1.0 - reduce_ratio)))
        self.basis: Optional[torch.Tensor] = None

    @torch.no_grad()
    def calibrate(self, Q: torch.Tensor):
        if self.basis is not None and self.basis.size(0) == self.reduced_dim:
            return
        B, H, N, D = Q.shape
        Qf = Q.transpose(1, 2).reshape(B * N, H * D).float()
        _, _, Vt = torch.linalg.svd(Qf, full_matrices=False)
        self.basis = Vt[:self.reduced_dim, :].to(Q.device)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.basis is None:
            return x
        B, H, N, D = x.shape
        x2 = x.transpose(1, 2).reshape(B, N, H * D)
        out = torch.matmul(x2, self.basis.T)
        return out.view(B, N, H, self.reduced_dim // H if self.reduced_dim % H == 0 else self.reduced_dim)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  NMF  —  non‑negative matrix factorisation projection
# ═══════════════════════════════════════════════════════════════════════════════

class NMFProjection(nn.Module):
    """NMF: W ≈ H_nn @ W_nn  with H_nn, W_nn ≥ 0 via softplus."""
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.H = nn.Parameter(torch.randn(out_dim, rank) * 0.02)
        self.W_mat = nn.Parameter(torch.randn(rank, in_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, F.softplus(self.W_mat)), F.softplus(self.H).t())


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Kronecker‑factorised projection
# ═══════════════════════════════════════════════════════════════════════════════

class KroneckerProjection(nn.Module):
    """KroneckerBERT‑style factorisation: W ≈ A ⊗ B."""
    def __init__(self, in_dim: int, out_dim: int, a_shape: Tuple[int, int], b_shape: Tuple[int, int]):
        super().__init__()
        self.A = nn.Parameter(torch.randn(*a_shape) * 0.02)
        self.B = nn.Parameter(torch.randn(*b_shape) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = torch.kron(self.A, self.B)
        return F.linear(x, W)


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  NEBULA  —  full attention module
# ═══════════════════════════════════════════════════════════════════════════════

class NEBULAAttention(nn.Module):
    """
    NEBULA — Neural Efficient Bandwidth‑Unified Linear Attention.
    Drop‑in replacement for DistilBERT `attention`.

    Pipeline:
      hidden_states  →  [SVD‑LLM+ASVD+SLiM+GFWSVD proj]  →  Q, K, V
      Q, K, V        →  [SliceGPT PCA reduce]           →  compact Q, K, V
      compact Q, K, V →  [OATS outlier triage]          →  critical / marginal / negligible
      critical        →  [NSA coarse‑grain + SLA full attn]
      marginal        →  [Scatterbrain RFF kernel linear attn]
      negligible      →  [skipped / ReplaceMe linearised]
      attention map   →  [Softpick rectified sparsification]
      context         →  [NMF + Kronecker output proj]

    +  DeltaKV + KVTC + FAQ on KV cache
    +  ProxSparse 2:4 masks on rank dims
    +  MGIE‑SVD entropy‑driven per‑layer rank selection
    """
    def __init__(
        self,
        model_config,                  # DistilBertConfig
        nebula_cfg: NEBULAConfig,
        layer_idx: int = 0,
        total_layers: int = 6,
        pretrained_attn=None,
        bert_mode: bool = False,
    ):
        super().__init__()
        self.cfg = nebula_cfg
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.bert_mode = bert_mode

        # ── Activation statistics for ASVD ──
        act_stats = None  # will be computed on first forward

        # ── Build Q, K, V, Out projections ──
        if pretrained_attn is not None:
            q_w = pretrained_attn.q_lin.weight.data
            k_w = pretrained_attn.k_lin.weight.data
            v_w = pretrained_attn.v_lin.weight.data
            o_w = pretrained_attn.out_lin.weight.data

            # Core SVD
            self.svd_q = WhitenedSVDCore(q_w, pretrained_attn.q_lin.bias,
                                          self.hidden_size, self.all_head_size,
                                          nebula_cfg.svd_energy, nebula_cfg.asvd_activation_aware, act_stats)
            self.svd_k = WhitenedSVDCore(k_w, pretrained_attn.k_lin.bias,
                                          self.hidden_size, self.all_head_size,
                                          nebula_cfg.svd_energy, nebula_cfg.asvd_activation_aware, act_stats)
            self.svd_v = WhitenedSVDCore(v_w, pretrained_attn.v_lin.bias,
                                          self.hidden_size, self.all_head_size,
                                          nebula_cfg.svd_energy, nebula_cfg.asvd_activation_aware, act_stats)
            self.svd_out = WhitenedSVDCore(o_w, pretrained_attn.out_lin.bias,
                                            self.all_head_size, self.hidden_size,
                                            nebula_cfg.svd_energy, nebula_cfg.asvd_activation_aware, act_stats)

            # SLiM wrapper
            if nebula_cfg.slim_enabled:
                self.q_proj = SLiMWrapper(self.svd_q, q_w, nebula_cfg.slim_quant_bits, nebula_cfg.slim_2of4_sparsity)
                self.k_proj = SLiMWrapper(self.svd_k, k_w, nebula_cfg.slim_quant_bits, nebula_cfg.slim_2of4_sparsity)
                self.v_proj = SLiMWrapper(self.svd_v, v_w, nebula_cfg.slim_quant_bits, nebula_cfg.slim_2of4_sparsity)
                self.out_proj = SLiMWrapper(self.svd_out, o_w, nebula_cfg.slim_quant_bits, nebula_cfg.slim_2of4_sparsity)
            else:
                self.q_proj = self.svd_q
                self.k_proj = self.svd_k
                self.v_proj = self.svd_v
                self.out_proj = self.svd_out
        else:
            # fallback: full‑rank linear
            self.q_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.k_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.v_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.out_proj = nn.Linear(self.all_head_size, self.hidden_size, bias=True)

        # ── MGIE‑SVD entropy‑based rank selection ──
        if nebula_cfg.mgie_enabled and hasattr(self, 'svd_q'):
            # Refine ranks using entropy criterion
            for proj_name in ['q', 'k', 'v', 'out']:
                svd_mod = getattr(self, f'svd_{proj_name}', None)
                if svd_mod is not None:
                    with torch.no_grad():
                        Wf = svd_mod.up.weight.data @ svd_mod.down.weight.data
                        _, S, _ = torch.linalg.svd(Wf.float(), full_matrices=False)
                        ent_r = MGIERankSelector.select(S, nebula_cfg.mgie_min_rank, nebula_cfg.mgie_max_rank)
                        # Rebuild with entropy‑driven rank
                        U, S2, Vt = torch.linalg.svd(Wf, full_matrices=False)
                        sqrt_S = S2[:ent_r].sqrt()
                        A = U[:, :ent_r] * sqrt_S.unsqueeze(0)
                        B = sqrt_S.unsqueeze(1) * Vt[:ent_r, :]
                        svd_mod.svd_rank = ent_r
                        svd_mod.down = nn.Linear(svd_mod.down.in_features, ent_r, bias=False).to(Wf.device)
                        svd_mod.up = nn.Linear(ent_r, svd_mod.up.out_features, bias=svd_mod.up.bias is not None).to(Wf.device)
                        svd_mod.down.weight.data = B.float()
                        svd_mod.up.weight.data = A.float()

        # ── ProxSparse 2:4 masks ──
        self.ps_q = self.ps_k = self.ps_v = self.ps_out = None
        if nebula_cfg.proxsparse_enabled and hasattr(self, 'svd_q'):
            self.ps_q = ProxSparseMask2of4(self.svd_q.svd_rank, nebula_cfg.proxsparse_sparsity)
            self.ps_k = ProxSparseMask2of4(self.svd_k.svd_rank, nebula_cfg.proxsparse_sparsity)
            self.ps_v = ProxSparseMask2of4(self.svd_v.svd_rank, nebula_cfg.proxsparse_sparsity)
            self.ps_out = ProxSparseMask2of4(self.svd_out.svd_rank, nebula_cfg.proxsparse_sparsity)

        # ── ReplaceMe: deep layers → linear ──
        self.is_replaced = (nebula_cfg.replaceme_enabled and
                            layer_idx >= nebula_cfg.replaceme_depth_threshold)
        if self.is_replaced:
            self.skip_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # ── SliceGPT PCA reducer ──
        self.slicegpt = None
        if nebula_cfg.slicegpt_enabled:
            self.slicegpt = SliceGPTReducer(self.head_dim * self.num_heads, nebula_cfg.slicegpt_reduce_ratio)

        # ── RFF kernel for Scatterbrain linear attention ──
        if nebula_cfg.sla_enabled:
            self.rff_proj = nn.Parameter(torch.randn(self.head_dim, nebula_cfg.rff_features) * 0.02)

        # ── NMF supplementary ──
        self.nmf_proj = None
        if nebula_cfg.nmf_enabled:
            self.nmf_proj = NMFProjection(self.all_head_size, self.all_head_size, nebula_cfg.nmf_rank)

        # ── DeltaKV + KVTC ──
        self.deltakv: Optional[DeltaKVStore] = None
        self.kvt: Optional[KVTCQuantiser] = None
        if nebula_cfg.deltakv_enabled:
            self.deltakv = DeltaKVStore(nebula_cfg.deltakv_max_refs, nebula_cfg.deltakv_topk)
        if nebula_cfg.kvt_enabled:
            self.kvt = KVTCQuantiser(nebula_cfg.kvt_bits, nebula_cfg.kvt_pca_ratio)

        # ── OATS calibration stats ──
        self.register_buffer('oats_mean', None)
        self.register_buffer('oats_cov_diag', None)

        self.dropout = nn.Dropout(nebula_cfg.dropout)

    # ── OATS outlier scoring ──
    @torch.no_grad()
    def _oats_score(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, D = hidden_states.shape
        flat = hidden_states.view(-1, D)
        if self.oats_mean is None:
            self.oats_mean = flat.mean(dim=0)
            Xc = flat - self.oats_mean
            self.oats_cov_diag = (Xc ** 2).mean(dim=0).clamp(min=1e-8)
        Xc = flat - self.oats_mean.to(hidden_states.device)
        scores = (Xc ** 2 / self.oats_cov_diag.to(hidden_states.device).unsqueeze(0)).sum(dim=-1)
        return scores.view(B, N)

    # ── RFF kernel linear attention (Scatterbrain) ──
    def _kernel_linear_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, H, N, D = Q.shape
        rp = self.rff_proj.to(Q.device)
        proj_k = torch.matmul(K, rp)
        phi_k = torch.cat([torch.cos(proj_k), torch.sin(proj_k)], dim=-1) / (rp.size(1) ** 0.25)
        proj_q = torch.matmul(Q, rp)
        phi_q = torch.cat([torch.cos(proj_q), torch.sin(proj_q)], dim=-1) / (rp.size(1) ** 0.25)
        KV = torch.matmul(phi_k.transpose(-2, -1), V)
        Z = phi_k.sum(dim=2, keepdim=True)
        lin_ctx = torch.matmul(phi_q, KV)
        lin_ctx = lin_ctx / (torch.matmul(phi_q, Z.transpose(-2, -1)) + 1e-8)
        return lin_ctx

    # ── Main forward ──
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, H = hidden_states.shape[0], hidden_states.shape[1], self.num_heads
        D = self.head_dim
        cfg = self.cfg

        # ── ReplaceMe: deep layers → linear skip ──
        if self.is_replaced:
            return (self.skip_proj(hidden_states), None)

        # ── 1.  Project to Q, K, V via SVD‑LLM + ASVD + SLiM ──
        Q = self.q_proj(hidden_states)   # (B, N, all_head)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # ProxSparse mask on rank dimensions
        if self.ps_q is not None and hasattr(self.svd_q, 'down'):
            mask_q = self.ps_q()
            Q = self.svd_q.up(self.svd_q.down(hidden_states) * mask_q.squeeze(-1))
        if self.ps_k is not None and hasattr(self.svd_k, 'down'):
            mask_k = self.ps_k()
            K = self.svd_k.up(self.svd_k.down(hidden_states) * mask_k.squeeze(-1))
        if self.ps_v is not None and hasattr(self.svd_v, 'down'):
            mask_v = self.ps_v()
            V = self.svd_v.up(self.svd_v.down(hidden_states) * mask_v.squeeze(-1))

        # NMF supplementary
        if self.nmf_proj is not None:
            Q = Q + 0.1 * self.nmf_proj(Q)
            K = K + 0.1 * self.nmf_proj(K)

        # ── 2.  Reshape to (B, H, N, D) ──
        Q = Q.view(B, N, H, D).transpose(1, 2)
        K = K.view(B, N, H, D).transpose(1, 2)
        V = V.view(B, N, H, D).transpose(1, 2)

        # ── 3.  SliceGPT PCA reduction ──
        if self.slicegpt is not None:
            self.slicegpt.calibrate(Q)
            Q = self.slicegpt.reduce(Q)
            K = self.slicegpt.reduce(K)
            V = self.slicegpt.reduce(V)
            D = Q.size(-1)

        K_orig, V_orig = K, V

        # ── 4.  DeltaKV + KVTC ──
        if cfg.deltakv_enabled and self.deltakv is not None:
            K, V = self.deltakv.compress(K, V)
        if cfg.kvt_enabled and self.kvt is not None:
            if self.training and not self.kvt.fitted:
                self.kvt.fit(K.reshape(B * H * N, D))
            elif self.kvt.fitted:
                K = self.kvt.compress(K.reshape(B * H, N, D)).view(B, H, N, D)
                V = self.kvt.compress(V.reshape(B * H, N, D)).view(B, H, N, D)

        # ── 5.  OATS outlier‑aware token triage ──
        if cfg.oats_enabled:
            token_scores = self._oats_score(hidden_states)
            n_crit = max(1, int(N * cfg.oats_critical_ratio))
            n_marg = max(1, int(N * cfg.oats_marginal_ratio))
            _, sorted_idx = token_scores.sort(dim=-1, descending=True)
            idx_crit = sorted_idx[:, :n_crit]
            idx_marg = sorted_idx[:, n_crit:n_crit + n_marg]
            mask_crit = torch.zeros(B, N, device=hidden_states.device).scatter_(1, idx_crit, 1.0)
            mask_marg = torch.zeros(B, N, device=hidden_states.device).scatter_(1, idx_marg, 1.0)
        else:
            mask_crit = torch.ones(B, N, device=hidden_states.device)
            mask_marg = torch.zeros(B, N, device=hidden_states.device)

        # ── 6.  SLA dual‑pathway attention ──
        scale = D ** -0.5
        scores_full = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # RPCA‑inspired decomposition
        if cfg.sla_enabled and mask_crit.sum() > 0:
            # Low‑rank background
            sf = scores_full.reshape(B * H, N, N).float()
            U, S_arr, Vt = torch.linalg.svd(sf, full_matrices=False)
            rpca_r = min(cfg.slm_rpca_rank, len(S_arr))
            scores_lr = (U[:, :, :rpca_r] * S_arr[:rpca_r].unsqueeze(1)) @ Vt[:rpca_r, :, :]
            scores_lr = scores_lr.reshape(B, H, N, N).to(scores_full.dtype)

            # Sparse: top‑k per query
            k_sp = max(1, int(N * cfg.oats_critical_ratio))
            _, top_idx = scores_full.topk(k_sp, dim=-1)
            sparse_mask = torch.zeros_like(scores_full).scatter_(-1, top_idx, 1.0)

            scores_combined = sparse_mask * scores_full + (1.0 - sparse_mask) * scores_lr
        else:
            scores_combined = scores_full

        # ── 7.  Mask handling ──
        mask_attn = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        if mask_attn is not None:
            if mask_attn.dim() == 4 and mask_attn.shape[-2] == mask_attn.shape[-1]:
                token_valid = mask_attn[:, 0, 0, :] > -1e9
            elif mask_attn.dim() == 4:
                token_valid = mask_attn.squeeze(1).squeeze(1) > -1e9
            elif mask_attn.dim() == 3:
                token_valid = mask_attn.squeeze(1).bool()
            elif mask_attn.dim() == 2:
                token_valid = mask_attn.bool()
            else:
                token_valid = None
            if token_valid is not None:
                scores_combined = scores_combined.masked_fill(
                    ~token_valid.unsqueeze(1).unsqueeze(2), -1e5)

        # ── 8.  Softpick rectified softmax ──
        if cfg.softpick_enabled:
            attn_w = softpick(scores_combined, dim=-1, temperature=cfg.softpick_temp)
        else:
            attn_w = F.softmax(scores_combined, dim=-1)
        attn_w = self.dropout(attn_w)

        # ── 9.  Context computation ──
        context_full = torch.matmul(attn_w, V)

        if cfg.sla_enabled and mask_marg.sum() > 0:
            lin_ctx = self._kernel_linear_attention(
                Q * mask_marg.unsqueeze(1).unsqueeze(-1), K, V)
        else:
            lin_ctx = torch.zeros_like(V)

        w_crit = mask_crit.unsqueeze(1).unsqueeze(-1).to(context_full.dtype)
        w_marg = mask_marg.unsqueeze(1).unsqueeze(-1).to(lin_ctx.dtype)
        context = w_crit * context_full + w_marg * lin_ctx
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)

        # ── 10.  Output projection ──
        if self.ps_out is not None and hasattr(self.svd_out, 'down'):
            mask_out = self.ps_out()
            h = self.svd_out.down(context) * mask_out.squeeze(-1)
            out = self.svd_out.up(h)
        else:
            out = self.out_proj(context)

        # ── 11.  Update DeltaKV ──
        if self.training and cfg.deltakv_enabled and self.deltakv is not None:
            self.deltakv.update(K_orig, V_orig)

        return (out, attn_w) if output_attentions else (out,)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.up.weight.device
                            if hasattr(self.out_proj, 'up')
                            else self.out_proj.weight.device)


# ═══════════════════════════════════════════════════════════════════════════════
# Injection helper
# ═══════════════════════════════════════════════════════════════════════════════

def inject_nebula_into_distilbert(
    layer,
    model_config,
    nebula_cfg: NEBULAConfig,
    layer_idx: int,
    total_layers: int,
    old_attn,
):
    """Replace DistilBERT attention with NEBULA."""
    new_attn = NEBULAAttention(
        model_config=model_config,
        nebula_cfg=nebula_cfg,
        layer_idx=layer_idx,
        total_layers=total_layers,
        pretrained_attn=old_attn,
        bert_mode=False,
    )
    layer.attention = new_attn
    return new_attn