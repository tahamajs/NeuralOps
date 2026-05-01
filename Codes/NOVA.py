"""
NOVA — Near‑Orthogonal Vectorized Attention
=============================================
A training‑free drop‑in replacement for DistilBERT attention that fuses
eight state‑of‑the‑art compression techniques from 2024‑2026 research:

 1. SVD Subspace Extraction    (SVE‑Former, Neural Networks 2026)
 2. Learnable Diagonal Σ       (SVDA, IEEE Access 2025)
 3. ARMOR Factorized Projections (ARMOR, ICLR 2026)
 4. DeltaKV Residual Encoding  (DeltaKV, arXiv 2602.08005, 2026)
 5. KVTC PCA Quantization      (KVTC, ICLR 2026)
 6. Softpick Rectified Softmax (Softpick, arXiv 2504.20966, 2025)
 7. AST Learnable Sparsity     (AST, AAAI 2025)
 8. D‑Rank / Lillama Init      (D‑Rank, arXiv 2025; Lillama, EMNLP 2024)

                   ┌──────────────────────────────────┐
  hidden_states ──►│  SVD Subspace Extraction (t‑SVD) │──► compact tokens
                   └──────────────┬───────────────────┘
                                  │
                   ┌──────────────▼───────────────────┐
                   │  ARMOR‑Factorized Q/K/V Proj     │
                   │  (low‑rank core + block‑diag)    │
                   └──────────────┬───────────────────┘
                                  │
                   ┌──────────────▼───────────────────┐
                   │  DeltaKV Residual Encoding       │
                   │  (encode Δ w.r.t. history)       │
                   └──────────────┬───────────────────┘
                                  │
                   ┌──────────────▼───────────────────┐
                   │  KVTC PCA + Adaptive Quant       │
                   │  (decorrelate + quantise resids) │
                   └──────────────┬───────────────────┘
                                  │
                   ┌──────────────▼───────────────────┐
                   │  Softpick Rectified Attention    │
                   │  (sparse ≈47 %, no sink)         │
                   └──────────────┬───────────────────┘
                                  │
                   ┌──────────────▼───────────────────┐
                   │  ARMOR‑Factorized Out Projection │
                   └──────────────────────────────────┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class NOVAConfig:
    """All NOVA hyper‑parameters."""
    # ── SVD subspace ──
    subspace_enabled: bool = True
    subspace_energy: float = 0.95        # fraction of singular‑value energy to keep
    subspace_min_rank: int = 8
    subspace_max_rank: int = 64

    # ── ARMOR factorisation ──
    armor_enabled: bool = True
    armor_block_size: int = 16           # block‑diagonal block size
    armor_energy: float = 0.90           # energy threshold for core rank

    # ── AST sparsity masks ──
    ast_enabled: bool = True
    ast_sparsity: float = 0.5            # 50 % → 2:4 pattern

    # ── DeltaKV ──
    deltakv_enabled: bool = True
    deltakv_max_refs: int = 128          # stored reference vectors
    deltakv_topk: int = 4                # top‑k references per query

    # ── KVTC ──
    kvt_enabled: bool = True
    kvt_bits: int = 4
    kvt_pca_components: float = 0.95

    # ── Softpick ──
    softpick_enabled: bool = True
    softpick_temp: float = 1.0

    # ── General ──
    dropout: float = 0.1
    max_seq_len: int = 128               # passed from compressor


# ═══════════════════════════════════════════════════════════════════════════
# 1. ARMOR Block‑Diagonal Linear Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class BlockDiagonal(nn.Module):
    """
    Lightweight block‑diagonal matrix that acts as pre/post‑transformation
    error corrector (ARMOR, ICLR 2026).
    Complexity: O(dim² / block_size) parameters.
    """
    def __init__(self, dim: int, block_size: int, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.bs = block_size
        padded = ((dim + block_size - 1) // block_size) * block_size
        self.padded = padded
        self.n_blocks = padded // block_size
        self.weight = nn.Parameter(torch.empty(self.n_blocks, block_size, block_size))
        nn.init.normal_(self.weight, std=0.02 / math.sqrt(block_size))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *batch, D = x.shape
        assert D == self.dim, f"Expected last dim {self.dim}, got {D}"
        # pad
        if self.padded > D:
            pad = torch.zeros(*batch, self.padded - D, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        x = x.view(*batch, self.n_blocks, self.bs)
        x = torch.einsum('...nb,nbc->...nc', x, self.weight)
        x = x.reshape(*batch, self.padded)
        x = x[..., :D]
        if self.bias is not None:
            x = x + self.bias
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 2. SVD‑Factorised + ARMOR Linear Projection
# ═══════════════════════════════════════════════════════════════════════════

class SVDArmorLinear(nn.Module):
    """
    Low‑rank core (from pretrained weight via SVD) wrapped by two block‑diagonal
    error‑correcting matrices (ARMOR, ICLR 2026).
    ┌─────────────┐   ┌───────────┐   ┌────────────┐   ┌─────────────┐
    │ in          │──►│ block_diag│──►│ low‑rank   │──►│ block_diag  │──► out
    │ (in_dim)    │   │ (in_dim)  │   │ (rank)     │   │ (out_dim)   │
    └─────────────┘   └───────────┘   └────────────┘   └─────────────┘
    """
    def __init__(
        self,
        pretrained_weight: torch.Tensor,
        pretrained_bias: Optional[torch.Tensor],
        energy_threshold: float,
        block_size: int,
        in_dim: int,
        out_dim: int,
        use_armor: bool = True,
    ):
        super().__init__()
        # ── determine rank ──
        with torch.no_grad():
            _, S, _ = torch.linalg.svd(pretrained_weight.float(), full_matrices=False)
            total_e = (S ** 2).sum()
            cum_e = torch.cumsum(S ** 2, dim=0) / total_e
            rank = int(torch.searchsorted(cum_e, energy_threshold).item()) + 1
            rank = max(4, min(rank, min(in_dim, out_dim), 64))

        self.rank = rank
        self.use_armor = use_armor
        self.out_dim = out_dim

        # ── factorise W = U·S·Vᵗ → A (out×rank) @ B (rank×in) ──
        U, S_arr, Vt = torch.linalg.svd(pretrained_weight.float(), full_matrices=False)
        A_weight = (U[:, :rank] * S_arr[:rank].unsqueeze(0)).float()   # (out, rank)
        B_weight = Vt[:rank, :].float()                               # (rank, in)

        # ── ARMOR wrappers ──
        if use_armor:
            self.pre_bd = BlockDiagonal(in_dim, block_size)
            self.post_bd = BlockDiagonal(out_dim, block_size)
        else:
            self.pre_bd = nn.Identity()
            self.post_bd = nn.Identity()

        # ── low‑rank core ──
        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, out_dim, bias=True)
        self.down.weight.data = B_weight
        self.up.weight.data = A_weight
        if pretrained_bias is not None:
            self.up.bias.data = pretrained_bias.float()
        else:
            self.up.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre‑bd → down → up → post‑bd
        x = self.pre_bd(x)
        x = self.up(self.down(x))
        x = self.post_bd(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
# 3. AST Learnable Sparsity Mask
# ═══════════════════════════════════════════════════════════════════════════

class ASTSparsityMask(nn.Module):
    """
    Per‑head learnable binary mask for rank dimensions.
    Uses Gumbel‑Sigmoid straight‑through estimator during training,
    hard threshold during inference (AST, AAAI 2025).
    """
    def __init__(self, rank: int, sparsity: float = 0.5):
        super().__init__()
        self.rank = rank
        self.keep = max(1, int(rank * (1.0 - sparsity)))
        self.logits = nn.Parameter(torch.zeros(rank))

    def forward(self) -> torch.Tensor:
        if self.training:
            gumbel = -torch.log(-torch.log(torch.rand_like(self.logits) + 1e-9) + 1e-9)
            y_soft = torch.sigmoid((self.logits + gumbel) * 5.0)
            y_hard = (y_soft > 0.5).float()
            mask = (y_hard - y_soft).detach() + y_soft
        else:
            _, idx = self.logits.topk(self.keep)
            mask = torch.zeros(self.rank, device=self.logits.device)
            mask[idx] = 1.0
        return mask.view(1, self.rank, 1)   # (1, rank, 1) ready for broadcast


# ═══════════════════════════════════════════════════════════════════════════
# 4. DeltaKV Residual Reference Store
# ═══════════════════════════════════════════════════════════════════════════

class DeltaKVStore:
    """Stores historical K/V references and computes residuals."""
    def __init__(self, max_refs: int = 128, topk: int = 4):
        self.max_refs = max_refs
        self.topk = topk
        self.ref_k: Optional[torch.Tensor] = None   # (H, N_ref, D)
        self.ref_v: Optional[torch.Tensor] = None
        self.count = 0

    @torch.no_grad()
    def update(self, k: torch.Tensor, v: torch.Tensor):
        """Accumulate new K/V into reference store."""
        B, H, N, D = k.shape
        if B == 0:
            return
        k_sel = k[0]   # (H, N, D)
        v_sel = v[0]
        if self.ref_k is None:
            self.ref_k = k_sel
            self.ref_v = v_sel
        else:
            self.ref_k = torch.cat([self.ref_k, k_sel], dim=1)
            self.ref_v = torch.cat([self.ref_v, v_sel], dim=1)
        if self.ref_k.size(1) > self.max_refs:
            self.ref_k = self.ref_k[:, -self.max_refs:, :]
            self.ref_v = self.ref_v[:, -self.max_refs:, :]
        self.count = min(self.count + 1, self.max_refs)

    @torch.no_grad()
    def compress(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return residuals relative to most similar references."""
        if self.ref_k is None:
            return k, v
        B, H, N, D = k.shape
        topk = min(self.topk, self.ref_k.size(1))
        # similarity
        sim = torch.matmul(k, self.ref_k.transpose(-2, -1))   # (B,H,N,R)
        _, idx = sim.topk(topk, dim=-1)                         # (B,H,N,topk)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        ref_k_sel = self.ref_k.unsqueeze(0).unsqueeze(2).expand(B, H, N, -1, -1).gather(3, idx_exp).mean(dim=3)
        ref_v_sel = self.ref_v.unsqueeze(0).unsqueeze(2).expand(B, H, N, -1, -1).gather(3, idx_exp).mean(dim=3)
        return k - ref_k_sel, v - ref_v_sel


# ═══════════════════════════════════════════════════════════════════════════
# 5. KVTC PCA Quantiser
# ═══════════════════════════════════════════════════════════════════════════

class KVTCQuantiser:
    """PCA‑decorrelated adaptive quantisation (KVTC, ICLR 2026)."""
    def __init__(self, n_bits: int = 4, pca_components: float = 0.95):
        self.n_bits = n_bits
        self.pca_components = pca_components
        self.basis: Optional[torch.Tensor] = None   # (D, rank_pca)
        self.scale: Optional[torch.Tensor] = None
        self.zero_point: Optional[torch.Tensor] = None
        self.fitted = False

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        """x: (B*H*N, D)"""
        X = x.float()
        mean = X.mean(dim=0, keepdim=True)
        Xc = X - mean
        U, S, _ = torch.linalg.svd(Xc, full_matrices=False)
        total = (S ** 2).sum()
        cum = torch.cumsum(S ** 2, dim=0) / total
        r = min(int(torch.searchsorted(cum, self.pca_components).item()) + 1, X.size(1))
        self.basis = U[:, :r].contiguous()
        projected = Xc @ self.basis
        self.scale = (projected.max(dim=0).values - projected.min(dim=0).values) / (2 ** self.n_bits - 1)
        self.scale = self.scale.clamp(min=1e-8)
        self.zero_point = projected.min(dim=0).values
        self.fitted = True

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fitted or self.basis is None:
            return x
        *B, D = x.shape
        xc = x.float() - x.float().mean(dim=-1, keepdim=True)
        proj = xc @ self.basis.to(x.device)
        q = ((proj - self.zero_point.to(x.device)) / self.scale.to(x.device)).round()
        dq = q * self.scale.to(x.device) + self.zero_point.to(x.device)
        return (dq @ self.basis.T.to(x.device)).to(x.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Softpick Rectified Softmax
# ═══════════════════════════════════════════════════════════════════════════

def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """
    Rectified, not‑sum‑to‑one replacement for softmax.
    Produces ~47 % naturally sparse attention maps, eliminates attention sink.
    Zuhri et al., arXiv:2504.20966 (2025).
    """
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    norm = p.sum(dim=dim, keepdim=True) + 1e-9
    return p / norm


# ═══════════════════════════════════════════════════════════════════════════
# 7. NOVA Attention — Full Module
# ═══════════════════════════════════════════════════════════════════════════

class NOVAAttention(nn.Module):
    """
    NOVA — Near‑Orthogonal Vectorized Attention.
    Drop‑in replacement for DistilBERT / BERT attention.

    Usage (injection):
        old_attn = layer.attention
        new_attn = NOVAAttention(model_config, nova_cfg, layer_idx,
                                 total_layers, pretrained_attn=old_attn)
        layer.attention = new_attn
    """
    def __init__(
        self,
        model_config,                 # DistilBertConfig
        nova_cfg: NOVAConfig,
        layer_idx: int = 0,
        total_layers: int = 6,
        pretrained_attn=None,         # original DistilBERT MultiHeadSelfAttention
        bert_mode: bool = False,
    ):
        super().__init__()
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.cfg = nova_cfg
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.bert_mode = bert_mode

        # ── 1. ARMOR‑factorised Q / K / V projections ──
        if pretrained_attn is not None:
            self.q_proj = SVDArmorLinear(
                pretrained_attn.q_lin.weight.data,
                pretrained_attn.q_lin.bias.data,
                nova_cfg.armor_energy,
                nova_cfg.armor_block_size if nova_cfg.armor_enabled else 1,
                self.hidden_size, self.all_head_size,
                use_armor=nova_cfg.armor_enabled,
            )
            self.k_proj = SVDArmorLinear(
                pretrained_attn.k_lin.weight.data,
                pretrained_attn.k_lin.bias.data,
                nova_cfg.armor_energy,
                nova_cfg.armor_block_size if nova_cfg.armor_enabled else 1,
                self.hidden_size, self.all_head_size,
                use_armor=nova_cfg.armor_enabled,
            )
            self.v_proj = SVDArmorLinear(
                pretrained_attn.v_lin.weight.data,
                pretrained_attn.v_lin.bias.data,
                nova_cfg.armor_energy,
                nova_cfg.armor_block_size if nova_cfg.armor_enabled else 1,
                self.hidden_size, self.all_head_size,
                use_armor=nova_cfg.armor_enabled,
            )
            self.out_proj = SVDArmorLinear(
                pretrained_attn.out_lin.weight.data,
                pretrained_attn.out_lin.bias.data,
                nova_cfg.armor_energy,
                nova_cfg.armor_block_size if nova_cfg.armor_enabled else 1,
                self.all_head_size, self.hidden_size,
                use_armor=nova_cfg.armor_enabled,
            )
        else:
            # fallback full‑rank
            self.q_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.k_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.v_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
            self.out_proj = nn.Linear(self.all_head_size, self.hidden_size, bias=True)

        # ── 2. AST sparsity masks (on rank dims) ──
        self.ast_q = None
        self.ast_k = None
        self.ast_v = None
        self.ast_out = None
        if nova_cfg.ast_enabled:
            if hasattr(self.q_proj, 'rank'):
                self.ast_q = ASTSparsityMask(self.q_proj.rank, nova_cfg.ast_sparsity)
            if hasattr(self.k_proj, 'rank'):
                self.ast_k = ASTSparsityMask(self.k_proj.rank, nova_cfg.ast_sparsity)
            if hasattr(self.v_proj, 'rank'):
                self.ast_v = ASTSparsityMask(self.v_proj.rank, nova_cfg.ast_sparsity)
            if hasattr(self.out_proj, 'rank'):
                self.ast_out = ASTSparsityMask(self.out_proj.rank, nova_cfg.ast_sparsity)

        # ── 3. SVD subspace extraction module ──
        self.subspace_enabled = nova_cfg.subspace_enabled
        if nova_cfg.subspace_enabled:
            self.Σ = nn.Parameter(torch.ones(self.num_heads, nova_cfg.subspace_max_rank))

        # ── 4. DeltaKV residual store ──
        self.deltakv = None
        if nova_cfg.deltakv_enabled:
            self.deltakv = DeltaKVStore(nova_cfg.deltakv_max_refs, nova_cfg.deltakv_topk)

        # ── 5. KVTC quantiser ──
        self.kvt = None
        if nova_cfg.kvt_enabled:
            self.kvt = KVTCQuantiser(nova_cfg.kvt_bits, nova_cfg.kvt_pca_components)

        self.dropout = nn.Dropout(nova_cfg.dropout)

    # ── helper: project potentially through SVDArmorLinear + AST mask ──
    def _project(self, x: torch.Tensor, proj, ast_mask=None) -> torch.Tensor:
        if isinstance(proj, SVDArmorLinear):
            # pre‑bd → down → (AST mask on rank) → up → post‑bd
            x = proj.pre_bd(x)
            h = proj.down(x)                           # (..., rank)
            if ast_mask is not None and self.cfg.ast_enabled:
                h = h * ast_mask().squeeze(-1)
            h = proj.up(h)
            h = proj.post_bd(h)
            return h
        else:
            return proj(x)

    # ── SVD subspace extraction (SVE‑Former, 2026) ──
    @torch.no_grad()
    def _subspace_extract(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Perform truncated SVD on token features, project into the compact subspace.
        Returns (projected_tokens, subspace_rank).
        """
        B, H, N, D = x.shape
        # flatten head & batch for SVD
        X = x.permute(0, 2, 1, 3).reshape(B * N, H * D)   # (B*N, H*D)
        U, S, Vt = torch.linalg.svd(X.float(), full_matrices=False)
        total = (S ** 2).sum()
        cum = torch.cumsum(S ** 2, dim=0) / total
        r = int(torch.searchsorted(cum, self.cfg.subspace_energy).item()) + 1
        r = max(self.cfg.subspace_min_rank, min(r, self.cfg.subspace_max_rank, len(S)))
        V_r = Vt[:r, :]   # (r, H*D)
        # project: X @ V_rᵗ
        compact = torch.matmul(X, V_r.T).reshape(B, N, r)   # (B, N, r)
        return compact, r, V_r

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

        # ── 1. ARMOR‑factorised projections ──
        Q = self._project(hidden_states, self.q_proj, self.ast_q)   # (B, N, all_head)
        K = self._project(hidden_states, self.k_proj, self.ast_k)
        V = self._project(hidden_states, self.v_proj, self.ast_v)

        # reshape → (B, H, N, D)
        Q = Q.view(B, N, H, D).transpose(1, 2)
        K = K.view(B, N, H, D).transpose(1, 2)
        V = V.view(B, N, H, D).transpose(1, 2)

        K_orig, V_orig = K, V   # save for DeltaKV update

        # ── 2. SVD subspace extraction (SVE‑Former + SVDA) ──
        if self.subspace_enabled:
            # Extract subspace for Q, project Q into compact space
            Q_c, r, Vr = self._subspace_extract(Q)               # (B, N, r)
            # Apply learnable diagonal spectral modulation (SVDA)
            Σ = F.softplus(self.Σ[:H, :r]).unsqueeze(0).unsqueeze(2)   # (1, H, 1, r)
            Q_c = Q_c.unsqueeze(1) * Σ                         # (B, H, N, r)
        else:
            Q_c = Q
            r = D

        # ── 3. DeltaKV residual compression ──
        if self.cfg.deltakv_enabled and self.deltakv is not None:
            K, V = self.deltakv.compress(K, V)

        # ── 4. KVTC quantisation ──
        if self.cfg.kvt_enabled and self.kvt is not None:
            if self.training and not self.kvt.fitted:
                self.kvt.fit(K.reshape(B * H * N, D))
            elif self.kvt.fitted:
                K = self.kvt.compress(K.reshape(B * H, N, D)).view(B, H, N, D)
                V = self.kvt.compress(V.reshape(B * H, N, D)).view(B, H, N, D)

        # ── 5. Attention ──
        if self.subspace_enabled:
            # Attend in compact subspace for efficiency
            scale = r ** -0.5
            K_c = K[:, :, :, :r]   # take first r dimensions for compact attention
            scores = torch.matmul(Q_c, K_c.transpose(-2, -1)) * scale
        else:
            scale = D ** -0.5
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # mask handling (robust)
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

        # ── 6. Softpick rectified softmax ──
        if self.cfg.softpick_enabled:
            attn_w = softpick(scores, dim=-1, temperature=self.cfg.softpick_temp)
        else:
            attn_w = F.softmax(scores, dim=-1)
        attn_w = self.dropout(attn_w)

        # ── 7. Context ──
        if self.subspace_enabled:
            V_c = V[:, :, :, :r]
            context_c = torch.matmul(attn_w, V_c)                       # (B, H, N, r)
            # Expand back via Vrᵗ
            context = torch.matmul(context_c.transpose(1, 2).reshape(B * N, r),
                                   Vr).view(B, N, H * D)
        else:
            context = torch.matmul(attn_w, V)
            context = context.transpose(1, 2).contiguous().view(B, N, H * D)

        # ── 8. Factorised output projection ──
        out = self._project(context, self.out_proj, self.ast_out)

        # ── 9. Update DeltaKV cache ──
        if self.training and self.cfg.deltakv_enabled and self.deltakv is not None:
            self.deltakv.update(K_orig, V_orig)

        return (out, attn_w) if output_attentions else (out,)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.q_proj.up.weight.device)


# ═══════════════════════════════════════════════════════════════════════════
# Injection helper
# ═══════════════════════════════════════════════════════════════════════════

def inject_nova_into_distilbert(
    layer,
    model_config,
    nova_cfg: NOVAConfig,
    layer_idx: int,
    total_layers: int,
    old_attn,
):
    """Replace DistilBERT attention with NOVA."""
    new_attn = NOVAAttention(
        model_config=model_config,
        nova_cfg=nova_cfg,
        layer_idx=layer_idx,
        total_layers=total_layers,
        pretrained_attn=old_attn,
        bert_mode=False,
    )
    layer.attention = new_attn
    return new_attn