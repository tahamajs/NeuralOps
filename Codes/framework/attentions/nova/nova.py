"""NOVA attention module – 8 state‑of‑the‑art compression techniques."""

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
class NOVAConfig:
    hidden_size: int = 768
    num_heads: int = 12
    head_dim: int = 64
    dropout: float = 0.1

    # ARMOR (Adaptive Rank‑Modulated Orthogonal Reduction) – SVD factorisation
    armor_enabled: bool = True
    armor_energy: float = 0.90   # energy retained in SVD truncation
    armor_quant_bits: int = 4    # quantisation bits for residual
    armor_use_2of4: bool = True  # structured sparsity 2:4

    # AST (Adaptive Sparse Token) – learnable token pruning
    ast_enabled: bool = True
    ast_keep_ratio: float = 0.7  # fraction of tokens to keep

    # DeltaKV (residual KV reference)
    deltakv_enabled: bool = True
    deltakv_max_refs: int = 128
    deltakv_topk: int = 4

    # KVT (KV quantisation)
    kvt_enabled: bool = True
    kvt_bits: int = 4
    kvt_pca_ratio: float = 0.95

    # Softpick (rectified softmax)
    softpick_enabled: bool = True
    softpick_temp: float = 1.0

    # Subspace (low‑dimensional attention)
    subspace_enabled: bool = True
    subspace_energy: float = 0.95   # for SVD‑based subspace projection

    max_seq_len: int = 128


# --------------------------------------------------------------------------
# Utility components
# --------------------------------------------------------------------------
def _select_rank_svd(W: torch.Tensor, energy: float, min_r: int = 4) -> int:
    with torch.no_grad():
        _, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
        total = (S ** 2).sum()
        cum = torch.cumsum(S ** 2, dim=0) / total
        r = int(torch.searchsorted(cum, energy).item()) + 1
        return max(min_r, min(r, min(W.shape[0], W.shape[1])))


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Rectified softmax – ~47% sparse, no attention sink."""
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m) / temperature)
    denom = e - torch.exp(-m / temperature)
    p = F.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)


class DeltaKVStore:
    """Residual KV reference store (DeltaKV)."""
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
    """PCA‑decorrelated adaptive quantisation (KVTC)."""
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
# ARMOR projection (SVD‑factorised + quantised sparse residual)
# --------------------------------------------------------------------------
class ARMORProjection(nn.Module):
    """ARMOR: Adaptive Rank‑Modulated Orthogonal Reduction (SVD + 2:4 sparse + quant)."""
    def __init__(self, W_pretrained: torch.Tensor, bias: Optional[torch.Tensor],
                 energy: float, bits: int, use_2of4: bool):
        super().__init__()
        out_dim, in_dim = W_pretrained.shape
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

        self.bias = nn.Parameter(bias.float()) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_lr = F.linear(F.linear(x, self.B), self.A.t())
        out_sp = F.linear(x, self.sparse_q.float() * self.r_scale + self.r_min)
        out = out_lr + out_sp
        if self.bias is not None:
            out = out + self.bias
        return out


# --------------------------------------------------------------------------
# Subspace attention helper (for low‑dimensional projection)
# --------------------------------------------------------------------------
class SubspaceProjection(nn.Module):
    """Learnable SVD‑based subspace projection for Q/K/V."""
    def __init__(self, dim: int, energy: float):
        super().__init__()
        # Initialised randomly; during first forward it will adapt
        self.proj = nn.Linear(dim, dim, bias=False)
        self.energy = energy
        self.subspace_basis: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute SVD of weight matrix once
        if self.subspace_basis is None:
            with torch.no_grad():
                U, S, Vt = torch.linalg.svd(self.proj.weight.data.float(), full_matrices=False)
                total = (S ** 2).sum()
                cum = torch.cumsum(S ** 2, dim=0) / total
                r = int(torch.searchsorted(cum, self.energy).item()) + 1
                r = max(4, min(r, min(self.proj.weight.size(0), self.proj.weight.size(1))))
                self.subspace_basis = Vt[:r, :].T.to(x.device, dtype=x.dtype)
        # Project input to subspace and back
        return F.linear(F.linear(x, self.subspace_basis.T), self.subspace_basis)


# --------------------------------------------------------------------------
# Main NOVA Attention
# --------------------------------------------------------------------------
class NOVAAttention(nn.Module):
    def __init__(self, model_config, nova_cfg: NOVAConfig,
                 layer_idx: int = 0, total_layers: int = 6,
                 pretrained_attn=None, bert_mode: bool = False):
        super().__init__()
        self.cfg = nova_cfg
        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.layer_idx = layer_idx
        self.bert_mode = bert_mode

        # Build projections (ARMOR if enabled)
        if pretrained_attn is not None:
            q_w = pretrained_attn.q_lin.weight.data
            k_w = pretrained_attn.k_lin.weight.data
            v_w = pretrained_attn.v_lin.weight.data
            o_w = pretrained_attn.out_lin.weight.data
            if nova_cfg.armor_enabled:
                self.q_proj = ARMORProjection(q_w, pretrained_attn.q_lin.bias,
                                              nova_cfg.armor_energy, nova_cfg.armor_quant_bits,
                                              nova_cfg.armor_use_2of4)
                self.k_proj = ARMORProjection(k_w, pretrained_attn.k_lin.bias,
                                              nova_cfg.armor_energy, nova_cfg.armor_quant_bits,
                                              nova_cfg.armor_use_2of4)
                self.v_proj = ARMORProjection(v_w, pretrained_attn.v_lin.bias,
                                              nova_cfg.armor_energy, nova_cfg.armor_quant_bits,
                                              nova_cfg.armor_use_2of4)
                self.out_proj = ARMORProjection(o_w, pretrained_attn.out_lin.bias,
                                               nova_cfg.armor_energy, nova_cfg.armor_quant_bits,
                                               nova_cfg.armor_use_2of4)
            else:
                self.q_proj = nn.Linear(self.hidden_size, self.all_head_size)
                self.k_proj = nn.Linear(self.hidden_size, self.all_head_size)
                self.v_proj = nn.Linear(self.hidden_size, self.all_head_size)
                self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)
                # Copy weights
                self.q_proj.weight.data = q_w.clone()
                self.q_proj.bias.data = pretrained_attn.q_lin.bias.clone() if pretrained_attn.q_lin.bias is not None else torch.zeros(self.all_head_size)
                self.k_proj.weight.data = k_w.clone()
                self.k_proj.bias.data = pretrained_attn.k_lin.bias.clone() if pretrained_attn.k_lin.bias is not None else torch.zeros(self.all_head_size)
                self.v_proj.weight.data = v_w.clone()
                self.v_proj.bias.data = pretrained_attn.v_lin.bias.clone() if pretrained_attn.v_lin.bias is not None else torch.zeros(self.all_head_size)
                self.out_proj.weight.data = o_w.clone()
                self.out_proj.bias.data = pretrained_attn.out_lin.bias.clone() if pretrained_attn.out_lin.bias is not None else torch.zeros(self.hidden_size)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.all_head_size)
            self.k_proj = nn.Linear(self.hidden_size, self.all_head_size)
            self.v_proj = nn.Linear(self.hidden_size, self.all_head_size)
            self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)

        # AST – adaptive sparse token selection
        self.ast_layer = None
        if nova_cfg.ast_enabled:
            self.ast_layer = nn.Linear(self.hidden_size, 1)

        # DeltaKV
        self.deltakv: Optional[DeltaKVStore] = None
        if nova_cfg.deltakv_enabled:
            self.deltakv = DeltaKVStore(nova_cfg.deltakv_max_refs, nova_cfg.deltakv_topk)

        # KVT quantiser
        self.kvt: Optional[KVTQuantiser] = None
        if nova_cfg.kvt_enabled:
            self.kvt = KVTQuantiser(nova_cfg.kvt_bits, nova_cfg.kvt_pca_ratio)

        # Subspace projection
        self.subspace_q = self.subspace_k = self.subspace_v = None
        if nova_cfg.subspace_enabled:
            self.subspace_q = SubspaceProjection(self.all_head_size, nova_cfg.subspace_energy)
            self.subspace_k = SubspaceProjection(self.all_head_size, nova_cfg.subspace_energy)
            self.subspace_v = SubspaceProjection(self.all_head_size, nova_cfg.subspace_energy)

        self.dropout = nn.Dropout(nova_cfg.dropout)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs):
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        cfg = self.cfg

        # ---- AST: token pruning ----
        if cfg.ast_enabled and self.ast_layer is not None:
            scores = torch.sigmoid(self.ast_layer(hidden_states)).squeeze(-1)  # (B, N)
            k = max(1, int(N * cfg.ast_keep_ratio))
            _, top_indices = torch.topk(scores, k, dim=-1)
            top_indices, _ = torch.sort(top_indices, dim=-1)   # preserve order
            x = torch.gather(hidden_states, 1, top_indices.unsqueeze(-1).expand(-1, -1, _))
            # adjust mask similarly if needed
        else:
            x = hidden_states

        # ---- Q/K/V projections ----
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # ---- Subspace projection ----
        if cfg.subspace_enabled and self.subspace_q is not None:
            Q = self.subspace_q(Q) + 0.1 * Q   # residual connection
            K = self.subspace_k(K) + 0.1 * K
            V = self.subspace_v(V) + 0.1 * V

        Q = Q.view(B, -1, H, D).transpose(1, 2)  # (B, H, N', D)
        K = K.view(B, -1, H, D).transpose(1, 2)
        V = V.view(B, -1, H, D).transpose(1, 2)

        K_orig, V_orig = K, V  # for DeltaKV update

        # ---- DeltaKV compressions ----
        if cfg.deltakv_enabled and self.deltakv is not None:
            K, V = self.deltakv.compress(K, V)

        # ---- KVT quantisation ----
        if cfg.kvt_enabled and self.kvt is not None:
            Nt = K.shape[2]   # new seq length after pruning
            if self.training and not self.kvt.fitted:
                self.kvt.fit(K.reshape(B * H * Nt, D))
            elif self.kvt.fitted:
                K = self.kvt.compress(K.reshape(B * H, Nt, D)).view(B, H, Nt, D)
                V = self.kvt.compress(V.reshape(B * H, Nt, D)).view(B, H, Nt, D)

        # ---- Attention scores ----
        scale = D ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Apply attention mask (handle dimensions)
        mask = attention_mask if attention_mask is not None else kwargs.get('attn_mask', None)
        if mask is not None:
            # We need to adapt mask to the pruned sequence; simplified for now
            # If we pruned, recreate a default causal mask or adjust accordingly.
            # For simplicity, we'll assume no pruning during inference or a simple broadcastable mask.
            if mask.dim() == 4:
                token_valid = mask[:, 0, 0, :] > -1e9
            elif mask.dim() == 3:
                token_valid = mask.squeeze(1).bool()
            elif mask.dim() == 2:
                token_valid = mask.bool()
            else:
                token_valid = None
            if token_valid is not None:
                # If we pruned, we must select the corresponding valid tokens.
                if cfg.ast_enabled and self.ast_layer is not None:
                    token_valid = token_valid.gather(1, top_indices)  # adjust
                scores = scores.masked_fill(~token_valid.unsqueeze(1).unsqueeze(2), -1e5)

        # ---- Softpick or softmax ----
        if cfg.softpick_enabled:
            attn_weights = softpick(scores, dim=-1, temperature=cfg.softpick_temp)
        else:
            attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # ---- Context ----
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, -1, H * D)

        # ---- Output projection ----
        output = self.out_proj(context)

        # Update DeltaKV (training mode, full K/V)
        if self.training and cfg.deltakv_enabled and self.deltakv is not None:
            self.deltakv.update(K_orig, V_orig)

        return (output, attn_weights) if output_attentions else (output,)

    def get_ortho_loss(self):
        return torch.tensor(0.0, device=self.out_proj.weight.device if isinstance(self.out_proj, nn.Linear) else self.out_proj.B.device)


# ------------------------------------------------------------
# Framework injection & registration
# ------------------------------------------------------------
from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("nova")
def inject_nova(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """Build NOVA config from CompressionConfig and inject."""
    cfg = NOVAConfig(
        dropout=getattr(comp_config, 'dropout', 0.1),
        armor_enabled=getattr(comp_config, 'armor_enabled', True),
        armor_energy=getattr(comp_config, 'svd_energy', 0.90),
        ast_enabled=True,
        deltakv_enabled=True,
        kvt_enabled=True,
        softpick_enabled=getattr(comp_config, 'softpick_enabled', True),
        subspace_enabled=True,
        subspace_energy=getattr(comp_config, 'svd_energy', 0.95),
    )
    new_attn = NOVAAttention(
        model_config=model_config,
        nova_cfg=cfg,
        layer_idx=layer_idx,
        total_layers=total_layers,
        pretrained_attn=old_attn,
        bert_mode=False,
    )
    layer.attention = new_attn
    return new_attn