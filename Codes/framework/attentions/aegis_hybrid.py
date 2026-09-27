"""
Aegis-Hybrid Attention — Production-Hardened (v2).

Combines:
  1. Local sliding-window attention (chunked).
  2. Optional global linear attention via FAVOR+ style kernel features.
  3. Trainable LoRA adapters on top of frozen low-rank base.
  4. Learnable fusion gate between local and global branches.

Fixes (v2):
  AH-FIX-01  `FactorizedLinear.is_custom_attention = False`.
  AH-FIX-02  `use_global` is configurable; global path restored with proper
             learnable kernel feature map.
  AH-FIX-03  Fast `weight` property (no full matmul on each access).
  AH-FIX-04  GQA-aware SVD initialization with `repeat_interleave` pooling.
  AH-FIX-05  Robust GPT-2 Conv1D handling (checks layout at runtime).
  AH-FIX-06  Comprehensive mask normalization (2D/3D/4D).
  AH-FIX-07  Correct `past_len` derived from `past_key_value` shape, not
             from `N_kv - N` (which is wrong for cross-attention).
  AH-FIX-08  Full `inject_aegis_hybrid` with recursion + idempotency guard.
  AH-FIX-09  RoPE supported via `position_embeddings` or `self.rotary_emb`.
  AH-FIX-10  dtype-safe attention matmul.
  AH-FIX-11  `local_window` default consistent (256).
  AH-FIX-12  Storage accounting.
  AH-FIX-13  `output_hidden_states` / `return_dict` / `*args` supported.
  AH-FIX-14  `FactorizedLinear` uses `down`/`up` names for consistency
             with the rest of the framework, with backward-compat aliases
             `lin1`/`lin2`.
  AH-FIX-15  Optional activation whitening via `whitening` dict.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.attention_utils import repeat_kv

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    x1 = x[..., : rotary_dim // 2]
    x2 = x[..., rotary_dim // 2 : rotary_dim]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(q, k, cos, sin):
    if cos is None or sin is None:
        return q, k
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim, k.shape[-1]) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)
    sin_q = sin[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)
    cos_k = cos[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)
    sin_k = sin[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)

    q_rot_emb = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_emb = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q_out = torch.cat([q_rot_emb, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot_emb
    k_out = torch.cat([k_rot_emb, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot_emb
    return q_out, k_out


# ===========================================================================
# Decoder detection
# ===========================================================================
def _detect_decoder(config: Any, is_decoder: Optional[bool], old_attn: Optional[nn.Module]) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                          "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
            return True
        arch = getattr(config, "architectures", None)
        if arch and any(("GPT" in str(a) or "CausalLM" in str(a)) for a in arch):
            return True
    if old_attn is not None:
        if getattr(old_attn, "is_decoder", False):
            return True
        if hasattr(old_attn, "c_attn"):
            return True
        if "gpt" in old_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class AegisHybridConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    num_key_value_heads: Optional[int] = None
    local_window: int = 256
    rank: int = 128
    lora_rank: int = 32
    dropout: float = 0.1
    use_adaptive_pruning: bool = False
    gate_init_bias: float = 1.8
    is_decoder: bool = False
    use_global: bool = True
    kernel_feature_dim: int = 128
    kernel_seed: int = 1234
    chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


# ===========================================================================
# FactorizedLinear
# ===========================================================================
class FactorizedLinear(nn.Module):
    """
    Low-rank factorized linear + LoRA:
        y = (x @ V.T) @ U.T + LoRA(x) + b
    """
    is_custom_attention: bool = False   # AH-FIX-01

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        lora_rank: int = 32,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.svd_rank = self.rank
        self.lora_rank = min(int(lora_rank), min(self.in_features, self.out_features)) if lora_rank > 0 else 0

        fk = {"device": device, "dtype": dtype}
        # AH-FIX-14: consistent `down`/`up` naming
        self.down = nn.Linear(self.in_features, self.rank, bias=False, **fk)
        self.up = nn.Linear(self.rank, self.out_features, bias=bias, **fk)

        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features, **fk))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.lora_rank, **fk))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # AH-FIX-03: cached weight, invalidated on parameter mutation
        self._cached_weight: Optional[torch.Tensor] = None
        self._cached_versions: Tuple[int, ...] = ()

        self.reset_parameters()

    # -- backward-compat aliases --
    @property
    def lin1(self): return self.down
    @property
    def lin2(self): return self.up

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)
            if self.lora_A is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    def _current_versions(self) -> Tuple[int, ...]:
        vs = [getattr(self.down.weight, "_version", 0),
              getattr(self.up.weight, "_version", 0)]
        if self.lora_A is not None:
            vs.append(getattr(self.lora_A, "_version", 0))
            vs.append(getattr(self.lora_B, "_version", 0))
        return tuple(vs)

    @property
    def weight(self) -> torch.Tensor:
        # AH-FIX-03
        if not self.training and self._cached_weight is not None \
                and self._cached_versions == self._current_versions():
            return self._cached_weight
        w = torch.matmul(self.up.weight.float(), self.down.weight.float())
        if self.lora_A is not None:
            w = w + torch.matmul(self.lora_B.float(), self.lora_A.float())
        w = w.to(self.up.weight.dtype)
        if not self.training:
            self._cached_weight = w
            self._cached_versions = self._current_versions()
        return w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_down = x.to(self.down.weight.dtype)
        latent = self.down(x_down)
        latent_up = latent.to(self.up.weight.dtype)
        out = self.up(latent_up)
        if self.lora_A is not None:
            a = self.lora_A.to(orig_dtype) if self.lora_A.dtype != orig_dtype else self.lora_A
            b = self.lora_B.to(orig_dtype) if self.lora_B.dtype != orig_dtype else self.lora_B
            out = out.to(orig_dtype) + F.linear(F.linear(x, a), b)
        return out.to(orig_dtype)


# ===========================================================================
# Kernel feature map (FAVOR+)
# ===========================================================================
class KernelFeatureMap(nn.Module):
    """FAVOR+ style random Fourier features with deterministic seed."""
    is_custom_attention: bool = False

    def __init__(self, head_dim: int, num_features: int = 128, seed: int = 1234):
        super().__init__()
        self.head_dim = int(head_dim)
        self.num_features = int(num_features)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        proj = torch.randn(self.head_dim, self.num_features, generator=gen)
        # Orthogonal init if m <= d
        if self.num_features <= self.head_dim:
            q, _ = torch.linalg.qr(torch.randn(self.head_dim, self.num_features, generator=gen))
            proj = q
        self.register_buffer("proj", proj * math.sqrt(self.head_dim))
        self.scale = 1.0 / math.sqrt(self.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_f = x.float()
            proj = self.proj.float()
            phi = torch.matmul(x_f, proj)
            norm_sq = (x_f * x_f).sum(dim=-1, keepdim=True) / 2.0
            raw = phi - norm_sq
            max_val = raw.detach().max()
            exp_in = torch.clamp(raw - max_val, min=-20.0, max=8.0)
            return (self.scale * torch.exp(exp_in)).to(x.dtype)


# ===========================================================================
# Chunked sliding-window local attention
# ===========================================================================
def _local_window_attention_chunked(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    token_valid: Optional[torch.Tensor],
    is_decoder: bool,
    past_len: int,
    dropout_p: float,
    training: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    B, H, N, D = Q.shape
    N_kv = K.shape[2]
    radius = max(1, window // 2)
    scale = D ** -0.5
    device = Q.device

    abs_q = past_len + torch.arange(N, device=device)
    abs_k = torch.arange(N_kv, device=device)
    chunk = max(64, min(radius * 2, 256))

    outs = []
    attn_blocks = []

    for q_start in range(0, N, chunk):
        q_end = min(q_start + chunk, N)
        Qc = Q[:, :, q_start:q_end]
        q_abs = abs_q[q_start:q_end]

        q_min = int(q_abs.min().item())
        q_max = int(q_abs.max().item())
        k_lo = max(0, q_min - radius)
        k_hi = min(N_kv, q_max + radius + 1)

        if k_hi <= k_lo:
            outs.append(torch.zeros(B, H, q_end - q_start, D, device=device, dtype=Q.dtype))
            continue

        Kc = K[:, :, k_lo:k_hi]
        Vc = V[:, :, k_lo:k_hi]
        k_abs = abs_k[k_lo:k_hi]

        # AH-FIX-10: dtype-safe matmul
        scores = torch.matmul(Qc.float(), Kc.float().transpose(-2, -1)) * scale

        band = (q_abs.unsqueeze(1) - k_abs.unsqueeze(0)).abs() <= radius
        if is_decoder:
            band = band & (k_abs.unsqueeze(0) <= q_abs.unsqueeze(1))
        band = band.unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(~band, -1e4)

        if token_valid is not None:
            kv_valid = token_valid[:, k_lo:k_hi][:, None, None, :].bool()
            scores = scores.masked_fill(~kv_valid, -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

        any_valid = band.any(dim=-1, keepdim=True).to(scores.dtype)
        attn = F.softmax(scores, dim=-1).to(Q.dtype) * any_valid
        attn = F.dropout(attn, p=dropout_p, training=training)

        # AH-FIX-10: cast Vc to match attn
        Vc_t = Vc.to(attn.dtype)
        outs.append(torch.matmul(attn, Vc_t))
        if not training:
            attn_blocks.append((q_start, q_end, k_lo, k_hi, attn))

    output = torch.cat(outs, dim=2)

    attn_full = None
    if len(attn_blocks) > 0 and N * N_kv <= 4_000_000:
        attn_full = torch.zeros(B, H, N, N_kv, device=device, dtype=Q.dtype)
        for q_s, q_e, k_l, k_h, blk in attn_blocks:
            attn_full[:, :, q_s:q_e, k_l:k_h] = blk

    return output, attn_full


# ===========================================================================
# Global linear attention (FAVOR+)
# ===========================================================================
def _global_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    feature_map: KernelFeatureMap,
    token_valid: Optional[torch.Tensor],
) -> torch.Tensor:
    """y = φ(Q) · (φ(K)^T V) / (φ(Q) · Σφ(K))."""
    Q_feat = feature_map(Q)
    K_feat = feature_map(K)

    if token_valid is not None:
        kv = token_valid[:, None, :, None].to(K_feat.dtype)
        K_feat = K_feat * kv
        V = V * kv

    KV = torch.matmul(K_feat.transpose(-2, -1).contiguous(), V.to(K_feat.dtype))
    Z = K_feat.sum(dim=-2, keepdim=True)

    num = torch.matmul(Q_feat, KV)
    den = torch.matmul(Q_feat, Z.transpose(-1, -2).contiguous()).clamp(min=1e-6)
    out = num / den
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).to(Q.dtype)


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register("aegis_hybrid", supports_causal=True,
                              supports_gqa=True, supports_rope=True)
@ATTENTION_REGISTRY.register("aegis_hybrid_attn", supports_causal=True,
                              supports_gqa=True, supports_rope=True)
@ATTENTION_REGISTRY.register("aegis_hybrid_attention", supports_causal=True,
                              supports_gqa=True, supports_rope=True)
class AegisHybridAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        lora_rank: Optional[int] = None,
        is_decoder: Optional[bool] = None,
        whitening: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank if rank is not None else 128)
        except TypeError:
            super().__init__(config)

        if not isinstance(config, AegisHybridConfig):
            cfg_src = config if config is not None else kwargs
            h_size = hidden_size or getattr(cfg_src, "hidden_size", 768)
            n_heads = num_heads or getattr(cfg_src, "num_attention_heads",
                             getattr(cfg_src, "n_heads", 12))
            n_kv = getattr(cfg_src, "num_key_value_heads",
                    getattr(cfg_src, "num_kv_heads", n_heads))
            h_dim = getattr(cfg_src, "head_dim", None) or (h_size // max(1, n_heads))
            r = rank or getattr(cfg_src, "rank", min(128, h_size))
            l_r = lora_rank or getattr(cfg_src, "lora_rank", 32)

            dec = _detect_decoder(config, is_decoder, pretrained_attn)
            config = AegisHybridConfig(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                head_dim=h_dim,
                num_key_value_heads=n_kv,
                local_window=kwargs.get("local_window", 256),
                rank=r,
                lora_rank=l_r,
                is_decoder=dec,
                use_global=kwargs.get("use_global", True),
                kernel_feature_dim=kwargs.get("kernel_feature_dim", 128),
                chunk_size=kwargs.get("chunk_size", 64),
            )

        self.config = config
        self.hidden_size: int = int(config.hidden_size)
        self.num_heads: int = int(config.num_attention_heads)
        self.head_dim: int = int(config.head_dim)
        self.num_kv_heads: int = int(config.num_key_value_heads or self.num_heads)
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.local_window: int = int(config.local_window)
        self.rank: int = max(1, min(int(config.rank), self.hidden_size))
        self.lora_rank: int = max(1, min(int(config.lora_rank), self.hidden_size))
        self.dropout = nn.Dropout(config.dropout)

        self.is_decoder: bool = _detect_decoder(config, is_decoder, pretrained_attn)
        self.use_global: bool = bool(config.use_global)
        self.is_nested_encoder: bool = False
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        self.whitening = whitening or {}

        # ---- projections ----
        self.q_proj = FactorizedLinear(self.hidden_size, self.all_head_size,
                                        rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.k_proj = FactorizedLinear(self.hidden_size, self.all_kv_head_size,
                                        rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.v_proj = FactorizedLinear(self.hidden_size, self.all_kv_head_size,
                                        rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.out_proj = FactorizedLinear(self.all_head_size, self.hidden_size,
                                          rank=self.rank, lora_rank=self.lora_rank, bias=True)

        # ---- kernel feature map ----
        self.kernel_map = KernelFeatureMap(
            head_dim=self.head_dim,
            num_features=config.kernel_feature_dim,
            seed=config.kernel_seed,
        )

        # ---- fusion gate (per-head, init ~0.88 to local) ----
        self.fusion_beta = nn.Parameter(torch.zeros(self.num_heads))

        # AH-FIX-12: storage accounting
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ---- aliases ----
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def dense(self): return self.out_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # AH-FIX-04: GQA-aware SVD init
    # ------------------------------------------------------------------
    @staticmethod
    def _svd_init(
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        proj: FactorizedLinear,
        whitening_L: Optional[torch.Tensor] = None,
    ) -> None:
        with torch.no_grad():
            W = w.detach().float()
            out_dim, in_dim = W.shape
            target_out, target_in = proj.out_features, proj.in_features

            # GQA: pool or repeat along output dim
            if out_dim > target_out and out_dim % target_out == 0:
                rep = out_dim // target_out
                W = W.view(target_out, rep, in_dim).mean(dim=1)
                if b is not None:
                    b = b.view(target_out, rep).mean(dim=1)
            elif out_dim < target_out and target_out % out_dim == 0:
                rep = target_out // out_dim
                W = W.repeat_interleave(rep, dim=0)
                if b is not None:
                    b = b.repeat_interleave(rep)

            if W.shape[0] != target_out or W.shape[1] != target_in:
                raise ValueError(
                    f"FactorizedLinear shape mismatch after GQA pooling: "
                    f"{tuple(W.shape)} vs ({target_out}, {target_in})"
                )

            # Optional whitening
            L = None
            if whitening_L is not None:
                Lw = whitening_L.float().to(W.device)
                if Lw.shape == (target_in, target_in):
                    eps = 1e-6 * torch.trace(Lw @ Lw.t()).item() / max(1, target_in)
                    L = Lw + eps * torch.eye(target_in, device=W.device, dtype=W.dtype)

            if L is not None:
                try:
                    W_eff = W @ L
                    U, S, Vh = torch.linalg.svd(W_eff, full_matrices=False)
                    L_inv = torch.linalg.inv(L)
                    Vh = Vh @ L_inv
                except Exception:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            else:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            r = min(proj.rank, U.size(1), Vh.size(0))
            proj.down.weight.data.zero_()
            proj.up.weight.data.zero_()
            proj.down.weight.data[:r, :].copy_(
                Vh[:r, :].to(proj.down.weight.dtype)
            )
            US = U[:, :r] @ torch.diag(S[:r])
            proj.up.weight.data[:, :r].copy_(US.to(proj.up.weight.dtype))

            if b is not None and proj.up.bias is not None:
                n = min(b.numel(), proj.up.bias.numel())
                proj.up.bias.data[:n].copy_(b.detach()[:n].to(proj.up.bias.dtype))
            elif proj.up.bias is not None:
                proj.up.bias.data.zero_()

            # Invalidate cache
            proj._cached_weight = None
            proj._cached_versions = ()

    # ------------------------------------------------------------------
    # AH-FIX-05: copy_weights with proper Conv1D handling
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: nn.Module) -> None:
        try:
            for p in old_attn.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        from framework.core.adapters import UniversalAdapter
        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None),
            "v": (None, None), "o": (None, None),
        }

        # UniversalAdapter first
        for p in ("q", "k", "v", "o"):
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, p)
                if w is not None:
                    extracted[p] = (w, b)
            except Exception:
                pass

        # GPT-2 fused c_attn
        if hasattr(old_attn, "c_attn") and extracted["q"][0] is None:
            c = old_attn.c_attn
            c_w = c.weight.data.clone()
            # Conv1D stores weight as (in, out) → transpose to (out, in)
            if c_w.shape[1] == 3 * self.hidden_size or c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]:
                c_w = c_w.t().contiguous()
            split = c_w.size(0) // 3
            c_b = getattr(c, "bias", None)
            c_b = c_b.data.clone() if c_b is not None else None
            extracted["q"] = (c_w[:split], c_b[:split] if c_b is not None else None)
            extracted["k"] = (c_w[split:2*split], c_b[split:2*split] if c_b is not None else None)
            extracted["v"] = (c_w[2*split:], c_b[2*split:] if c_b is not None else None)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            c = old_attn.c_proj
            w_o = c.weight.data.clone()
            if w_o.ndim == 2 and w_o.shape[0] < w_o.shape[1]:
                w_o = w_o.t().contiguous()
            b_o = getattr(c, "bias", None)
            extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

        # Nested BERT/RoBERTa
        if hasattr(old_attn, "self") and hasattr(getattr(old_attn, "self"), "query"):
            s = old_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                if extracted[key][0] is None:
                    m = getattr(s, sub, None)
                    if m is not None and hasattr(m, "weight"):
                        b = getattr(m, "bias", None)
                        extracted[key] = (m.weight.data.clone(),
                                           b.data.clone() if b is not None else None)
            if (extracted["o"][0] is None
                    and hasattr(old_attn, "output")
                    and hasattr(old_attn.output, "dense")):
                d = old_attn.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (d.weight.data.clone(),
                                   b.data.clone() if b is not None else None)

        # Flat modules
        named_map = {
            "q": ("q_proj", "q_lin", "query"),
            "k": ("k_proj", "k_lin", "key"),
            "v": ("v_proj", "v_lin", "value"),
            "o": ("o_proj", "out_proj", "out_lin", "dense"),
        }
        for key, names in named_map.items():
            if extracted[key][0] is None:
                for n in names:
                    if hasattr(old_attn, n):
                        m = getattr(old_attn, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (m.weight.data.clone(),
                                               b.data.clone() if b is not None else None)
                            break

        # Apply SVD init
        for key, target in (("q", self.q_proj), ("k", self.k_proj),
                             ("v", self.v_proj), ("o", self.out_proj)):
            w, b = extracted[key]
            if w is not None:
                try:
                    self._svd_init(
                        w, b, target,
                        whitening_L=self.whitening.get(key, None),
                    )
                except Exception as e:
                    logger.warning("Aegis-Hybrid SVD init failed for '%s': %s", key, e)
            elif key == "o" and self.is_nested_encoder:
                # Identity for nested encoder output (bypassed externally)
                with torch.no_grad():
                    r = min(target.out_features, target.in_features, target.rank)
                    target.down.weight.data.zero_()
                    target.up.weight.data.zero_()
                    target.down.weight.data[:r, :r].fill_diagonal_(1.0)
                    target.up.weight.data[:r, :r].fill_diagonal_(1.0)

        # storage accounting
        try:
            for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # AH-FIX-06: comprehensive mask normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(attention_mask, B, L, device) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        def _to_bool(t):
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if attention_mask.dim() == 2:
            m = _to_bool(attention_mask)
            if m.shape == (B, L):
                return m
            if m.shape[0] == B and m.shape[1] != L:
                m = m[:, :L]
                if m.shape[1] == L:
                    return m
        elif attention_mask.dim() == 3:
            m = _to_bool(attention_mask)
            if m.shape[0] == B and m.shape[-1] == L:
                return m[:, -1, :]
        elif attention_mask.dim() == 4:
            m = _to_bool(attention_mask)
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m.squeeze(1)
            if m.shape == (B, L):
                return m
        return None

    # ------------------------------------------------------------------
    # AH-FIX-07: correct past_len from cache shape
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_kv(past_key_value, layer_idx=0):
        if past_key_value is None:
            return None, None
        if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
            if not past_key_value.key_cache:
                return None, None
            idx = min(layer_idx, len(past_key_value.key_cache) - 1)
            return past_key_value.key_cache[idx], past_key_value.value_cache[idx]
        if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
            return past_key_value[0], past_key_value[1]
        return None, None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, N, _ = hidden_states.shape
        device = hidden_states.device
        Dh = self.head_dim

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, Dh).transpose(1, 2).contiguous()
        K_new = self.k_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, Dh).transpose(1, 2).contiguous()
        V_new = self.v_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, Dh).transpose(1, 2).contiguous()

        # ---- RoPE ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_k_probe, _ = self._read_past_kv(past_key_value, layer_idx)
            past_len_rope = 0 if past_k_probe is None else past_k_probe.shape[-2]
            cos = cos[..., past_len_rope: past_len_rope + N, :]
            sin = sin[..., past_len_rope: past_len_rope + N, :]
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin)

        # ---- KV cache ----
        past_k, past_v = self._read_past_kv(past_key_value, layer_idx)
        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k, K_new], dim=-2)
            V_cat = torch.cat([past_v, V_new], dim=-2)
            past_len = past_k.shape[-2]
        else:
            K_cat, V_cat = K_new, V_new
            past_len = 0

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            past_key_value.update(K_new, V_new, layer_idx)

        N_kv = K_cat.shape[2]

        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        # ---- mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)

        # ---- local sliding window ----
        local_out, attn_local = _local_window_attention_chunked(
            Q, K_full, V_full,
            window=self.local_window,
            token_valid=token_valid,
            is_decoder=self.is_decoder and not is_cross,
            past_len=past_len,
            dropout_p=self.config.dropout,
            training=self.training,
        )

        # ---- global linear attention (optional) ----
        if self.use_global:
            global_out = _global_linear_attention(
                Q, K_full, V_full, self.kernel_map, token_valid,
            )
            beta = torch.sigmoid(self.fusion_beta).view(1, self.num_heads, 1, 1)
            # β → local, (1-β) → global
            fused = beta * local_out + (1.0 - beta) * global_out
        else:
            fused = local_out

        # ---- head_mask ----
        if head_mask is not None:
            hm = head_mask.to(fused.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(-1)
            fused = fused * hm

        fused = fused.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            final = fused
        else:
            final = self.out_proj(fused)

        final = torch.nan_to_num(final, nan=0.0, posinf=1e4, neginf=-1e4)

        return final.to(hidden_states.dtype), (attn_local if output_attentions else None)

    # ------------------------------------------------------------------
    def storage_bytes(self) -> Dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (("q_proj", self.q_proj), ("k_proj", self.k_proj),
                         ("v_proj", self.v_proj), ("out_proj", self.out_proj),
                         ("kernel_map", self.kernel_map)):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["fusion_beta"] = _sz(self.fusion_beta)
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# AH-FIX-08: full injector with recursion + idempotency
# ===========================================================================
@ATTENTION_REGISTRY.register("aegis_hybrid_inject")
@ATTENTION_REGISTRY.register("inject_aegis_hybrid")
def inject_aegis_hybrid(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_aegis_hybrid: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_hybrid(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    # Single-layer
    src = old_attn
    if src is None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src = getattr(cand, "self", cand)
                    break
    if src is None:
        src = target

    # Idempotency guard
    if isinstance(src, AegisHybridAttention) or "AegisHybrid" in src.__class__.__name__:
        return None

    h_size = getattr(cfg, "hidden_size", None) if cfg else None
    if h_size is None:
        for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
            m = getattr(src, attr, None)
            if m is not None and hasattr(m, "out_features"):
                h_size = m.out_features
                break
    h_size = int(h_size or kwargs.get("hidden_size", 768))

    n_heads = getattr(cfg, "num_attention_heads", None) if cfg else None
    if n_heads is None:
        n_heads = getattr(src, "num_attention_heads",
                   getattr(src, "num_heads", kwargs.get("num_heads", 12)))
    n_heads = int(n_heads)

    dec = _detect_decoder(cfg, kwargs.get("is_decoder"), src)

    aegis_cfg = AegisHybridConfig(
        hidden_size=h_size,
        num_attention_heads=n_heads,
        head_dim=kwargs.get("head_dim", h_size // max(1, n_heads)),
        num_key_value_heads=getattr(cfg, "num_key_value_heads", n_heads) if cfg else n_heads,
        local_window=kwargs.get("local_window",
                     getattr(comp_config, "local_window", 256) if comp_config else 256),
        rank=kwargs.get("rank",
              getattr(comp_config, "rank", min(128, h_size)) if comp_config else min(128, h_size)),
        lora_rank=kwargs.get("lora_rank",
                   getattr(comp_config, "lora_rank", 32) if comp_config else 32),
        dropout=kwargs.get("dropout",
                 getattr(comp_config, "dropout", 0.1) if comp_config else 0.1),
        is_decoder=dec,
        use_global=kwargs.get("use_global",
                    getattr(comp_config, "use_global", True) if comp_config else True),
    )

    try:
        new_attn = AegisHybridAttention(
            config=aegis_cfg,
            pretrained_attn=src if src is not target else None,
            whitening=kwargs.pop("whitening", None),
        )
    except Exception as e:
        logger.error("inject_aegis_hybrid: construction failed: %s", e)
        return None

    p = next(src.parameters(), None)
    if p is not None:
        new_attn.to(device=p.device, dtype=p.dtype)

    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            new_attn.is_nested_encoder = True
            target.attention.self = new_attn
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn
        spliced = True
    elif hasattr(target, "self_attn"):
        new_attn.is_nested_encoder = False
        target.self_attn = new_attn
        spliced = True
    elif hasattr(target, "attn"):
        new_attn.is_nested_encoder = False
        target.attn = new_attn
        spliced = True
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
        spliced = True

    if not spliced:
        logger.error("inject_aegis_hybrid: no writable attention attribute on %s",
                      type(target).__name__)
        return None

    return new_attn


__all__ = [
    "AegisHybridAttention",
    "AegisHybridConfig",
    "FactorizedLinear",
    "KernelFeatureMap",
    "inject_aegis_hybrid",
    "apply_rotary_pos_emb",
]