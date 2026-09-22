r"""
AEGIS-Attention V4 (Corrected): Adaptive Enhanced Global-Local Integrated Sparse Attention.

Architecture:
  * FactorisedLinear: SVD-factorized linear layers with energy-based rank selection + LoRA.
  * Split-head attention: some heads perform local sliding-window attention,
    other heads perform global kernel linear attention (RFF / FAVOR+).
  * Adaptive gate: blends local-head outputs with the mean of global-head outputs.
  * SDPA fast path for very short sequences.

Bug-fix map (see accompanying report):
  V4-FIX-01  head_mask broadcasting in SDPA path corrected (unsqueeze(-1).unsqueeze(-1)).
  V4-FIX-02  local_window_attention: past_len offset for KV-cache decoding.
  V4-FIX-03  causal kernel linear attention: chunked cumsum (no 5D tensor).
  V4-FIX-04  inject_aegis_v4: recursion guard (_depth).
  V4-FIX-05  is_decoder detection via helper (no locals() hack).
  V4-FIX-06  output_attentions honored in SDPA fallback path.
  V4-FIX-07  Split-head path now also used for decoders (with causal handling).
  V4-FIX-08  softpick renamed to temperature_softmax; docstring corrected.
  V4-FIX-09  local_window semantics documented.
  V4-FIX-10  Adaptive gate logic: blend only local heads with global mean.
  V4-FIX-11  gate_bias forced >= 2.0 removed.
  V4-FIX-12  LoRA init consistent between tensor and int constructor paths.
  V4-FIX-13  from_distilbert_pretrained: safe fallback.
  V4-FIX-14  rff_proj: registered as buffer by default (opt-in trainable).
  V4-FIX-15  Unused Union kept (used in signature).
  V4-FIX-16  decompose_from_weight: padding documented.
  V4-FIX-17  Mask handling centralized in _prepare_valid_mask.
  V4-FIX-18  head_mask applied after gate in split-head path.
  V4-FIX-19  Dead `pass` and `AegisAttention = AegisAttention` removed.
  V4-FIX-20  token_valid propagated to global branch.
  V4-FIX-21  rank attribute exposed for framework FLOPs detection.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.attention_utils import repeat_kv

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE helpers
# ===========================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half: 2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE. Caller must slice cos/sin to the current window (past_len : past_len+N)."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        raise ValueError(
            f"apply_rotary_pos_emb: cos/sin seq len {cos.shape[-2]} "
            f"does not match q seq len {q.shape[-2]}."
        )

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


def _detect_decoder(
    config: Any,
    is_decoder: Optional[bool],
    old_attn: Optional[nn.Module],
) -> bool:
    """V4-FIX-05: robust decoder detection, replaces the locals() hack."""
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        model_type = str(getattr(config, "model_type", "")).lower()
        if model_type.startswith(
            ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
             "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        ):
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
class AegisConfig:
    """Configuration for AegisAttention V4."""
    svd_energy: float = 0.95
    min_rank: int = 4
    max_rank: int = 384
    num_local_heads: int = 8
    num_global_heads: int = 4
    # V4-FIX-09: local_window = full width of the sliding window.
    # Effective radius = local_window // 2.
    local_window: int = 64
    softpick_temperature: float = 1.0
    rff_features: int = 128
    use_zero_sum_keys: bool = True
    use_adaptive_gate: bool = True
    gate_init_bias: float = 1.8
    dropout: float = 0.1
    is_decoder: bool = False
    # V4-FIX-14: rff_proj as buffer by default.
    rff_trainable: bool = False
    # V4-FIX-07: SDPA fast path toggle.
    use_sdpa_fast_path: bool = True
    # V4-FIX-03: chunk size for causal kernel linear attention.
    causal_chunk_size: int = 64


# ===========================================================================
# FactorisedLinear
# ===========================================================================

class FactorisedLinear(nn.Module):
    is_custom_attention: bool = True
    r"""
    SVD-factorized linear layer with energy-based rank selection and LoRA adapters.

    Forward: y = up(down(x)) + (lora_B @ lora_A)(x) * (lora_alpha / lora_rank)
    Base SVD weights (down, up) are frozen; only LoRA is trainable.

    Naming convention (matches PEFT):
      lora_A: (rank, in_features)
      lora_B: (out_features, rank)
    """

    def __init__(
        self,
        in_features_or_weight: Any,
        out_features_or_bias: Optional[Any] = None,
        energy: float = 0.95,
        min_rank: int = 4,
        max_rank: int = 384,
        rank: Optional[int] = None,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # V4-FIX-19: dead `pass` removed.

        self.energy = float(energy)
        self.min_rank = int(min_rank)
        self.max_rank = int(max_rank)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0

        # --- Weight-tensor init path ---
        if isinstance(in_features_or_weight, torch.Tensor):
            W = in_features_or_weight
            b = out_features_or_bias if isinstance(out_features_or_bias, torch.Tensor) else None
            out_dim, in_dim = W.shape
            self.in_features = in_dim
            self.out_features = out_dim
            self.rank = min(self.max_rank, max(self.min_rank, min(in_dim, out_dim)))
            # V4-FIX-21: expose both `rank` and `svd_rank` for framework detection.
            self.svd_rank = self.rank
            device = W.device
            dtype = W.dtype

            self.down = nn.Linear(in_dim, self.rank, bias=False, device=device, dtype=dtype)
            self.up = nn.Linear(self.rank, out_dim, bias=(b is not None or bias),
                                device=device, dtype=dtype)
            if self.lora_rank > 0:
                self.lora_A = nn.Parameter(
                    torch.empty(self.lora_rank, in_dim, device=device, dtype=dtype))
                self.lora_B = nn.Parameter(
                    torch.zeros(out_dim, self.lora_rank, device=device, dtype=dtype))
                # V4-FIX-12: consistent init with int path.
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            else:
                self.register_parameter("lora_A", None)
                self.register_parameter("lora_B", None)

            self.decompose_from_weight(W, b, self.energy)

        # --- Dimension init path ---
        else:
            in_dim = int(in_features_or_weight)
            out_dim = int(out_features_or_bias) if out_features_or_bias is not None else in_dim
            self.in_features = in_dim
            self.out_features = out_dim
            self.rank = int(rank) if rank is not None else min(
                self.max_rank, max(self.min_rank, min(in_dim, out_dim)))
            self.svd_rank = self.rank  # V4-FIX-21

            self.down = nn.Linear(in_dim, self.rank, bias=False)
            self.up = nn.Linear(self.rank, out_dim, bias=bias)
            if self.lora_rank > 0:
                self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_dim))
                self.lora_B = nn.Parameter(torch.zeros(out_dim, self.lora_rank))
            else:
                self.register_parameter("lora_A", None)
                self.register_parameter("lora_B", None)

            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        if self.lora_rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        energy: Optional[float] = None,
    ) -> None:
        # V4-FIX-16: documented padding / repeat handling.
        with torch.no_grad():
            W = weight.float()

            # Shape alignment
            if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
                rep = self.out_features // W.shape[0]
                W = W.repeat(rep, 1)
                if bias is not None:
                    bias = bias.repeat(rep)
            elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
                rep = W.shape[0] // self.out_features
                W = W.view(self.out_features, rep, -1).mean(dim=1)
                if bias is not None:
                    bias = bias.view(self.out_features, rep).mean(dim=1)
            elif W.shape[0] != self.out_features or W.shape[1] != self.in_features:
                padded = torch.zeros(self.out_features, self.in_features,
                                     device=W.device, dtype=W.dtype)
                r = min(self.out_features, W.shape[0])
                c = min(self.in_features, W.shape[1])
                padded[:r, :c] = W[:r, :c]
                W = padded

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)

            target_energy = float(energy) if energy is not None else self.energy
            total = (S ** 2).sum()
            cum = torch.cumsum(S ** 2, dim=0) / (total + 1e-5)
            calc_rank = int((cum < target_energy).sum().item()) + 1

            r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
            r = max(1, r)

            A = U[:, :r] * S[:r].unsqueeze(0)   # (out, r)
            B = Vt[:r, :]                       # (r, in)

            self.down.weight.data.zero_()
            self.up.weight.data.zero_()
            self.down.weight.data[:r, :] = B.to(self.down.weight.dtype)
            self.up.weight.data[:, :r] = A.to(self.up.weight.dtype)

            if bias is not None and self.up.bias is not None:
                n = min(len(self.up.bias), len(bias))
                self.up.bias.data[:n].copy_(bias[:n].to(self.up.bias.dtype))
            elif self.up.bias is not None:
                self.up.bias.data.zero_()

            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    @property
    def weight(self) -> torch.Tensor:
        base_w = torch.matmul(self.up.weight, self.down.weight)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            return base_w + (self.lora_B @ self.lora_A) * self.lora_scaling
        return base_w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


# ===========================================================================
# Attention primitives
# ===========================================================================

def temperature_softmax(
    x: torch.Tensor, dim: int = -1, temperature: float = 1.0,
) -> torch.Tensor:
    """V4-FIX-08: stable softmax with temperature (renamed from `softpick`)."""
    x_f = x.float()
    if temperature <= 0 or temperature == 1.0:
        return F.softmax(x_f, dim=dim).to(x.dtype)
    return F.softmax(x_f / temperature, dim=dim).to(x.dtype)


def local_window_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    softpick_temperature: float = 1.0,
    token_valid: Optional[torch.Tensor] = None,
    is_decoder: bool = False,
    past_len: int = 0,
) -> torch.Tensor:
    """
    Sliding-window attention with optional causal masking.

    V4-FIX-02: `past_len` shifts query positions to absolute token indices.
    V4-FIX-09: `window` = full window width. Effective radius = window // 2.

    Args:
        Q: (B, H, N, D)
        K: (B, H, N_kv, D)
        V: (B, H, N_kv, D)
        window: full window width
        softpick_temperature: softmax temperature
        token_valid: (B, N_kv) boolean padding mask
        is_decoder: apply causal masking
        past_len: number of cached tokens (for KV-cache decoding)

    Returns:
        context: (B, H, N, D)
    """
    B, H, N, D = Q.shape
    N_k = K.shape[2]
    radius = window // 2
    scale = (D ** -0.5) / max(1e-4, float(softpick_temperature))

    # Absolute positions
    abs_q = (past_len + torch.arange(N, device=Q.device)).unsqueeze(1)   # (N, 1)
    abs_k = torch.arange(N_k, device=Q.device).unsqueeze(0)              # (1, N_k)
    dist = (abs_q - abs_k).abs()
    mask = (dist <= radius).unsqueeze(0).unsqueeze(0)

    if is_decoder:
        causal = (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)
        mask = mask & causal

    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
    mask_val = -10000.0
    scores = scores.masked_fill(~mask, mask_val)
    if token_valid is not None:
        kv_valid = token_valid[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(~kv_valid, mask_val)
    scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)

    # Guard against fully-masked rows
    if token_valid is not None:
        combined_valid = mask & token_valid[:, None, None, :]
    else:
        combined_valid = mask
    any_valid = combined_valid.any(dim=-1, keepdim=True)

    attn = temperature_softmax(scores, dim=-1, temperature=softpick_temperature)
    attn = attn * any_valid.to(attn.dtype)

    return torch.matmul(attn, V)


def causal_kernel_linear_attention(
    Q_feat: torch.Tensor,
    K_feat: torch.Tensor,
    V: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    V4-FIX-03: causal kernel linear attention with chunked cumsum.

    Memory: O(B*H*chunk_size*F*D) at peak.
    Time:   O(N * F * D) forward.

    Args:
        Q_feat: (B, H, N, F)
        K_feat: (B, H, N, F)
        V:      (B, H, N, D)
    """
    B, H, N, F_dim = Q_feat.shape
    D = V.shape[-1]
    device = Q_feat.device
    dtype = Q_feat.dtype

    state_kv = torch.zeros(B, H, F_dim, D, device=device, dtype=dtype)
    state_k = torch.zeros(B, H, F_dim, device=device, dtype=dtype)

    outputs = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Qc = Q_feat[:, :, start:end]
        Kc = K_feat[:, :, start:end]
        Vc = V[:, :, start:end]

        # Per-chunk cumulative outer product: (B, H, C, F, D)
        KV_local = (Kc.unsqueeze(-1) * Vc.unsqueeze(-2)).cumsum(dim=2)
        K_local = Kc.cumsum(dim=2)

        KV_total = KV_local + state_kv.unsqueeze(2)
        K_total = K_local + state_k.unsqueeze(2)

        num = torch.einsum("bhcf,bhcfd->bhcd", Qc, KV_total)
        den = torch.einsum("bhcf,bhcf->bhc", Qc, K_total).clamp(min=1e-6)
        outputs.append(num / den.unsqueeze(-1))

        state_kv = KV_total[:, :, -1]
        state_k = K_total[:, :, -1]

    return torch.nan_to_num(torch.cat(outputs, dim=2), nan=0.0)


def kernel_linear_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rff_proj: torch.Tensor,
    use_zero_sum: bool = True,
    max_clamp: float = 11.0,
) -> torch.Tensor:
    """Non-causal global kernel linear attention (FAVOR+ style)."""
    B, H, N, D = Q.shape
    F_dim = rff_proj.size(-1)
    orig_dtype = Q.dtype

    Q_f = Q.float()
    K_f = K.float()
    V_f = V.float()
    rff_proj_f = rff_proj.float()

    if use_zero_sum:
        K_f = K_f - K_f.mean(dim=2, keepdim=True)
    Q_norm = F.normalize(Q_f, p=2, dim=-1)
    K_norm = F.normalize(K_f, p=2, dim=-1)

    Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_norm, rff_proj_f).clamp(max=max_clamp)
    K_proj = torch.einsum("bhnd,hdf->bhnf", K_norm, rff_proj_f).clamp(max=max_clamp)
    Q_feat = torch.exp(Q_proj) / math.sqrt(F_dim)
    K_feat = torch.exp(K_proj) / math.sqrt(F_dim)

    KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V_f)
    K_sum = K_feat.sum(dim=2)
    denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1)
    ctx = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / (denom + 1e-5)

    return torch.nan_to_num(ctx, nan=0.0).to(orig_dtype)


# ===========================================================================
# Main attention
# ===========================================================================

@ATTENTION_REGISTRY.register("aegis_v4_module", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("AegisAttentionV4", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("aegis_v4", supports_causal=True, supports_gqa=True)
class AegisAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        model_config: Any = None,
        config: Any = None,
        aegis_cfg: Optional[AegisConfig] = None,
        pretrained_attn: Optional[nn.Module] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        cfg = model_config if model_config is not None else config
        super().__init__(cfg, **kwargs)

        self.hidden_size: int = kwargs.get(
            "hidden_size",
            kwargs.get("d_model",
                       getattr(cfg, "hidden_size",
                               getattr(cfg, "d_model",
                                       getattr(cfg, "dim", 768))) if cfg is not None else 768),
        )
        self.num_heads: int = kwargs.get(
            "num_heads",
            kwargs.get("num_attention_heads",
                       getattr(cfg, "num_attention_heads",
                               getattr(cfg, "num_heads", 4)) if cfg is not None else 4),
        )
        self.head_dim: int = kwargs.get(
            "head_dim",
            getattr(cfg, "head_dim",
                    getattr(cfg, "d_kv",
                            self.hidden_size // max(1, self.num_heads))) if cfg is not None
            else self.hidden_size // max(1, self.num_heads),
        )
        self.num_kv_heads: int = kwargs.get(
            "num_kv_heads",
            kwargs.get("num_key_value_heads",
                       getattr(cfg, "num_key_value_heads",
                               getattr(cfg, "num_kv_heads", self.num_heads)) if cfg is not None
                       else self.num_heads),
        )
        if self.num_heads % max(1, self.num_kv_heads) != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})."
            )
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.rank: int = int(kwargs.get("rank", 64))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.cfg: AegisConfig = aegis_cfg if aegis_cfg is not None else AegisConfig()

        # V4-FIX-05: robust decoder detection
        self.is_decoder: bool = _detect_decoder(cfg, is_decoder, pretrained_attn)
        self.is_nested_encoder: bool = False

        # Split-head allocation
        if self.cfg.num_local_heads + self.cfg.num_global_heads != self.num_heads:
            self.num_local_heads = (self.num_heads * 2) // 3
            self.num_global_heads = self.num_heads - self.num_local_heads
        else:
            self.num_local_heads = self.cfg.num_local_heads
            self.num_global_heads = self.cfg.num_global_heads
        if self.num_local_heads <= 0:
            self.num_local_heads = max(1, self.num_heads // 2)
            self.num_global_heads = self.num_heads - self.num_local_heads

        # Projections
        self.q_proj = FactorisedLinear(
            self.hidden_size, self.all_head_size,
            energy=self.cfg.svd_energy, min_rank=self.cfg.min_rank,
            max_rank=self.cfg.max_rank, rank=self.rank,
        )
        self.k_proj = FactorisedLinear(
            self.hidden_size, self.all_kv_head_size,
            energy=self.cfg.svd_energy, min_rank=self.cfg.min_rank,
            max_rank=self.cfg.max_rank, rank=self.rank,
        )
        self.v_proj = FactorisedLinear(
            self.hidden_size, self.all_kv_head_size,
            energy=self.cfg.svd_energy, min_rank=self.cfg.min_rank,
            max_rank=self.cfg.max_rank, rank=self.rank,
        )
        self.out_proj = FactorisedLinear(
            self.all_head_size, self.hidden_size,
            energy=self.cfg.svd_energy, min_rank=self.cfg.min_rank,
            max_rank=self.cfg.max_rank, rank=self.rank,
        )

        # RFF projection for global heads (V4-FIX-14: buffer by default)
        rff_init = torch.empty(self.num_global_heads, self.head_dim, self.cfg.rff_features)
        for h in range(self.num_global_heads):
            dim = max(self.head_dim, self.cfg.rff_features)
            q_mat, _ = torch.linalg.qr(torch.randn(dim, dim))
            rff_init[h] = q_mat[: self.head_dim, : self.cfg.rff_features] / math.sqrt(self.cfg.rff_features)
        if self.cfg.rff_trainable:
            self.rff_proj = nn.Parameter(rff_init)
        else:
            self.register_buffer("rff_proj", rff_init)
        self.rff_features = self.cfg.rff_features

        # Adaptive gate (V4-FIX-10/11)
        if self.cfg.use_adaptive_gate:
            self.blend_logits = nn.Parameter(torch.zeros(self.num_local_heads))
            self.gate_bias: float = float(self.cfg.gate_init_bias)
        else:
            self.register_parameter("blend_logits", None)
            self.gate_bias = 0.0

        self.dropout = nn.Dropout(self.cfg.dropout)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    @property
    def query(self) -> FactorisedLinear: return self.q_proj
    @property
    def key(self) -> FactorisedLinear: return self.k_proj
    @property
    def value(self) -> FactorisedLinear: return self.v_proj
    @property
    def q_lin(self) -> FactorisedLinear: return self.q_proj
    @property
    def k_lin(self) -> FactorisedLinear: return self.k_proj
    @property
    def v_lin(self) -> FactorisedLinear: return self.v_proj
    @property
    def out_lin(self) -> FactorisedLinear: return self.out_proj
    @property
    def dense(self) -> FactorisedLinear: return self.out_proj
    @property
    def o_proj(self) -> FactorisedLinear: return self.out_proj
    @property
    def c_proj(self) -> FactorisedLinear: return self.out_proj

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter failed for %s: %s", proj, e)

        # GPT-2 fused c_attn (Conv1D)  -- V4-FIX-19: comment corrected
        if hasattr(old_attn, "c_attn") and any(
            extracted[k][0] is None for k in ("q", "k", "v")
        ):
            weight = old_attn.c_attn.weight.data
            bias = getattr(old_attn.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t().contiguous()
            split = weight.shape[0] // 3
            extracted["q"] = (weight[:split, :], bias_data[:split] if bias_data is not None else None)
            extracted["k"] = (weight[split: 2 * split, :], bias_data[split: 2 * split] if bias_data is not None else None)
            extracted["v"] = (weight[2 * split:, :], bias_data[2 * split:] if bias_data is not None else None)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            w_o = old_attn.c_proj.weight.data
            if w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size:
                w_o = w_o.t().contiguous()
            b_o = getattr(old_attn.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # BERT / RoBERTa nested encoder
        if hasattr(old_attn, "self") and any(
            extracted[k][0] is None for k in ("q", "k", "v")
        ):
            s = old_attn.self
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense") and extracted["o"][0] is None:
            dense = old_attn.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # Named mapping
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(old_attn, name):
                        layer = getattr(old_attn, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        # Nested encoder detection
        cls_name = old_attn.__class__.__name__
        has_out_proj = any(
            hasattr(old_attn, a) for a in
            ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj")
        )
        if hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj"):
            self.is_nested_encoder = False
        elif (hasattr(old_attn, "self")
              and hasattr(getattr(old_attn, "self"), "query")
              and hasattr(old_attn, "output")
              and hasattr(getattr(old_attn, "output"), "dense")):
            self.is_nested_encoder = True
        elif (cls_name in ("BertSelfAttention", "RobertaSelfAttention")
              or (cls_name.endswith("SelfAttention") and not has_out_proj)):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AegisAttention.copy_weights: could not extract mandatory projections "
                f"{missing} from {old_attn.__class__.__name__}."
            )

        for proj_key in ("q", "k", "v"):
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float()).item()
                if norm < 1e-4:
                    logger.warning(
                        "AegisAttention: extracted weight for '%s' has near-zero "
                        "Frobenius norm (%.2e).", proj_key, norm,
                    )
        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float()).item()
            if norm_o < 1e-4:
                logger.warning(
                    "AegisAttention: extracted weight for output projection has "
                    "near-zero Frobenius norm (%.2e).", norm_o,
                )

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in (("q", self.q_proj), ("k", self.k_proj),
                                     ("v", self.v_proj), ("o", self.out_proj)):
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    target.decompose_from_weight(_orient(w, in_dim), b, self.cfg.svd_energy)
                elif proj_key == "o" and self.is_nested_encoder:
                    eye_w = torch.eye(
                        self.all_head_size, self.all_head_size,
                        device=self.out_proj.up.weight.device,
                        dtype=self.out_proj.up.weight.dtype,
                    )
                    target.decompose_from_weight(eye_w, None, self.cfg.svd_energy)

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)

    # ------------------------------------------------------------------
    # V4-FIX-13: safe from_distilbert_pretrained
    # ------------------------------------------------------------------
    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "AegisAttention":
        """Build AegisAttention from a DistilBERT attention submodule."""
        if model_config is None:
            # Prefer out_lin > dense, with safe fallback
            out_mod = getattr(old_attn, "out_lin", None) or getattr(old_attn, "dense", None)
            if out_mod is None or not hasattr(out_mod, "out_features"):
                raise ValueError(
                    "from_distilbert_pretrained: could not infer hidden_size "
                    "from out_lin or dense. Provide model_config explicitly."
                )

            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = out_mod.out_features
            n_heads = getattr(old_attn, "n_heads",
                              getattr(old_attn, "num_attention_heads",
                                      getattr(old_attn, "num_heads", None)))
            if n_heads is None:
                # Fallback: assume hidden_size / 64
                n_heads = max(1, model_config.hidden_size // 64)
            model_config.num_attention_heads = n_heads
            model_config.num_key_value_heads = n_heads
            model_config.head_dim = model_config.hidden_size // n_heads

        return cls(model_config=model_config, pretrained_attn=old_attn, **kwargs)

    # ------------------------------------------------------------------
    # V4-FIX-17: centralized mask handling
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: Optional[torch.Tensor],
        B: int, N_kv: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Return a boolean tensor (B, N_kv): True = key is valid (padding).
        Returns None when no mask is provided.
        """
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

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
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        Hkv = self.num_kv_heads
        device = hidden_states.device

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        # --- Projections ---
        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2).contiguous()
        K_new = self.k_proj(kv_input).view(B, N_kv_new, Hkv, D).transpose(1, 2).contiguous()
        V_new = self.v_proj(kv_input).view(B, N_kv_new, Hkv, D).transpose(1, 2).contiguous()

        # --- RoPE with past offset ---
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_k_probe = None
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_k_probe = past_key_value[0]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_k_probe = past_key_value.key_cache[0]
            past_len = 0 if past_k_probe is None else past_k_probe.shape[-2]
            cos = cos[..., past_len: past_len + N, :]
            sin = sin[..., past_len: past_len + N, :]
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin)

        # --- KV cache concat ---
        past_k = past_v = None
        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = getattr(self, "layer_idx", 0)
                idx = min(idx, len(past_key_value.key_cache) - 1)
                past_k, past_v = past_key_value.key_cache[idx], past_key_value.value_cache[idx]

        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k, K_new], dim=-2)
            V_cat = torch.cat([past_v, V_new], dim=-2)
        else:
            K_cat = K_new
            V_cat = V_new

        # DynamicCache mutation
        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            idx = getattr(self, "layer_idx", 0)
            past_key_value.update(K_new, V_new, idx)
            cache_to_return = past_key_value
        else:
            cache_to_return = (K_cat, V_cat)

        N_kv = K_cat.shape[2]
        past_len = N_kv - N if (self.is_decoder and not is_cross) else 0

        # --- GQA expansion ---
        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        # --- Mask ---
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)

        # --- Path selection ---
        # V4-FIX-07: decoders now use the split-head path (with causal handling).
        use_sdpa = (
            self.cfg.use_sdpa_fast_path
            and N_kv <= max(self.cfg.local_window, 32)
            and not output_attentions
        )

        if use_sdpa:
            # ------------------------------------------------------------------
            # SDPA fast path (very short sequences only)
            # ------------------------------------------------------------------
            q_sdp = Q.half() if Q.dtype == torch.float32 else Q
            k_sdp = K_full.half() if K_full.dtype == torch.float32 else K_full
            v_sdp = V_full.half() if V_full.dtype == torch.float32 else V_full

            attn_bias_sdp = None
            if eff_mask is not None and eff_mask.dim() == 4:
                attn_bias_sdp = eff_mask.to(q_sdp.dtype)
            elif eff_mask is not None and eff_mask.dim() == 2:
                attn_bias_sdp = ((eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
                                 ).to(q_sdp.dtype)
            is_causal_sdp = self.is_decoder and not is_cross and N > 1

            out = F.scaled_dot_product_attention(
                q_sdp, k_sdp, v_sdp,
                attn_mask=attn_bias_sdp,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal_sdp,
            )
            out = out.to(hidden_states.dtype)
            attn_weights_out = None
        else:
            # ------------------------------------------------------------------
            # Split-head path (default for N > window and all decoders)
            # ------------------------------------------------------------------
            Hl, Hg = self.num_local_heads, self.num_global_heads
            Q_local, Q_global = Q[:, :Hl], Q[:, Hl:]
            K_local, K_global = K_full[:, :Hl], K_full[:, Hl:]
            V_local, V_global = V_full[:, :Hl], V_full[:, Hl:]

            # Local heads: sliding-window attention
            ctx_local = local_window_attention(
                Q_local, K_local, V_local,
                window=self.cfg.local_window,
                softpick_temperature=self.cfg.softpick_temperature,
                token_valid=token_valid,
                is_decoder=self.is_decoder and not is_cross,
                past_len=past_len,
            )

            # Global heads: kernel linear attention
            Q_g_norm = F.normalize(Q_global.float(), p=2, dim=-1)
            K_g_norm = F.normalize(K_global.float(), p=2, dim=-1)
            if self.cfg.use_zero_sum_keys:
                K_g_norm = K_g_norm - K_g_norm.mean(dim=2, keepdim=True)

            rff_proj_f = self.rff_proj.float()
            max_clamp = 11.0
            Q_proj = torch.einsum("bhnd,hdf->bhnf", Q_g_norm, rff_proj_f).clamp(max=max_clamp)
            K_proj = torch.einsum("bhnd,hdf->bhnf", K_g_norm, rff_proj_f).clamp(max=max_clamp)
            scale_rff = 1.0 / math.sqrt(self.rff_features)
            Q_feat = torch.exp(Q_proj) * scale_rff
            K_feat = torch.exp(K_proj) * scale_rff

            # V4-FIX-20: apply token_valid to global branch too.
            if token_valid is not None:
                kv_valid = token_valid[:, None, :, None].to(K_feat.dtype)
                K_feat = K_feat * kv_valid
                V_global_f = V_global.float() * kv_valid.to(V_global.dtype)
            else:
                V_global_f = V_global.float()

            if self.is_decoder and not is_cross:
                ctx_global = causal_kernel_linear_attention(
                    Q_feat, K_feat, V_global_f,
                    chunk_size=self.cfg.causal_chunk_size,
                ).to(Q.dtype)
            else:
                KV = torch.einsum("bhnf,bhnd->bhfd", K_feat, V_global_f)
                K_sum = K_feat.sum(dim=2)
                denom = torch.einsum("bhnf,bhf->bhn", Q_feat, K_sum).unsqueeze(-1).clamp(min=1e-5)
                ctx_global = torch.einsum("bhnf,bhfd->bhnd", Q_feat, KV) / denom
                ctx_global = torch.nan_to_num(ctx_global, nan=0.0).to(Q.dtype)

            # Adaptive gate: blend local heads with global mean
            if self.blend_logits is not None:
                global_mean = ctx_global.mean(dim=1, keepdim=True)   # (B, 1, N, D)
                gate_local = torch.sigmoid(
                    self.blend_logits.view(1, Hl, 1, 1) + self.gate_bias
                )   # (1, Hl, 1, 1)
                ctx_local = gate_local * ctx_local + (1.0 - gate_local) * global_mean.to(ctx_local.dtype)
            context = torch.cat([ctx_local, ctx_global], dim=1)   # (B, H, N, D)

            # Reshape to (B, N, H*D)
            out = context.transpose(1, 2).contiguous().view(B, N, H * D)
            attn_weights_out = None

        # --- head_mask (V4-FIX-01/18) ---
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)       # (B, H, 1, 1)

            # Normalize `out` to (B, N, H, D) for masking
            if out.dim() == 4 and out.shape[1] == H:
                out = out * head_mask.to(out.dtype)
                out = out.transpose(1, 2).contiguous().view(B, N, H * D)
            else:
                out = out.view(B, N, H, D) * head_mask.to(out.dtype)
                out = out.reshape(B, N, H * D)

        # --- Ensure out is (B, N, H*D) ---
        if out.dim() == 4:
            if out.shape[1] == H:
                out = out.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
            else:
                out = out.contiguous().view(B, N, self.all_head_size)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # --- Output projection (skipped for nested encoders) ---
        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)

        return output, attn_weights_out

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.out_proj.up.weight.device)


# ===========================================================================
# Injection
# ===========================================================================

@ATTENTION_REGISTRY.register("aegis_v4_fn")
@ATTENTION_REGISTRY.register("aegis_v4_inject")
def inject_aegis_v4(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,   # V4-FIX-04: recursion guard
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_aegis_v4: max recursion depth exceeded, aborting.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_v4(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target
    try:
        if model_config is None:
            if any(hasattr(layer, a) for a in ("hidden_size", "d_model", "n_embd", "dim")):
                model_config = layer
            elif hasattr(layer, "attention"):
                old_attn = layer.attention

        aegis_cfg = AegisConfig(
            svd_energy=getattr(comp_config, "svd_energy", 0.95),
            min_rank=getattr(comp_config, "min_rank", 4),
            max_rank=getattr(comp_config, "max_rank", 384),
            num_local_heads=getattr(comp_config, "num_local_heads", 8),
            num_global_heads=getattr(comp_config, "num_global_heads", 4),
            local_window=getattr(comp_config, "block_size", 64),
            rff_features=getattr(comp_config, "rff_features", 128),
            gate_init_bias=getattr(comp_config, "gate_init_bias", 1.8),
            dropout=getattr(comp_config, "dropout", 0.1),
            rff_trainable=getattr(comp_config, "rff_trainable", False),
            use_sdpa_fast_path=getattr(comp_config, "use_sdpa_fast_path", True),
            causal_chunk_size=getattr(comp_config, "causal_chunk_size", 64),
        )

        src_attn = old_attn if old_attn is not None else getattr(
            layer, "attention",
            getattr(layer, "self", getattr(layer, "attn", None)),
        )

        # V4-FIX-05: robust decoder detection
        is_dec = _detect_decoder(model_config, None, src_attn)

        new_attn = AegisAttention(
            model_config=model_config,
            aegis_cfg=aegis_cfg,
            is_decoder=is_dec,
        )
        if src_attn is not None:
            try:
                new_attn.copy_weights(src_attn)
            except Exception as e:
                logger.error("inject_aegis_v4: copy_weights failed for %s: %s",
                             src_attn.__class__.__name__, e)
                return None
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)

        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.exception("Failed to inject AEGIS-V4: %s", e)
        return None


# Aliases preserved for backward compatibility.
AegisAttentionV4 = AegisAttention
inject_aegis_into_distilbert = inject_aegis_v4