r"""
Aegis-Attention V2 (Corrected).

Hybrid low-rank factorized attention with:
  * SVD + LoRA projections (FactorisedLinear)
  * Local sliding-window attention over all heads
  * Global ELU+1 linear kernel attention with adaptive soft-pruning gate
  * Data-dependent multi-head fusion gate

Bug-fix map:
  V2-FIX-01  decoder detection via helper (no locals()).
  V2-FIX-02  head_mask broadcasting corrected (B,H,1,1).
  V2-FIX-03  local window: past_len offset for KV-cache decoding.
  V2-FIX-04  injector recursion guard (_depth).
  V2-FIX-05  unified mask handling (_prepare_valid_mask).
  V2-FIX-06  prune gate broadcasts over GQA correctly.
  V2-FIX-07  from_distilbert_pretrained safe fallback.
  V2-FIX-08  decoder path uses causal masked local + causal global.
  V2-FIX-09  removed final clamp(min=-1e4, max=1e4).
  V2-FIX-10  alias rank == svd_rank for framework FLOPs detection.
"""

import logging
import math
from dataclasses import dataclass
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
    """Apply RoPE. cos/sin must already be sliced to the current window."""
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


# ===========================================================================
# V2-FIX-01: decoder detection
# ===========================================================================

def _detect_decoder(
    config: Any,
    is_decoder: Optional[bool],
    old_attn: Optional[nn.Module],
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                          "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
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
class AegisAttentionConfigV2:
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_heads: int = 12
    head_dim: int = 64
    num_key_value_heads: int = 12
    local_window: int = 64
    rank: int = 384
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_adaptive_pruning: bool = True
    gate_init_bias: float = 1.8
    is_decoder: bool = False


AegisConfigV2 = AegisAttentionConfigV2


# ===========================================================================
# FactorisedLinear (V2-FIX-10)
# ===========================================================================

class FactorisedLinear(nn.Module):
    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 384,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        energy: float = 0.95,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        # V2-FIX-10: alias so framework FLOPs detector sees the rank.
        self.svd_rank = self.rank
        self.energy = energy
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=bias)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        energy: Optional[float] = None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
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
            S = torch.clamp(S, min=1e-6)
            target_energy = energy if energy is not None else self.energy
            total = (S ** 2).sum()
            cum = torch.cumsum(S ** 2, dim=0) / (total + 1e-5)
            calc_rank = int((cum < target_energy).sum().item()) + 1
            r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
            r = max(1, r)

            A = U[:, :r] * S[:r].unsqueeze(0)
            B = Vt[:r, :]

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

def _local_window_attention_v2(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    window: int,
    token_valid: Optional[torch.Tensor] = None,
    is_decoder: bool = False,
    past_len: int = 0,
) -> torch.Tensor:
    """
    V2 local attention primitive.

    V2-FIX-03: past_len shifts query positions to absolute indices.
    window here is the FULL window width (not radius); radius = window // 2.

    Args:
        Q: (B, H, N, D)
        K: (B, H, N_kv, D)
        V: (B, H, N_kv, D)
        token_valid: (B, N_kv) boolean padding mask
        past_len: cached tokens (for KV-cache decode)
    """
    B, H, N, D = Q.shape
    N_k = K.shape[2]
    radius = window // 2
    scale = D ** -0.5

    abs_q = (past_len + torch.arange(N, device=Q.device)).unsqueeze(1)
    abs_k = torch.arange(N_k, device=Q.device).unsqueeze(0)
    dist = (abs_q - abs_k).abs()
    band = (dist <= radius).unsqueeze(0).unsqueeze(0)

    if is_decoder:
        causal = (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)
        band = band & causal

    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
    mask_val = -1e4
    scores = scores.masked_fill(~band, mask_val)
    if token_valid is not None:
        kv_valid = token_valid[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(~kv_valid, mask_val)
    scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e4, neginf=mask_val)

    combined_valid = band & token_valid[:, None, None, :] if token_valid is not None else band
    any_valid = combined_valid.any(dim=-1, keepdim=True).to(scores.dtype)

    attn = F.softmax(scores, dim=-1).to(Q.dtype) * any_valid
    return torch.matmul(attn, V)


def _phi_elu(x: torch.Tensor) -> torch.Tensor:
    """V2 kernel feature map: ELU + 1 (Katharopoulos et al.)."""
    return F.elu(x) + 1.0


def _global_linear_attention_v2(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    token_valid: Optional[torch.Tensor],
    is_decoder: bool = False,
) -> torch.Tensor:
    """
    Non-causal or causal ELU+1 kernel linear attention.

    V2-FIX-08: adds a causal path for decoders so the global branch is
    meaningful in GPT-2 / Qwen and does not leak future tokens.
    """
    B, H, N, D = Q.shape
    N_k = K.shape[2]

    phi_Q = _phi_elu(Q.float())
    phi_K = _phi_elu(K.float())
    V_f = V.float()

    if token_valid is not None:
        mask_kv = token_valid[:, None, :, None].to(phi_K.dtype)  # (B,1,N_k,1)
        phi_K = phi_K * mask_kv
        V_f = V_f * mask_kv

    # ---- Causal case: chunked cumsum ----
    if is_decoder:
        if N != N_k:
            # Single-token decode: exact non-causal over the full history.
            KV = torch.matmul(phi_K.transpose(-2, -1).contiguous(), V_f)  # (B,H,F,D)
            Z = phi_K.sum(dim=-2, keepdim=True)                           # (B,H,1,F)
            num = torch.matmul(phi_Q, KV)
            den = torch.matmul(phi_Q, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6)
            return torch.nan_to_num(num / den, nan=0.0).to(Q.dtype)

        chunk = 64
        state_kv = torch.zeros(B, H, phi_Q.shape[-1], D, device=Q.device, dtype=phi_Q.dtype)
        state_k = torch.zeros(B, H, phi_Q.shape[-1], device=Q.device, dtype=phi_Q.dtype)
        outs = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            Qc = phi_Q[:, :, start:end]
            Kc = phi_K[:, :, start:end]
            Vc = V_f[:, :, start:end]

            KV_local = (Kc.unsqueeze(-1) * Vc.unsqueeze(-2)).cumsum(dim=2)
            K_local = Kc.cumsum(dim=2)

            KV_total = KV_local + state_kv.unsqueeze(2)
            K_total = K_local + state_k.unsqueeze(2)

            num = torch.einsum("bhcf,bhcfd->bhcd", Qc, KV_total)
            den = torch.einsum("bhcf,bhcf->bhc", Qc, K_total).clamp(min=1e-6)
            outs.append(num / den.unsqueeze(-1))

            state_kv = KV_total[:, :, -1]
            state_k = K_total[:, :, -1]

        return torch.nan_to_num(torch.cat(outs, dim=2), nan=0.0).to(Q.dtype)

    # ---- Non-causal case ----
    KV = torch.matmul(phi_K.transpose(-2, -1).contiguous(), V_f)   # (B,H,F,D)
    Z = phi_K.sum(dim=-2, keepdim=True)                             # (B,H,1,F)
    num = torch.matmul(phi_Q, KV)                                   # (B,H,N,D)
    den = torch.matmul(phi_Q, Z.transpose(-2, -1).contiguous()).clamp(min=1e-6)
    return torch.nan_to_num(num / den, nan=0.0).to(Q.dtype)


# ===========================================================================
# Main attention
# ===========================================================================

@ATTENTION_REGISTRY.register("aegis_v2", supports_causal=True, supports_gqa=True)
class AegisAttentionV2(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any = None,
        pretrained_attn: Optional[nn.Module] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if config is None or not isinstance(config, AegisConfigV2):
            h_size = kwargs.get("hidden_size",
                     getattr(config, "hidden_size",
                     getattr(config, "d_model",
                     getattr(config, "dim", 768)))) if config is not None else 768
            n_heads = kwargs.get("num_heads",
                      getattr(config, "num_attention_heads",
                      getattr(config, "num_heads", 12))) if config is not None else 12
            h_dim = kwargs.get("head_dim",
                    getattr(config, "head_dim",
                    getattr(config, "d_kv", h_size // max(1, n_heads)))) if config is not None else h_size // max(1, n_heads)
            n_kv_heads = kwargs.get("num_key_value_heads",
                         getattr(config, "num_key_value_heads",
                         getattr(config, "num_kv_heads", n_heads))) if config is not None else n_heads

            # V2-FIX-01: robust decoder detection
            dec = _detect_decoder(config, is_decoder, pretrained_attn)

            config = AegisConfigV2(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                num_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                head_dim=h_dim,
                local_window=kwargs.get("local_window",
                             getattr(config, "block_size",
                             getattr(config, "local_window", 64))) if config is not None else 64,
                rank=kwargs.get("rank",
                     getattr(config, "lora_r",
                     getattr(config, "rank", 64))) if config is not None else 64,
                kernel_feature_dim=getattr(config, "rff_features",
                                    getattr(config, "kernel_feature_dim", 128)) if config is not None else 128,
                dropout=kwargs.get("dropout", getattr(config, "dropout", 0.1)) if config is not None else 0.1,
                use_adaptive_pruning=getattr(config, "use_gate", True) if config is not None else True,
                gate_init_bias=getattr(config, "gate_init_bias", 1.8) if config is not None else 1.8,
                is_decoder=dec,
            )

        super().__init__(config)
        self.config = config
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim
        self.num_kv_heads: int = getattr(config, "num_key_value_heads", self.num_heads)
        if self.num_heads % max(1, self.num_kv_heads) != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})."
            )
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.local_window: int = config.local_window
        self.rank: int = config.rank
        self.kernel_dim: int = config.kernel_feature_dim
        self.dropout = nn.Dropout(config.dropout)
        self.is_decoder: bool = getattr(config, "is_decoder", False)
        self.is_nested_encoder: bool = False

        self.q_proj = FactorisedLinear(self.hidden_size, self.all_head_size, rank=self.rank)
        self.k_proj = FactorisedLinear(self.hidden_size, self.all_kv_head_size, rank=self.rank)
        self.v_proj = FactorisedLinear(self.hidden_size, self.all_kv_head_size, rank=self.rank)
        self.out_proj = FactorisedLinear(self.all_head_size, self.hidden_size, rank=self.rank)

        if config.use_adaptive_pruning:
            self.prune_gate: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(self.hidden_size, max(1, self.hidden_size // 4)),
                nn.ReLU(),
                nn.Linear(max(1, self.hidden_size // 4), self.num_heads),
            )
        else:
            self.prune_gate = None

        self.fusion_gate = nn.Linear(self.hidden_size, self.num_heads)
        self.fusion_gate.bias.data.fill_(config.gate_init_bias)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    @property
    def q_proj1(self): return self.q_proj.down
    @property
    def q_proj2(self): return self.q_proj.up
    @property
    def k_proj1(self): return self.k_proj.down
    @property
    def k_proj2(self): return self.k_proj.up
    @property
    def v_proj1(self): return self.v_proj.down
    @property
    def v_proj2(self): return self.v_proj.up
    @property
    def out_proj1(self): return self.out_proj.down
    @property
    def out_proj2(self): return self.out_proj.up
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def dense(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # Weight transfer (same logic as V5)
    # ------------------------------------------------------------------
    def _copy_pretrained(self, old_attn: nn.Module) -> None:
        self.copy_weights(old_attn)

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
                logger.debug("UniversalAdapter fallback failed: %s", e)

        # GPT-2 fused c_attn (Conv1D)
        if hasattr(old_attn, "c_attn") and any(
            extracted[k][0] is None for k in ("q", "k", "v")
        ):
            c_attn_mod = old_attn.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                is_conv1d = (c_attn_mod.__class__.__name__ == "Conv1D") or \
                            (base_c.__class__.__name__ == "Conv1D") or \
                            hasattr(c_attn_mod, "nf") or hasattr(base_c, "nf")
                if is_conv1d or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size):
                    weight = weight.t().contiguous()
                bias = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bd = bias.data if bias is not None else None
                split = weight.shape[0] // 3
                extracted["q"] = (weight[:split, :], bd[:split] if bd is not None else None)
                extracted["k"] = (weight[split: 2 * split, :], bd[split: 2 * split] if bd is not None else None)
                extracted["v"] = (weight[2 * split:, :], bd[2 * split:] if bd is not None else None)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            c_proj_mod = old_attn.c_proj
            base_p = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_p, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (c_proj_mod.__class__.__name__ == "Conv1D") or \
                            (base_p.__class__.__name__ == "Conv1D") or \
                            hasattr(c_proj_mod, "nf") or hasattr(base_p, "nf")
                if is_conv1d or (w_o.dim() == 2 and w_o.shape[0] != self.hidden_size
                                 and w_o.shape[1] == self.hidden_size):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_p, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # BERT / RoBERTa nested encoder
        if hasattr(old_attn, "self") and any(
            extracted[k][0] is None for k in ("q", "k", "v")
        ):
            s = getattr(old_attn, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense") \
                and extracted["o"][0] is None:
            dense = old_attn.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

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

        cls_name = old_attn.__class__.__name__
        has_out_proj = any(hasattr(old_attn, a) for a in
                           ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        if hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj"):
            self.is_nested_encoder = False
        elif (hasattr(old_attn, "self")
              and hasattr(getattr(old_attn, "self"), "query")
              and hasattr(old_attn, "output")
              and hasattr(getattr(old_attn, "output"), "dense")):
            self.is_nested_encoder = True
        elif (cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                           "XLMRobertaSelfAttention", "ElectraSelfAttention")
              or "RobertaSelfAttention" in str(type(old_attn))
              or "BertSelfAttention" in str(type(old_attn))
              or (cls_name.endswith("SelfAttention") and not has_out_proj)):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        missing = [k for k in ("q", "k", "v") if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AegisAttentionV2.copy_weights: missing {missing} from "
                f"{old_attn.__class__.__name__}. Refusing random init."
            )

        for proj_key in ("q", "k", "v"):
            w = extracted[proj_key][0]
            if w is not None:
                norm = float(torch.linalg.norm(w.float()).item())
                if norm < 1e-4:
                    logger.warning(
                        "AegisAttentionV2: weight for '%s' has near-zero norm (%.2e).",
                        proj_key, norm,
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
                    target.decompose_from_weight(_orient(w, in_dim), b)
                elif proj_key == "o" and self.is_nested_encoder:
                    eye_w = torch.eye(
                        self.all_head_size, self.all_head_size,
                        device=self.out_proj.up.weight.device,
                        dtype=self.out_proj.up.weight.dtype,
                    )
                    target.decompose_from_weight(eye_w, None)

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
            self.is_decoder = True

    # ------------------------------------------------------------------
    # V2-FIX-07: safe from_distilbert_pretrained
    # ------------------------------------------------------------------
    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "AegisAttentionV2":
        if model_config is None:
            out_mod = getattr(old_attn, "out_lin", None) or getattr(old_attn, "dense", None)
            if out_mod is None or not hasattr(out_mod, "out_features"):
                raise ValueError(
                    "from_distilbert_pretrained: could not infer hidden_size "
                    "from out_lin or dense. Pass model_config explicitly."
                )
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = out_mod.out_features
            n_heads = getattr(old_attn, "n_heads",
                              getattr(old_attn, "num_attention_heads",
                                      getattr(old_attn, "num_heads", None)))
            if n_heads is None:
                n_heads = max(1, model_config.hidden_size // 64)
            model_config.num_attention_heads = n_heads
            model_config.num_key_value_heads = n_heads
            model_config.head_dim = model_config.hidden_size // n_heads
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    # ------------------------------------------------------------------
    # V2-FIX-05: unified mask handling
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: Optional[torch.Tensor],
        B: int, N_kv: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return boolean (B, N_kv): True = key is valid. None if no mask."""
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
            # (B, 1, N_q, N_kv) -> OR over query dim -> (B, N_kv)
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    # V2-FIX-02 + V2-FIX-09: forward
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

        # ---- Mask (V2-FIX-05) ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv_new, device)

        # ---- Projections ----
        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2).contiguous()
        K_new = self.k_proj(kv_input).view(B, N_kv_new, Hkv, D).transpose(1, 2).contiguous()
        V_new = self.v_proj(kv_input).view(B, N_kv_new, Hkv, D).transpose(1, 2).contiguous()

        # ---- RoPE with past offset ----
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len_rope = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len_rope = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len_rope = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len_rope: past_len_rope + N, :]
            sin = sin[..., past_len_rope: past_len_rope + N, :]
            Q, K_new = apply_rotary_pos_emb(Q, K_new, cos, sin)

        # ---- KV-cache concat ----
        past_k = past_v = None
        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(getattr(self, "layer_idx", 0), len(past_key_value.key_cache) - 1)
                past_k, past_v = past_key_value.key_cache[idx], past_key_value.value_cache[idx]

        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k, K_new], dim=-2)
            V_cat = torch.cat([past_v, V_new], dim=-2)
        else:
            K_cat = K_new
            V_cat = V_new

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            idx = getattr(self, "layer_idx", 0)
            past_key_value.update(K_new, V_new, idx)
            cache_to_return = past_key_value
        else:
            cache_to_return = (K_cat, V_cat)

        N_kv = K_cat.shape[2]
        past_len = N_kv - N if (self.is_decoder and not is_cross) else 0

        # ---- GQA expansion ----
        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        # ---- Local attention (V2-FIX-03) ----
        local_out = _local_window_attention_v2(
            Q, K_full, V_full,
            window=self.local_window,
            token_valid=token_valid,
            is_decoder=self.is_decoder and not is_cross,
            past_len=past_len,
        )   # (B, H, N, D)

        # ---- Global attention (V2-FIX-08) ----
        if not is_cross:
            if self.prune_gate is not None:
                keep = torch.sigmoid(self.prune_gate(hidden_states))  # (B, N, H)
                # V2-FIX-06: expand across GQA heads
                if self.kv_repeat > 1:
                    keep_kv = keep[:, :, :self.num_kv_heads]
                else:
                    keep_kv = keep
                Kg = K_cat * keep_kv.transpose(1, 2).unsqueeze(-1)
                Vg = V_cat * keep_kv.transpose(1, 2).unsqueeze(-1)
            else:
                Kg, Vg = K_cat, V_cat

            if self.kv_repeat > 1:
                Kg = repeat_kv(Kg, self.kv_repeat)
                Vg = repeat_kv(Vg, self.kv_repeat)

            global_out = _global_linear_attention_v2(
                Q, Kg, Vg,
                token_valid=token_valid,
                is_decoder=self.is_decoder,
            )   # (B, H, N, D)

            alpha = torch.sigmoid(self.fusion_gate(hidden_states)).unsqueeze(-1)  # (B, N, H, 1)
            alpha = alpha.transpose(1, 2)  # (B, H, N, 1)
            fused = alpha * local_out + (1.0 - alpha) * global_out
        else:
            # Cross-attention: local only (global over encoder is questionable)
            fused = local_out

        # ---- head_mask (V2-FIX-02) ----
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)   # (B, H, 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)       # (B, H, 1, 1)
            fused = fused * head_mask.to(fused.dtype)

        # ---- Reshape to (B, N, H*D) ----
        out = fused.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)
            output = self.dropout(output)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        # V2-FIX-09: no artificial clamp.

        return output, None

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.out_proj.up.weight.device)


# ===========================================================================
# Injection (V2-FIX-04)
# ===========================================================================

@ATTENTION_REGISTRY.register("aegis_v2_inject")
@ATTENTION_REGISTRY.register("inject_aegis_v2")
def inject_aegis_v2(
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
        logger.warning("inject_aegis_v2: max recursion depth, aborting.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
            res = inject_aegis_v2(
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

        h_size = getattr(model_config, "hidden_size",
                 getattr(model_config, "d_model",
                 getattr(model_config, "n_embd",
                 getattr(model_config, "dim", 768)))) if model_config is not None else 768
        n_heads = getattr(model_config, "num_attention_heads",
                  getattr(model_config, "num_heads",
                  getattr(model_config, "n_heads",
                  getattr(model_config, "n_head", 12)))) if model_config is not None else 12
        h_dim = getattr(model_config, "head_dim",
                getattr(model_config, "d_kv",
                h_size // max(1, n_heads))) if model_config is not None else h_size // max(1, n_heads)
        n_kv_heads = getattr(model_config, "num_key_value_heads",
                     getattr(model_config, "num_kv_heads", n_heads)) if model_config is not None else n_heads

        src_attn = old_attn if old_attn is not None else \
                   getattr(layer, "attention",
                   getattr(layer, "self",
                   getattr(layer, "attn", None)))

        # V2-FIX-01: use shared helper
        is_dec = _detect_decoder(model_config, None, src_attn)

        cfg = AegisConfigV2(
            hidden_size=h_size,
            num_attention_heads=n_heads,
            num_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=h_dim,
            local_window=getattr(comp_config, "block_size", 64),
            rank=getattr(comp_config, "rank", getattr(comp_config, "lora_r", 128)),
            kernel_feature_dim=getattr(comp_config, "rff_features", 128),
            dropout=getattr(comp_config, "dropout", 0.1),
            use_adaptive_pruning=getattr(comp_config, "use_gate", True),
            gate_init_bias=getattr(comp_config, "gate_init_bias", 1.8),
            is_decoder=is_dec,
        )
        new_attn = AegisAttentionV2(config=cfg)
        if src_attn is not None:
            try:
                new_attn.copy_weights(src_attn)
            except Exception as e:
                logger.error("inject_aegis_v2: copy_weights failed: %s", e)
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
        logger.exception("Failed to inject AEGIS-V2: %s", e)
        return None


AegisAttentionV2 = AegisAttentionV2
inject_aegis_into_distilbert_v2 = inject_aegis_v2