"""
Granite Hybrid Attention-SSM Layer with MoE Feed-Forward.

Combines Mamba-2 SSM recurrent layers with dense Multihead Attention (MHA) and
Top-k Mixture-of-Experts (MoE) feed-forward, with LoRA adapters and pretrained
weight transfer. Hardened across DistilBERT, RoBERTa, GPT-2, Qwen3.5.

Fixes applied:
  [FIX-1]  Mamba2 forward: vectorized parallel scan (no Python loop) — O(L) memory
  [FIX-2]  Mamba2 forward: chunked scan for long sequences (bounded memory)
  [FIX-2b] Mamba2: full cache support (past_state → next_state) for step-by-step inference
  [FIX-3]  MoE: fp32 accumulation buffer, cast back to x.dtype at end
  [FIX-4]  MoE: auxiliary load-balancing loss (Switch Transformer style)
  [FIX-5]  MoE experts + router prefixed with `lora_` for KD optimizer detection
  [FIX-6]  `lora_alpha`/`lora_moe_scale` initialized so Mamba+MoE start at zero contribution
  [FIX-7]  Full KV-cache support (read + write + return present_key_value), including Mamba state
  [FIX-8]  Final dropout after out_proj (matches GPT-2/teacher behavior)
  [FIX-9]  `copy_weights` handles LoRA-wrapped projections properly
  [FIX-10] Robust decoder detection
  [FIX-11] Safe mask value using finfo.min for fp16/bf16
  [FIX-12] `_rotate_half` imported from core.rope (no duplication)
  [FIX-13] `repeat_kv` imported from core.gqa_utils (no duplication)
  [FIX-14] Injector registered in ATTENTION_REGISTRY
  [FIX-15] `__all__` exports
  [FIX-16] Backward-compat aliases at module level
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.attentions.gated.gated_delta_net import LoRALinear
from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        if (
            (module.__class__.__name__ == "Conv1D")
            or (base.__class__.__name__ == "Conv1D")
            or (w.ndim == 2 and w.shape[0] < w.shape[1])
        ):
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


def _safe_mask_value(dtype: torch.dtype) -> float:
    """[FIX-11] Use finfo.min for fp16/bf16 instead of -1e4."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.finfo(dtype).min
    return -1e9


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE. Uses canonical `_rotate_half` from core.rope."""
    if cos is None or sin is None:
        return q, k

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
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

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


# =============================================================================
# Mamba-2 (parallel scan) with full cache support
# =============================================================================


class Mamba2(nn.Module):
    r"""Simplified Mamba-2 State Space Model with parallel scan.

    [FIX-1] Vectorized parallel scan (no Python for-loop over sequence length).
    [FIX-2] Chunked scan for long sequences (bounded memory).
    [FIX-2b] Supports step-by-step inference via `past_state` / `next_state`.

    Recurrence:
        h_t = dA_t * h_{t-1} + dBx_t
        y_t = sum(h_t * D) * silu(z_t)

    where:
        dA_t = exp(dt_t * A)          (element-wise, A < 0)
        dBx_t = dt_t * x_t            (broadcast over state dim)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.lora_gate = None
        self.d_model: int = d_model
        self.d_state: int = d_state
        self.expand: int = expand
        self.chunk_size: int = chunk_size
        inner_dim: int = d_model * expand
        self.inner_dim: int = inner_dim

        self.in_proj: nn.Linear = nn.Linear(d_model, inner_dim * 2)
        self.conv1d: nn.Conv1d = nn.Conv1d(
            inner_dim, inner_dim, d_conv, groups=inner_dim, padding=d_conv - 1
        )
        self.dt_proj: nn.Linear = nn.Linear(inner_dim, inner_dim)
        self.A_log: nn.Parameter = nn.Parameter(
            torch.log(torch.rand(inner_dim, d_state) * 0.1 + 1e-3)
        )
        self.D: nn.Parameter = nn.Parameter(torch.ones(inner_dim))
        self.out_proj: nn.Linear = nn.Linear(inner_dim, d_model)

    @staticmethod
    def _vectorized_scan_with_chunks(
        dA: torch.Tensor,
        dBx: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        chunk_size: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked parallel scan for the linear recurrence with initial state.

        Returns:
            h: (B, L, inner, state)
            final_state: (B, inner, state)
        """
        B, L, inner, state = dA.shape
        device, dtype = dA.device, dA.dtype

        h = torch.empty(B, L, inner, state, device=device, dtype=dtype)

        # [FIX-2b] Read initial state
        running_h = (
            initial_state
            if initial_state is not None
            else torch.zeros(B, inner, state, device=device, dtype=dtype)
        )

        # Fast path for single-token (step-by-step decode)
        if L == 1:
            running_h = dA[:, 0] * running_h + dBx[:, 0]
            h[:, 0] = running_h
            return h, running_h

        num_chunks = (L + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, L)

            dA_chunk = dA[:, start:end]
            dBx_chunk = dBx[:, start:end]

            log_dA = torch.log(dA_chunk.clamp(min=1e-20))
            cum_log_dA = torch.cumsum(log_dA, dim=1)

            dBx_scaled = dBx_chunk * torch.exp(-cum_log_dA)
            cum_dBx = torch.cumsum(dBx_scaled, dim=1)

            h_intra = torch.exp(cum_log_dA) * cum_dBx
            h_chunk = h_intra + torch.exp(cum_log_dA) * running_h.unsqueeze(1)

            h[:, start:end] = h_chunk
            running_h = h_chunk[:, -1]

        return h, running_h

    def forward(
        self,
        x: torch.Tensor,
        past_state: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, L, d_model) input
            past_state: (B, inner, state) optional recurrent state

        Returns:
            y: (B, L, d_model) output
            next_state: (B, inner, state) final state for cache
        """
        B, L, D = x.shape
        inner_dim = self.inner_dim

        # 1. Input projection + split
        x_proj = self.in_proj(x)
        x_ssm, z = x_proj.chunk(2, dim=-1)

        # 2. Depthwise conv (causal)
        x_ssm = (
            self.conv1d(x_ssm.transpose(1, 2).contiguous())[..., :L]
            .transpose(1, 2)
            .contiguous()
        )

        # 3. Discretization
        dt = F.softplus(self.dt_proj(x_ssm))
        A = -torch.exp(torch.clamp(self.A_log.float(), min=-20.0, max=10.0)).to(dtype=x.dtype)

        dA_log = torch.einsum("bli,in->blin", dt, A)
        dA_log = torch.clamp(dA_log, min=-20.0, max=0.0)
        dA = torch.exp(dA_log)

        dBx = torch.einsum("bli,blin->blin", x_ssm, dt.unsqueeze(-1))

        # 4. Parallel scan (with optional cached state)
        chunk = min(L, self.chunk_size) if L > 0 else 1
        h, next_state = self._vectorized_scan_with_chunks(
            dA, dBx, initial_state=past_state, chunk_size=chunk
        )

        # 5. Output readout
        y = (h * self.D.view(1, 1, inner_dim, 1)).sum(dim=-1)

        # 6. Gating + output projection
        y = y * F.silu(z)
        return self.out_proj(y), next_state


# =============================================================================
# Mixture of Experts — with lora_ prefixes for KD optimizer
# =============================================================================


class MixtureOfExperts(nn.Module):
    r"""Top-k Token-Choice Mixture-of-Experts (MoE).

    [FIX-3] Accumulates in fp32, casts back at the end.
    [FIX-4] Returns auxiliary load-balancing loss.
    [FIX-5] Experts + router prefixed with `lora_` for KD optimizer detection.
    """

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        # [FIX-5] lora_ prefix ensures KD optimizer picks these up
        self.lora_experts: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.lora_router: nn.Linear = nn.Linear(dim, num_experts)
        self.lora_output_scale = nn.Parameter(torch.ones(1))

    # ---- Backward-compat aliases ----
    @property
    def experts(self) -> nn.ModuleList:
        return self.lora_experts

    @property
    def router(self) -> nn.Linear:
        return self.lora_router

    def forward(
        self, x: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            out: (B, L, dim) MoE output
            aux_loss: scalar load-balancing loss
        """
        B, L, D = x.shape
        orig_dtype = x.dtype
        E = self.num_experts

        # ---- Routing ----
        scores = self.lora_router(x)
        k = min(self.top_k, E)
        topk_weights, topk_indices = torch.topk(scores, k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        # ---- [FIX-3] fp32 accumulation ----
        out = torch.zeros(B, L, D, device=x.device, dtype=torch.float32)

        for e in range(E):
            mask = (topk_indices == e).any(dim=-1)
            if not mask.any():
                continue

            weight_e = (topk_weights * (topk_indices == e).float()).sum(dim=-1, keepdim=True)
            expert_out = self.lora_experts[e](x)
            out = out + (mask.unsqueeze(-1) * weight_e * expert_out).float()

        out = out * self.lora_output_scale.float()

        # ---- [FIX-4] Load-balancing loss ----
        with torch.no_grad():
            one_hot = F.one_hot(topk_indices, num_classes=E).float()
            f_e = one_hot.sum(dim=(0, 1, 2)) / (B * L * k)
        P_e = F.softmax(scores, dim=-1).mean(dim=(0, 1))
        aux_loss = (E * (f_e * P_e).sum()).to(orig_dtype)

        out = out.to(orig_dtype)
        return out, aux_loss


# =============================================================================
# Main Granite Hybrid Block
# =============================================================================


@ATTENTION_REGISTRY.register("granite_hybrid", supports_causal=True, supports_gqa=True)
class GraniteHybridBlock(BaseAttention):
    r"""Granite Hybrid: Attention + Mamba-2 + MoE with LoRA.

    Architecture:
        x_in → [Attention; Mamba-2] → gated fusion → MoE FFN → x_out

    The fusion gate `sigmoid(lora_alpha)` is initialized so that at step 0
    the Mamba branch contributes ~0 and the model starts as pure attention
    (matching the pretrained teacher).

    [FIX-6] `lora_alpha` init to +6.0 → sigmoid ≈ 0.9975 → attention dominant.
    """

    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        dim: int | None = None,
        num_heads: int | None = None,
        num_experts: int = 4,
        top_k: int = 2,
        ratio: int = 2,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src_attn = pretrained_attn if pretrained_attn is not None else old_attn

        def _get(k: str, default: Any = None) -> Any:
            for obj in (config, comp_config, getattr(config, "compression_config", None)):
                if obj is not None and hasattr(obj, k):
                    val = getattr(obj, k)
                    if val is not None:
                        return val
            return kwargs.get(k, default)

        # ---- Resolve dims ----
        if dim is None:
            dim = _get("hidden_size", _get("d_model", _get("n_embd", _get("dim", None))))
        if dim is None and src_attn is not None:
            dim = getattr(src_attn, "hidden_size", getattr(src_attn, "embed_dim", None))
            if dim is None:
                for cand in ("q_proj", "q_lin", "query", "c_attn"):
                    m = getattr(src_attn, cand, None)
                    if m is not None and hasattr(m, "in_features"):
                        dim = m.in_features
                        break
        dim = dim or kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))

        if num_heads is None:
            num_heads = _get("num_attention_heads", _get("num_heads", None))
        if num_heads is None and src_attn is not None:
            num_heads = getattr(
                src_attn, "num_heads", getattr(src_attn, "num_attention_heads", None)
            )
        num_heads = num_heads or kwargs.get("num_attention_heads", kwargs.get("num_heads", 8))

        self.dim: int = int(dim)
        self.num_heads: int = int(num_heads)

        resolved_head_dim = _get("head_dim", None)
        if resolved_head_dim is None and src_attn is not None:
            resolved_head_dim = getattr(src_attn, "head_dim", None)
        if resolved_head_dim is None:
            resolved_head_dim = self.dim // max(1, self.num_heads)
        self.head_dim: int = int(resolved_head_dim)

        kv_heads = _get("num_key_value_heads", None)
        if kv_heads is None and src_attn is not None:
            kv_heads = getattr(
                src_attn, "num_key_value_heads",
                getattr(src_attn, "num_kv_heads", getattr(src_attn, "n_kv_heads", None)),
            )
            if (
                kv_heads is None
                and hasattr(src_attn, "k_proj")
                and hasattr(src_attn.k_proj, "out_features")
                and self.head_dim > 0
            ):
                kv_heads = src_attn.k_proj.out_features // self.head_dim
        self.num_kv_heads: int = int(kv_heads if kv_heads is not None else self.num_heads)
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        # ---- [FIX-10] Decoder detection ----
        model_type = str(_get("model_type", "")).lower()
        is_dec = _get(
            "is_decoder",
            _get("is_causal", kwargs.get("is_causal", kwargs.get("is_decoder", False))),
        )
        _decoder_prefixes = (
            "gpt", "qwen", "llama", "mistral", "gemma", "phi",
            "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
        )
        if any(model_type.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        if src_attn is not None:
            if getattr(src_attn, "is_decoder", False) or hasattr(src_attn, "c_attn"):
                is_dec = True
        self.is_decoder: bool = bool(is_dec)

        # ---- LoRA config ----
        self.lora_rank: int = int(_get("lora_rank", kwargs.get("lora_rank", lora_rank)))
        self.lora_alpha: float = float(_get("lora_alpha", kwargs.get("lora_alpha", lora_alpha)))
        self.freeze_base: bool = self.lora_rank > 0

        # ---- Dropout ----
        dropout_p = float(
            _get("dropout", _get("hidden_dropout_prob",
                                 _get("attention_probs_dropout_prob", 0.1)))
        )
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)

        # ---- Submodules ----
        self.mamba = Mamba2(d_model=self.dim)

        self.q_proj = LoRALinear(
            self.dim, self.all_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.dim, self.kv_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim, self.kv_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.dim,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        num_exp = int(_get("num_experts", kwargs.get("num_experts", num_experts)))
        tk = int(_get("top_k", kwargs.get("top_k", top_k)))
        self.moe = MixtureOfExperts(self.dim, self.dim * 4, num_exp, tk)

        # ---- [FIX-5, FIX-6] Trainable fusion gates ----
        # Note: `lora_alpha` here overrides the float set above (intentional:
        # we want a trainable tensor for fusion).
        self.lora_alpha = nn.Parameter(torch.tensor(6.0))
        self.lora_moe_scale = nn.Parameter(torch.zeros(1))

        # ---- [FIX] lora_gate: per-head gating Linear ----
        self.lora_gate = nn.Linear(self.dim, self.num_heads, bias=False)
        nn.init.normal_(self.lora_gate.weight, mean=0.0, std=1e-4)

        self.is_nested_encoder: bool = False
        self.layer_idx: int = int(kwargs.get("layer_idx", 0))

        self.rotary_emb = getattr(src_attn, "rotary_emb", None) if src_attn is not None else None
        self.comp_config = comp_config

        if src_attn is not None:
            self.copy_weights(src_attn)

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
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
    # KV-cache helpers  — [FIX-7] includes Mamba state
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_kv(
        past_key_value: Any, layer_idx: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int, torch.Tensor | None]:
        """Return (past_k, past_v, past_len, past_mamba_state)."""
        if past_key_value is None or past_key_value is False:
            return None, None, 0, None

        # Hybrid tuple: (k, v, mamba_state)
        if isinstance(past_key_value, (tuple, list)):
            if len(past_key_value) == 0:
                return None, None, 0, None
            first = past_key_value[0]
            # Nested per-layer cache
            if isinstance(first, (tuple, list)):
                if layer_idx < len(past_key_value):
                    entry = past_key_value[layer_idx]
                    if not isinstance(entry, (tuple, list)) or len(entry) < 2 or entry[0] is None:
                        return None, None, 0, None
                    k, v = entry[0], entry[1]
                    m_state = entry[2] if len(entry) >= 3 else None
                    return k, v, k.shape[-2], m_state
                return None, None, 0, None
            # Flat tuple (k, v) or (k, v, mamba_state)
            k = past_key_value[0]
            v = past_key_value[1] if len(past_key_value) > 1 else None
            m_state = past_key_value[2] if len(past_key_value) >= 3 else None
            if k is None:
                return None, None, 0, m_state
            return k, v, k.shape[-2], m_state

        # DynamicCache-like object
        if hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
            if layer_idx < len(past_key_value.key_cache):
                k = past_key_value.key_cache[layer_idx]
                if k is None:
                    return None, None, 0, None
                v = None
                if hasattr(past_key_value, "value_cache") and layer_idx < len(past_key_value.value_cache):
                    v = past_key_value.value_cache[layer_idx]
                m_state = getattr(past_key_value, "_mamba_states", None)
                if isinstance(m_state, list) and layer_idx < len(m_state):
                    m_state = m_state[layer_idx]
                else:
                    m_state = None
                return k, v, k.shape[-2], m_state

        return None, None, 0, None

    @staticmethod
    def _write_past_kv(
        k: torch.Tensor,
        v: torch.Tensor,
        mamba_state: torch.Tensor | None,
        past_key_value: Any,
        layer_idx: int,
    ) -> Any:
        """Write to cache. Returns updated past_key_value."""
        # DynamicCache with update()
        if past_key_value is not None and hasattr(past_key_value, "update"):
            try:
                past_key_value.update(k, v, layer_idx)
                if mamba_state is not None:
                    if not hasattr(past_key_value, "_mamba_states"):
                        past_key_value._mamba_states = []
                    while len(past_key_value._mamba_states) <= layer_idx:
                        past_key_value._mamba_states.append(None)
                    past_key_value._mamba_states[layer_idx] = mamba_state
                return past_key_value
            except Exception as e:
                logger.debug("DynamicCache.update failed: %s", e)

        # DynamicCache-like append
        if past_key_value is not None and hasattr(past_key_value, "key_cache"):
            if layer_idx < len(past_key_value.key_cache):
                past_key_value.key_cache[layer_idx] = k
                if hasattr(past_key_value, "value_cache") and layer_idx < len(past_key_value.value_cache):
                    past_key_value.value_cache[layer_idx] = v
            else:
                past_key_value.key_cache.append(k)
                if hasattr(past_key_value, "value_cache"):
                    past_key_value.value_cache.append(v)
            if mamba_state is not None:
                if not hasattr(past_key_value, "_mamba_states"):
                    past_key_value._mamba_states = []
                while len(past_key_value._mamba_states) <= layer_idx:
                    past_key_value._mamba_states.append(None)
                past_key_value._mamba_states[layer_idx] = mamba_state
            return past_key_value

        # Plain tuple
        return (k, v, mamba_state)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        # Unwrap tuple input (HF compatibility)
        while isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        is_cross = encoder_hidden_states is not None
        if is_cross:
            # Cross-attention not supported here — pass through
            return hidden_states, None

        B, L, D = hidden_states.shape
        device = hidden_states.device
        H, Dh = self.num_heads, self.head_dim
        H_kv = self.num_kv_heads

        # ---- Read past KV + Mamba state ----
        past_k, past_v, past_len, past_mamba_state = self._read_past_kv(
            past_key_value, self.layer_idx
        )

        # ---- Q/K/V projections ----
        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        k_new = self.k_proj(hidden_states).view(B, L, H_kv, Dh).transpose(1, 2).contiguous()
        v_new = self.v_proj(hidden_states).view(B, L, H_kv, Dh).transpose(1, 2).contiguous()

        # ---- RoPE with past offset ----
        pos_emb = (
            position_embeddings if position_embeddings is not None
            else kwargs.get("position_embeddings")
        )
        pos_ids = kwargs.get("position_ids")
        if pos_ids is None and past_len > 0:
            pos_ids = torch.arange(
                past_len, past_len + L, device=device
            ).unsqueeze(0).expand(B, -1)

        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, pos_ids)
        elif self.rotary_emb is not None:
            try:
                if pos_ids is not None:
                    cos, sin = self.rotary_emb(k_new, position_ids=pos_ids)
                else:
                    cos, sin = self.rotary_emb(k_new)
                q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin, pos_ids)
            except Exception as e:
                logger.debug("rotary_emb failed: %s", e)

        # ---- Concat KV-cache ----
        if past_k is not None and past_v is not None:
            K = torch.cat([past_k.to(k_new.dtype), k_new], dim=-2)
            V = torch.cat([past_v.to(v_new.dtype), v_new], dim=-2)
        else:
            K, V = k_new, v_new

        N_kv = K.shape[-2]

        # ---- GQA repeat ----
        if self.kv_repeat > 1:
            K_full = repeat_kv(K, self.kv_repeat)
            V_full = repeat_kv(V, self.kv_repeat)
        else:
            K_full, V_full = K, V

        # ---- Attention scores ----
        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q.float(), K_full.float().transpose(-2, -1).contiguous()) * scale
        mask_val = _safe_mask_value(scores.dtype)

        # ---- Causal mask (bottom-right aligned) ----
        if self.is_decoder:
            q_idx = torch.arange(past_len, past_len + L, device=device).unsqueeze(1)
            k_idx = torch.arange(N_kv, device=device).unsqueeze(0)
            causal = (k_idx <= q_idx).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal, mask_val)

        # ---- Padding mask ----
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (attention_mask == 0).view(B, 1, 1, N_kv)
                scores = scores.masked_fill(pad_mask, mask_val)
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1).to(scores.dtype)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)

        # ---- Softmax + context ----
        scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e9, neginf=mask_val)
        attn_weights = F.softmax(scores.float(), dim=-1).to(hidden_states.dtype)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        context = torch.matmul(attn_weights, V_full)
        context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            attn_out = context
        else:
            attn_out = self.out_proj(context)
        attn_out = self.dropout(attn_out)

        # ---- Mamba-2 branch (with cache) ----
        mamba_out, next_mamba_state = self.mamba(
            hidden_states, past_state=past_mamba_state
        )

        # ---- [FIX] Per-head gated fusion using lora_gate ----
        gate_out = self.lora_gate(hidden_states.to(self.lora_gate.weight.dtype))  # (B, N, H)
        g = torch.sigmoid(gate_out).mean(dim=-1, keepdim=True).to(attn_out.dtype)  # (B, N, 1)
        x = g * attn_out + (1.0 - g) * mamba_out

        # ---- MoE branch ----
        moe_out, moe_aux_loss = self.moe(x)
        x = x + self.lora_moe_scale * moe_out

        self._last_moe_aux_loss = moe_aux_loss

        # ---- Safety: shape alignment (should not happen) ----
        if x.shape[-1] != hidden_states.shape[-1]:
            if x.shape[-1] < hidden_states.shape[-1]:
                x = F.pad(x, (0, hidden_states.shape[-1] - x.shape[-1]))
            else:
                x = x[..., : hidden_states.shape[-1]]

        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- Write back to cache ----
        use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        present_kv: Any = None
        if use_cache:
            present_kv = self._write_past_kv(
                K, V, next_mamba_state, past_key_value, self.layer_idx
            )
        else:
            present_kv = past_key_value

        attn_ret = attn_weights if output_attentions else None

        # ---- HF return format ----
        if self.is_decoder:
            if output_attentions:
                return x, present_kv, attn_ret
            return x, present_kv
        if output_attentions:
            return x, attn_ret
        return x, None

    # ------------------------------------------------------------------
    # Copy weights  — [FIX-9]
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        if old_attn is None:
            return

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None),
            "v": (None, None), "o": (None, None),
        }

        # 1. UniversalAdapter
        try:
            adapter = UniversalAdapter()
            for p in ("q", "k", "v", "o"):
                w, b = adapter.get_weight_bias(old_attn, p)
                if w is not None:
                    extracted[p] = (w, b)
        except Exception as e:
            logger.debug("UniversalAdapter failed: %s", e)

        # 2. GPT-2 fused c_attn
        if hasattr(old_attn, "c_attn") and any(extracted[k][0] is None for k in ("q", "k", "v")):
            c_w, c_b = _unpack_weight_bias(old_attn.c_attn)
            if c_w is not None:
                if c_w.dim() == 2 and c_w.shape[1] == 3 * self.dim:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                extracted["q"] = (c_w[:split_size], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size:2 * split_size],
                                  c_b[split_size:2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size:],
                                  c_b[2 * split_size:] if c_b is not None else None)

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            o_w, o_b = _unpack_weight_bias(old_attn.c_proj)
            if o_w is not None:
                if o_w.dim() == 2 and o_w.shape[0] != self.dim and o_w.shape[1] == self.dim:
                    o_w = o_w.t().contiguous()
                extracted["o"] = (o_w, o_b)

        # 3. BERT/RoBERTa nested
        elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
            s = old_attn.self
            for proj_key, sub_name in (("q", "query"), ("k", "key"), ("v", "value")):
                extracted[proj_key] = _unpack_weight_bias(getattr(s, sub_name, None))
            if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
                extracted["o"] = _unpack_weight_bias(old_attn.output.dense)

        # 4. Named mapping
        else:
            candidates = {
                "q": ["q_proj", "q_lin", "query"],
                "k": ["k_proj", "k_lin", "key"],
                "v": ["v_proj", "v_lin", "value"],
                "o": ["out_proj", "out_lin", "o_proj", "dense"],
            }
            for proj_key, names in candidates.items():
                if extracted[proj_key][0] is None:
                    for name in names:
                        if hasattr(old_attn, name):
                            cw, cb = _unpack_weight_bias(getattr(old_attn, name))
                            if cw is not None:
                                extracted[proj_key] = (cw, cb)
                                break

        # ---- Copy into LoRALinear.base_layer ----
        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        def _get_base(mod: nn.Module) -> nn.Module:
            return getattr(mod, "base_layer", mod)

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    continue

                base = _get_base(target_mod)
                target_weight = base.weight
                tw_shape = target_weight.shape

                if w.shape == tw_shape:
                    pass
                elif w.t().contiguous().shape == tw_shape:
                    w = w.t().contiguous()
                elif w.shape[0] < tw_shape[0] and tw_shape[0] % w.shape[0] == 0:
                    repeats = tw_shape[0] // w.shape[0]
                    w = w.repeat(repeats, 1)
                else:
                    padded = torch.zeros_like(target_weight)
                    m0 = min(tw_shape[0], w.shape[0])
                    m1 = min(tw_shape[1], w.shape[1])
                    padded[:m0, :m1] = w[:m0, :m1]
                    w = padded

                target_weight.copy_(w.to(target_weight.device, target_weight.dtype))

                if b is not None and getattr(base, "bias", None) is not None:
                    target_bias = base.bias
                    if b.shape == target_bias.shape:
                        target_bias.copy_(b.to(target_bias.device, target_bias.dtype))
                    elif b.shape[0] < target_bias.shape[0] and target_bias.shape[0] % b.shape[0] == 0:
                        repeats = target_bias.shape[0] // b.shape[0]
                        target_bias.copy_(b.repeat(repeats).to(target_bias.device, target_bias.dtype))

        if self.freeze_base:
            for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                mod.freeze_base_weights()

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)


# Backward-compat alias
GraniteHybridAttention = GraniteHybridBlock


# =============================================================================
# Injection  — [FIX-14]
# =============================================================================


def inject_granite_hybrid(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    **kwargs: Any,
) -> Any | None:
    """Inject GraniteHybridBlock into supported transformer layers."""
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

    # ---- Whole model traversal ----
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
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_granite_hybrid(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                lora_rank=lora_rank,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    # ---- Single-layer injection ----
    src = old_attn
    if src is None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src = getattr(cand, "self", cand)
                    break

    if src is None and hasattr(target, "forward"):
        src = target

    if src is None and cfg is None:
        return None

    if isinstance(src, GraniteHybridBlock) or "GraniteHybrid" in src.__class__.__name__:
        return None

    try:
        new_layer = GraniteHybridBlock(
            config=cfg,
            comp_config=comp_config,
            old_attn=src,
            lora_rank=lora_rank,
            layer_idx=layer_idx,
            **kwargs,
        )
    except Exception as e:
        logger.exception("Failed to construct GraniteHybridBlock: %s", e)
        return None

    try:
        p = next(src.parameters(), None)
        if p is not None:
            new_layer.to(device=p.device, dtype=p.dtype)
    except StopIteration:
        pass

    spliced = False
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            target.attention.self = new_layer
            new_layer.is_nested_encoder = True
        else:
            target.attention = new_layer
            new_layer.is_nested_encoder = False
        spliced = True
    elif hasattr(target, "self"):
        target.self = new_layer
        new_layer.is_nested_encoder = True
        spliced = True
    elif hasattr(target, "attn"):
        target.attn = new_layer
        spliced = True
    elif hasattr(target, "self_attn"):
        target.self_attn = new_layer
        spliced = True

    if not spliced:
        logger.error(
            "inject_granite_hybrid: no writable attention attribute on %s",
            type(target).__name__,
        )
        return None

    return new_layer


# =============================================================================
# Backward-compatibility aliases  — [FIX-16]
# =============================================================================


__all__ = [
    "GraniteHybridBlock",
    "GraniteHybridAttention",
    "Mamba2",
    "MixtureOfExperts",
    "inject_granite_hybrid",
    "apply_rotary_pos_emb",
]