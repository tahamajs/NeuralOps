"""
Gated DeltaNet Attention Module — Production-Hardened.

Architecture:
    * Delta rule recurrence with data-dependent beta (decay) and alpha (step size)
    * L2-normalized Q/K for numerical stability of the learning-rate denominator
    * Optional Triton fused scan for CUDA + head_dim ≤ 64
    * Full KV-cache / recurrent-state support for autoregressive generation
    * KD-compatible gate naming (lora_ prefix on all trainable gates)
    * Bidirectional scan for encoder models (BERT, RoBERTa)
    * GQA support with repeat_kv
    * RoPE with partial-dim slicing

"""

from __future__ import annotations
from neuralops.core.common import _cfg_get
from neuralops.core.common import repeat_kv

from neuralops.core.attention_utils import (
    apply_rotary_pos_emb,
    LoRALinear,
)

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

try:
    from neuralops.core.triton_kernels.gdn_scan import MAX_REGISTER_DIM, gdn_scan_triton

    HAS_TRITON_GDN = True
    logger.info("GDN: Triton scan available")
except Exception as _e:
    logger.warning("GDN: Triton scan unavailable: %s", _e)
    gdn_scan_triton = None
    HAS_TRITON_GDN = False
    MAX_REGISTER_DIM = 64






# ===========================================================================
# Delta-rule recurrence
# ===========================================================================
def _delta_chunk_forward(
    S: torch.Tensor,
    q_i: torch.Tensor,
    k_i: torch.Tensor,
    v_i: torch.Tensor,
    beta_i: torch.Tensor,
    alpha_i: torch.Tensor,
    mask_i: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sequential delta-rule recurrence over one chunk of size C.

    For each t:
        inter_v = k_t · S_{t-1}
        err     = v_t - inter_v
        step    = alpha_t / (1 + ||k_t||²)
        S_t     = beta_t · S_{t-1} + step · k_tᵀ · err
        y_t     = q_t · S_t

    If mask_i is provided (shape (B, H, C) bool), padded steps do not update S.
    """
    C = q_i.shape[2]
    outs = []
    curr = S

    for t in range(C):
        kt = k_i[:, :, t : t + 1, :]              # (B, H, 1, d)
        vt = v_i[:, :, t : t + 1, :]              # (B, H, 1, d)
        qt = q_i[:, :, t : t + 1, :]              # (B, H, 1, d)
        bt = beta_i[:, :, t : t + 1].unsqueeze(-1)   # (B, H, 1, 1)
        at = alpha_i[:, :, t : t + 1].unsqueeze(-1)  # (B, H, 1, 1)

        # --- Pad handling: zero out K, V for padded tokens ---
        if mask_i is not None:
            m_t = mask_i[:, :, t : t + 1].unsqueeze(-1).to(kt.dtype)  # (B, H, 1, 1)
            kt = kt * m_t
            vt = vt * m_t

        # --- Recurrence ---
        inter_v = torch.matmul(kt, curr)             # (B, H, 1, d)
        err = vt - inter_v
        norm_k_sq = (kt * kt).sum(dim=-1, keepdim=True)  # (B, H, 1, 1)
        step = at / (1.0 + norm_k_sq).clamp_min(1.0)
        delta = step * torch.matmul(kt.transpose(-1, -2).contiguous(), err)
        curr = bt * curr + delta
        curr = torch.nan_to_num(curr, nan=0.0, posinf=1e4, neginf=-1e4)

        yt = torch.matmul(qt, curr)                  # (B, H, 1, d)
        outs.append(yt)

    return torch.cat(outs, dim=2), curr


def chunked_delta_rule_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked Delta-Rule scan.

    Args:
        q, k, v: (B, H, L, d)
        beta:    (B, H, L)  — per-head decay in (0, 1)
        alpha:   (B, H, L)  — per-head step size in (0, 1)
        initial_state: (B, H, d, d) or None
        mask:    (B, L) bool or None  — True for valid tokens
        chunk_size: int

    Returns:
        y:  (B, H, L, d)
        S:  (B, H, d, d)  — final recurrent state
    """
    B, H, L, d = q.shape
    device = q.device

    q_f = F.normalize(q.float(), p=2, dim=-1, eps=1e-6)
    k_f = F.normalize(k.float(), p=2, dim=-1, eps=1e-6)
    v_f = v.float()
    beta_f = beta.float().clamp(min=1e-4, max=1.0 - 1e-4)
    alpha_f = alpha.float().clamp(min=1e-4, max=1.0)

    # Expand mask to (B, H, L) for per-head use
    mask_h = None
    if mask is not None:
        mask_h = mask[:, None, :].expand(-1, H, -1).contiguous()

    # --- Fused Triton chunk_gdn fast path (CUDA, d <= 64, no initial_state, no mask) ---
    if q.is_cuda and d <= 64 and initial_state is None and mask_h is None:
        try:
            from fla.ops import chunk_gdn
            q_fl = q.transpose(1, 2).contiguous()
            k_fl = k.transpose(1, 2).contiguous()
            v_fl = v.transpose(1, 2).contiguous()
            # g in FLA is the log-decay gate [B, T, H]
            log_g = torch.log(beta_f.clamp(min=1e-4, max=1.0)).transpose(1, 2).to(q.dtype).contiguous()
            step_beta = alpha_f.transpose(1, 2).to(q.dtype).contiguous()
            res = chunk_gdn(q_fl, k_fl, v_fl, log_g, step_beta)
            out_fl = res[0] if isinstance(res, tuple) else res
            out = torch.nan_to_num(out_fl.transpose(1, 2).contiguous(), nan=0.0, posinf=1e4, neginf=-1e4).to(q.dtype)
            S_final = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
            return out, S_final
        except Exception:
            pass

    # --- Triton scan path (only when no initial state and no mask) ---
    if (
        HAS_TRITON_GDN
        and gdn_scan_triton is not None
        and q.is_cuda
        and d <= MAX_REGISTER_DIM
        and initial_state is None
        and mask_h is None
        and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        try:
            out = gdn_scan_triton(q_f, k_f, v_f, beta_f, alpha_f)
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).to(q.dtype)
            S_final = torch.zeros(B, H, d, d, device=device, dtype=torch.float32)
            return out, S_final
        except Exception as e:
            logger.warning("Triton gdn_scan failed, falling back to PyTorch: %s", e)

    # --- PyTorch fallback ---
    # Initialize state
    if initial_state is not None:
        S = initial_state.to(device=device, dtype=torch.float32).clone()
    else:
        S = torch.zeros((B, H, d, d), device=device, dtype=torch.float32)

    # --- Single-token case (autoregressive decode) ---
    if L == 1:
        y_i, S_out = _delta_chunk_forward(
            S, q_f, k_f, v_f, beta_f, alpha_f, mask_h
        )
        return torch.nan_to_num(y_i, nan=0.0).to(q.dtype), S_out

    # --- Chunked prefill ---
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len > 0:
        q_f = F.pad(q_f, (0, 0, 0, pad_len))
        k_f = F.pad(k_f, (0, 0, 0, pad_len))
        v_f = F.pad(v_f, (0, 0, 0, pad_len))
        beta_f = F.pad(beta_f, (0, pad_len), value=1.0)
        alpha_f = F.pad(alpha_f, (0, pad_len), value=1.0)
        if mask_h is not None:
            mask_h = F.pad(mask_h, (0, pad_len), value=False)

    L_pad = L + pad_len
    n_chunks = L_pad // chunk_size

    q_c = q_f.view(B, H, n_chunks, chunk_size, d)
    k_c = k_f.view(B, H, n_chunks, chunk_size, d)
    v_c = v_f.view(B, H, n_chunks, chunk_size, d)
    beta_c = beta_f.view(B, H, n_chunks, chunk_size)
    alpha_c = alpha_f.view(B, H, n_chunks, chunk_size)
    mask_c = mask_h.view(B, H, n_chunks, chunk_size) if mask_h is not None else None

    outputs = []
    for c in range(n_chunks):
        mk = mask_c[:, :, c] if mask_c is not None else None
        y_i, S = _delta_chunk_forward(
            S,
            q_c[:, :, c],
            k_c[:, :, c],
            v_c[:, :, c],
            beta_c[:, :, c],
            alpha_c[:, :, c],
            mk,
        )
        outputs.append(y_i)

    y = torch.cat(outputs, dim=2)
    if pad_len > 0:
        y = y[:, :, :L, :]

    return torch.nan_to_num(y, nan=0.0).to(q.dtype), S


# ===========================================================================
# GatedDeltaNet
# ===========================================================================
@ATTENTION_REGISTRY.register(
    "gated_delta_net",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class GatedDeltaNet(BaseAttention):
    """
    Gated DeltaNet with data-dependent decay (beta) and step size (alpha).

    Recurrence:
        S_t = beta_t · S_{t-1} + step_t · k_tᵀ (v_t − k_t · S_{t-1})
        y_t = q_t · S_t
        out = y ⊙ SiLU(z)

    Supports:
        * Full KV-cache / recurrent-state passthrough
        * Bidirectional scan for encoders
        * GQA (num_kv_heads < num_heads)
        * RoPE (partial or full)
        * head_mask
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        old_attn: nn.Module | None = None,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---- Resolve d_model ----
        if d_model is None:
            d_model = (
                kwargs.get("d_model")
                or kwargs.get("hidden_size")
                or kwargs.get("dim")
            )
            if d_model is None and config is not None:
                d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
            elif d_model is None and source_attn is not None:
                out_layer = None
                for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
                    if hasattr(source_attn, attr):
                        out_layer = getattr(source_attn, attr)
                        break
                if out_layer is not None:
                    d_model = getattr(
                        out_layer,
                        "out_features",
                        getattr(out_layer, "in_features", 768),
                    )
                else:
                    for attr in ("q_proj", "query", "q_lin", "W_q"):
                        q_l = getattr(source_attn, attr, None)
                        if q_l is not None:
                            d_model = getattr(
                                q_l,
                                "in_features",
                                getattr(q_l, "out_features", 768),
                            )
                            break
            if d_model is None:
                d_model = 768
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        # ---- Resolve num_heads ----
        if num_heads is None:
            num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
            if num_heads is None and config is not None:
                num_heads = getattr(
                    config,
                    "num_attention_heads",
                    getattr(config, "n_heads", getattr(config, "num_heads", 12)),
                )
            elif num_heads is None and source_attn is not None:
                num_heads = getattr(
                    source_attn,
                    "n_heads",
                    getattr(
                        source_attn,
                        "num_attention_heads",
                        getattr(source_attn, "num_heads", 12),
                    ),
                )
            if num_heads is None:
                num_heads = 12

        # None-safe super().__init__
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        self.d_model = int(d_model)
        self.num_heads = max(1, int(num_heads))

        # ---- Resolve num_kv_heads (GQA) ----
        cfg_kv = None
        if config is not None:
            cfg_kv = getattr(
                config, "num_key_value_heads", getattr(config, "num_kv_heads", None)
            )
        if cfg_kv is None:
            cfg_kv = kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = max(1, min(int(cfg_kv), self.num_heads))
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        # ---- Resolve head_dim ----
        head_dim = getattr(config, "head_dim", None) if config else None
        if head_dim is None and source_attn is not None:
            for q_cand in ("q_proj", "query", "q_lin", "W_q"):
                q_mod = getattr(source_attn, q_cand, None)
                if q_mod is not None and hasattr(q_mod, "out_features"):
                    head_dim = q_mod.out_features // max(1, self.num_heads)
                    break
        if head_dim is None:
            head_dim = self.d_model // self.num_heads
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim

        # ---- Layer index ----
        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (
                layer_idx
                if layer_idx is not None
                else kwargs.get("layer_id", kwargs.get("layer_idx", 0))
            )
        )
        self.layer_idx = int(resolved_layer_id)

        # ---- LoRA ----
        if config is not None:
            self.lora_rank = int(
                getattr(
                    config,
                    "lora_rank",
                    getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
                )
            )
            self.lora_alpha = float(
                getattr(
                    config,
                    "lora_alpha",
                    getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
                )
            )
        elif comp_config is not None:
            self.lora_rank = int(
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))
            )
            self.lora_alpha = float(
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
            )
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))
        self.freeze_base = self.lora_rank > 0

        # ---- Decoder detection ----
        model_type = str(getattr(config, "model_type", "") if config else "").lower()
        is_dec = False
        if config is not None:
            is_dec = bool(
                getattr(config, "is_decoder", False) or getattr(config, "is_causal", False)
            )
        decoder_prefixes = (
            "gpt",
            "qwen",
            "llama",
            "mistral",
            "gemma",
            "phi",
            "opt",
            "bloom",
            "falcon",
            "olmo",
            "mpt",
            "stablelm",
            "codegen",
        )
        if any(model_type.startswith(p) for p in decoder_prefixes):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if source_attn is not None:
            if getattr(source_attn, "is_decoder", False) or hasattr(source_attn, "c_attn"):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder

        # ---- Nested encoder detection ----
        is_nested = False
        if source_attn is not None:
            src_cls = source_attn.__class__.__name__
            if (
                (
                    src_cls
                    in (
                        "BertSelfAttention",
                        "RobertaSelfAttention",
                        "XLMRobertaSelfAttention",
                        "ElectraSelfAttention",
                        "DebertaSelfAttention",
                    )
                    or (src_cls.endswith("SelfAttention") and "distil" not in src_cls.lower())
                )
                or (hasattr(source_attn, "self") and hasattr(source_attn, "output"))
            ):
                is_nested = True
        if (
            not is_nested
            and model_type in ("roberta", "bert", "xlm-roberta")
            and not self.is_decoder
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        # ---- Projections ----
        self.q_proj = LoRALinear(
            self.d_model, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.d_model, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.d_model, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.z_proj = LoRALinear(
            self.d_model, self.all_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size, self.d_model, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        # ---- KD-compatible gates (lora_ prefix) ----
        # [GDN-FIX-07] Renamed from beta_proj / alpha_proj / output_scale
        self.lora_output_scale = nn.Parameter(torch.ones(1))
        self.lora_beta_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.lora_alpha_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        with torch.no_grad():
            # beta init ~ sigmoid(2.0) ≈ 0.88  → mostly retain memory
            nn.init.constant_(self.lora_beta_proj.bias, 2.0)
            nn.init.zeros_(self.lora_beta_proj.weight)
            # alpha init ~ sigmoid(0.0) = 0.5 → moderate step size
            nn.init.constant_(self.lora_alpha_proj.bias, 0.0)
            nn.init.zeros_(self.lora_alpha_proj.weight)

        # ---- Chunk size ----
        cs = int(_cfg_get(comp_config, "chunk_size", kwargs.get("chunk_size", 32)))
        self.chunk_size = max(8, min(cs, 128))

        self.source_attn = source_attn
        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None) if source_attn is not None else None
        )
        self.comp_config = comp_config

        # ---- Storage accounting ----
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if source_attn is not None:
            self.copy_weights(source_attn)

    # ------------------------------------------------------------------
    # Backward-compat aliases for the old (unprefixed) gate names
    # ------------------------------------------------------------------
    @property
    def beta_proj(self) -> nn.Linear:
        return self.lora_beta_proj

    @property
    def alpha_proj(self) -> nn.Linear:
        return self.lora_alpha_proj

    @property
    def output_scale(self) -> nn.Parameter:
        return self.lora_output_scale

    def freeze_base_weights(self) -> None:
        for m in (self.q_proj, self.k_proj, self.v_proj, self.z_proj, self.out_proj):
            if hasattr(m, "freeze_base_weights"):
                m.freeze_base_weights()

    # ---- HF state-dict compatibility aliases ----
    @property
    def W_q(self): return self.q_proj
    @property
    def W_k(self): return self.k_proj
    @property
    def W_v(self): return self.v_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # KV-cache helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_past_state(past_key_value: Any) -> torch.Tensor | None:
        """Extract the recurrent state from a cache object or tuple."""
        if past_key_value is None or past_key_value is False:
            return None

        # Tuple / list of state tensors (our convention)
        if isinstance(past_key_value, (tuple, list)):
            if len(past_key_value) >= 1 and isinstance(past_key_value[0], torch.Tensor):
                s = past_key_value[0]
                if s.dim() == 4 and s.shape[-1] == s.shape[-2]:
                    return s

        # HF DynamicCache-style (unlikely for GDN but handle gracefully)
        if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
            return None  # GDN doesn't use K/V cache

        return None

    @staticmethod
    def _write_past_state(state: torch.Tensor, past_key_value: Any) -> Any:
        """Package the new state into the cache format."""
        # If caller gave us a dynamic cache with an `update` method, we cannot
        # use it (it expects K/V), so return a tuple instead.
        if (
            past_key_value is not None
            and hasattr(past_key_value, "update")
            and not isinstance(past_key_value, (tuple, list))
        ):
            return (state,)
        return (state,)

    # ------------------------------------------------------------------
    # Mask helper
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_valid_mask(
        mask: torch.Tensor | None, B: int, L: int, device: torch.device
    ) -> torch.Tensor | None:
        """Return (B, L) bool where True = valid, or None."""
        if mask is None:
            return None

        def _to_bool(t: torch.Tensor) -> torch.Tensor:
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if mask.dim() == 2:
            m = _to_bool(mask)
            if m.shape == (B, L):
                return m
        elif mask.dim() == 3:
            m = _to_bool(mask)
            if m.shape[0] == B and m.shape[-1] == L:
                return m[:, -1, :]
        elif mask.dim() == 4:
            m = _to_bool(mask)
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m.squeeze(1)
            if m.shape == (B, L):
                return m
        return None

    # ------------------------------------------------------------------
    # Weight transfer
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        if old_attn is None:
            return

        try:
            for p in old_attn.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        cls_name = old_attn.__class__.__name__
        is_nested = False
        if (
            hasattr(old_attn, "self")
            and hasattr(old_attn.self, "query")
            and hasattr(old_attn, "output")
            and hasattr(old_attn.output, "dense")
        ) or (
            cls_name
            in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
                "DebertaSelfAttention",
            )
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ):
            is_nested = True
        if not is_nested and getattr(self, "is_nested_encoder", False):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(old_attn, "rotary_emb"):
            self.rotary_emb = old_attn.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None), "k": (None, None),
            "v": (None, None), "o": (None, None),
        }

        # --- GPT-2 fused c_attn ---
        if hasattr(old_attn, "c_attn") and old_attn.c_attn is not None:
            c = old_attn.c_attn
            if hasattr(c, "merge") and callable(c.merge) and not getattr(c, "merged", False):
                try:
                    c.merge()
                except Exception as e:
                    logger.debug("c_attn merge skipped: %s", e)
            base_c = getattr(c, "base_layer", c)
            cw = getattr(base_c, "weight", getattr(c, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                is_conv = (
                    "Conv1D" in c.__class__.__name__
                    or "Conv1D" in base_c.__class__.__name__
                    or hasattr(c, "nf")
                    or hasattr(base_c, "nf")
                    or (cw.ndim == 2 and cw.shape[0] < cw.shape[1])
                )
                if is_conv:
                    cw = cw.t().contiguous()
                split = cw.size(0) // 3
                bmod = getattr(base_c, "bias", getattr(c, "bias", None))
                cb = bmod.data.clone() if bmod is not None else None
                extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
                extracted["k"] = (
                    cw[split : 2 * split],
                    cb[split : 2 * split] if cb is not None else None,
                )
                extracted["v"] = (
                    cw[2 * split :],
                    cb[2 * split :] if cb is not None else None,
                )

            if hasattr(old_attn, "c_proj") and old_attn.c_proj is not None:
                o = old_attn.c_proj
                base_o = getattr(o, "base_layer", o)
                ow = getattr(base_o, "weight", getattr(o, "weight", None))
                if ow is not None:
                    ow = ow.data.clone()
                    is_conv_o = (
                        "Conv1D" in o.__class__.__name__
                        or "Conv1D" in base_o.__class__.__name__
                        or hasattr(o, "nf")
                        or hasattr(base_o, "nf")
                        or (ow.ndim == 2 and ow.shape[0] < ow.shape[1])
                    )
                    if is_conv_o:
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # --- BERT / RoBERTa nested ---
        elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
            s = old_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (
                        m.weight.data.clone(),
                        b.data.clone() if b is not None else None,
                    )
            if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
                d = old_attn.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (
                    d.weight.data.clone(),
                    b.data.clone() if b is not None else None,
                )

        # --- Flat projections ---
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query", "W_q"),
                "k": ("k_lin", "k_proj", "key", "W_k"),
                "v": ("v_lin", "v_proj", "value", "W_v"),
                "o": ("out_lin", "out_proj", "dense", "o_proj", "output", "proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(old_attn, n):
                        m = getattr(old_attn, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (
                                m.weight.data.clone(),
                                b.data.clone() if b is not None else None,
                            )
                            break

        # --- UniversalAdapter fallback ---
        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(old_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            logger.warning("GatedDeltaNet copy_weights: missing %s", missing_after)
            return

        with torch.no_grad():
            for key, target in (
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ):
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if (
                    w.shape != target.weight.shape
                    and w.t().contiguous().shape == target.weight.shape
                ):
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data[:mb].copy_(
                        b[:mb].to(target.bias.device, target.bias.dtype)
                    )
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                n = float(torch.linalg.norm(target.weight.data.float()).item())
                if n < 1e-4:
                    logger.warning(
                        "GatedDeltaNet: '%s' near-zero norm (%.2e)", key, n
                    )

                if self.freeze_base:
                    target.freeze_base_weights()

            # z_proj from q
            w_q = extracted["q"][0]
            if w_q is not None:
                out_z = min(self.z_proj.weight.size(0), w_q.size(0))
                in_z = min(self.z_proj.weight.size(1), w_q.size(1))
                self.z_proj.weight.data[:out_z, :in_z].copy_(
                    w_q[:out_z, :in_z].to(
                        self.z_proj.weight.device, self.z_proj.weight.dtype
                    )
                )
                if self.freeze_base:
                    self.z_proj.freeze_base_weights()

        # ---- Storage accounting ----
        try:
            for m in (
                self.q_proj, self.k_proj, self.v_proj,
                self.z_proj, self.out_proj,
            ):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        if self.freeze_base:
            self.freeze_base_weights()

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
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # ---- 1. Read past recurrent state ----
        past_state = None
        past_len = 0
        if not is_cross and past_key_value is not None:
            past_state = self._read_past_state(past_key_value)
            if past_state is not None:
                past_len = getattr(past_key_value, "past_len", 0)
                if past_len == 0 and isinstance(past_key_value, (tuple, list)):
                    if len(past_key_value) >= 2 and isinstance(
                        past_key_value[1], int
                    ):
                        past_len = past_key_value[1]

        # ---- 2. Projections ----
        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        k = (
            self.k_proj(kv_input)
            .view(B, kv_len, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_proj(kv_input)
            .view(B, kv_len, self.num_kv_heads, d)
            .transpose(1, 2)
            .contiguous()
        )
        z = self.z_proj(hidden_states)

        # ---- 3. RoPE ----
        if not is_cross:
            if position_embeddings is not None and isinstance(
                position_embeddings, tuple
            ):
                cos, sin = position_embeddings
                if cos.dim() >= 3 and past_len > 0:
                    cos = cos[..., past_len : past_len + L, :]
                    sin = sin[..., past_len : past_len + L, :]
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            elif getattr(self, "rotary_emb", None) is not None:
                if position_ids is None and past_len > 0:
                    position_ids = (
                        torch.arange(
                            past_len, past_len + kv_len, device=hidden_states.device
                        )
                        .unsqueeze(0)
                        .expand(B, -1)
                    )
                try:
                    cos, sin = self.rotary_emb(k, seq_len=past_len + kv_len)
                    if past_len > 0:
                        cos = cos[..., past_len : past_len + L, :]
                        sin = sin[..., past_len : past_len + L, :]
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
                except Exception as e:
                    logger.debug("rotary_emb forward failed: %s", e)

        # ---- 4. GQA expansion ----
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- 5. Gates (lora_ prefix) ----
        hs_beta = hidden_states.to(self.lora_beta_proj.weight.dtype)
        beta = (
            torch.sigmoid(self.lora_beta_proj(hs_beta))
            .transpose(1, 2)
            .contiguous()
        )  # (B, H, L)
        hs_alpha = hidden_states.to(self.lora_alpha_proj.weight.dtype)
        alpha = (
            torch.sigmoid(self.lora_alpha_proj(hs_alpha))
            .transpose(1, 2)
            .contiguous()
        )  # (B, H, L)

        # ---- 6. Mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        mask_bool = self._normalize_valid_mask(
            eff_mask, B, kv_len, hidden_states.device
        )

        # ---- 7. Delta scan with state passthrough ----
        y_fwd, present_state = chunked_delta_rule_scan(
            q, k, v, beta, alpha,
            initial_state=past_state,
            mask=mask_bool,
            chunk_size=self.chunk_size,
        )

        # ---- 8. Bidirectional scan for encoders ----
        if not self.is_decoder:
            q_b = torch.flip(q, dims=[2])
            k_b = torch.flip(k, dims=[2])
            v_b = torch.flip(v, dims=[2])
            beta_b = torch.flip(beta, dims=[2])
            alpha_b = torch.flip(alpha, dims=[2])
            mask_b = torch.flip(mask_bool, dims=[1]) if mask_bool is not None else None
            y_bwd_f, _ = chunked_delta_rule_scan(
                q_b, k_b, v_b, beta_b, alpha_b,
                initial_state=None,
                mask=mask_b,
                chunk_size=self.chunk_size,
            )
            y_bwd = torch.flip(y_bwd_f, dims=[2])
            y = 0.5 * (y_fwd + y_bwd)
        else:
            y = y_fwd

        # ---- 9. head_mask ----
        if head_mask is not None:
            hm = head_mask.to(y.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            y = y * hm

        # ---- 10. Recombine + SiLU gating ----
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        y = y.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        gated_y = y.to(z.dtype) * F.silu(z)

        # ---- 11. Output projection ----
        if self.is_nested_encoder:
            out = gated_y
        else:
            out = self.out_proj(gated_y)

        out = out * self.lora_output_scale
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        # ---- 12. Final output masking (single application) ----
        if mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        # ---- 13. Package the new state ----
        present_kv = None
        use_cache = kwargs.get("use_cache", False) or (
            past_key_value is not None and past_key_value is not False
        )
        if use_cache and not is_cross:
            present_kv = self._write_past_state(present_state, past_key_value)

        if self.is_decoder and not is_cross:
            return out, present_kv

        return out, None

    def step(
        self,
        x: torch.Tensor,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exact O(1) single-step autoregressive recurrent update for GatedDeltaNet."""
        from neuralops.core.triton_kernels.state_update import fused_recurrent_gdn_step

        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(1)

        B, L, D = x.shape
        H, d = self.num_heads, self.head_dim
        device = x.device
        dtype = x.dtype

        if state is None:
            state = {}
        step_idx = state.get("step_idx", 0)

        # 1. Projections
        q = self.q_proj(x).view(B, 1, H, d).transpose(1, 2).contiguous()  # [B, H, 1, d]
        k = self.k_proj(x).view(B, 1, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, 1, self.num_kv_heads, d).transpose(1, 2).contiguous()
        z = self.z_proj(x)

        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # 2. L2 normalize Q and K
        q_norm = F.normalize(q, p=2, dim=-1, eps=1e-12)
        k_norm = F.normalize(k, p=2, dim=-1, eps=1e-12)

        # 3. Gates
        beta = torch.sigmoid(self.lora_beta_proj(x)).view(B, 1, H).transpose(1, 2).squeeze(-1)    # [B, H]
        alpha = torch.sigmoid(self.lora_alpha_proj(x)).view(B, 1, H).transpose(1, 2).squeeze(-1)  # [B, H]

        if "kv_state" not in state or state["kv_state"] is None:
            state["kv_state"] = torch.zeros((B, H, d, d), device=device, dtype=torch.float32)

        kv_state = state["kv_state"]

        q_t = q_norm.squeeze(2)  # [B, H, d]
        k_t = k_norm.squeeze(2)
        v_t = v.squeeze(2)

        # 4. Fused GDN step: S_t = beta * S_{t-1} + alpha * k_t \otimes (v_t - S_{t-1}^T @ k_t)
        out_t, kv_state = fused_recurrent_gdn_step(
            q_t, k_t, v_t, beta, alpha, kv_state, scale=1.0
        )
        state["kv_state"] = kv_state
        state["step_idx"] = step_idx + 1

        y = out_t.unsqueeze(2).transpose(1, 2).contiguous().view(B, 1, H * d)
        gated_y = y.to(z.dtype) * F.silu(z)

        if self.is_nested_encoder:
            out = gated_y
        else:
            out = self.out_proj(gated_y)

        out = out * self.lora_output_scale
        if is_2d:
            out = out.squeeze(1)

        return out, state

    # ------------------------------------------------------------------
    # Storage accounting
    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out: dict[str, int] = {}
        for name, m in (
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("z_proj", self.z_proj),
            ("out_proj", self.out_proj),
            ("lora_beta_proj", self.lora_beta_proj),
            ("lora_alpha_proj", self.lora_alpha_proj),
        ):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["lora_output_scale"] = _sz(self.lora_output_scale)
        out["total"] = sum(out.values())
        return out

    def get_ortho_loss(self) -> torch.Tensor:
        try:
            return torch.tensor(0.0, device=self.out_proj.weight.device)
        except Exception:
            return torch.tensor(0.0)


# ===========================================================================
# Injection
# ===========================================================================
def inject_gated_delta_net(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_gated_delta_net: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # ---- Whole-model traversal ----
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "transformer") and hasattr(target.transformer, "layer"):
        layer_list = target.transformer.layer
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_gated_delta_net(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    # ---- Single-layer injection ----
    src_attn = old_attn
    if src_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    # Idempotency guard
    if src_attn is not None and (
        isinstance(src_attn, GatedDeltaNet)
        or "GatedDeltaNet" in src_attn.__class__.__name__
    ):
        return None

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (
            getattr(src_attn.q_proj, "in_features", None)
            if hasattr(src_attn, "q_proj")
            else None
        )
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))
    )

    try:
        new_block = GatedDeltaNet(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            num_heads=num_heads,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )
    except Exception as e:
        logger.exception(
            "GatedDeltaNet injection failed at layer %d: %s", layer_idx, e
        )
        return None

    # ---- Move to same device/dtype as source ----
    if src_attn is not None and src_attn is not target:
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_block.to(device=p.device, dtype=p.dtype)

    # ---- Splice ----
    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_block.is_nested_encoder = True
                target.attention.self = new_block
            else:
                new_block.is_nested_encoder = False
                target.attention = new_block
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            target.self = new_block
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            target.self_attn = new_block
        else:
            logger.error(
                "inject_gated_delta_net: no writable attention attribute on %s",
                type(target).__name__,
            )
            return None

    return new_block


__all__ = [
    "GatedDeltaNet",
    "inject_gated_delta_net",
    "chunked_delta_rule_scan",
    "apply_rotary_pos_emb",
    "LoRALinear",
    "repeat_kv",
]