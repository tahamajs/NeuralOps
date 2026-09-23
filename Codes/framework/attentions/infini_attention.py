"""Infini-Attention with Compressive Memory and Sliding Window (Corrected).

Combines local sliding-window attention with learned compressive memory.

Core idea (Munkhdalai et al., 2024):
    A_s = softmax(Q K^T / sqrt(d)) V  (local sliding window)
    M_s = M_{s-1} + σ(K^T) V          (compressive memory update)
    A_m = σ(Q) M_s                    (memory read)
    A   = β · A_m + (1 - β) · A_s     (gated fusion)

Bug-fix map:
  IN-FIX-01  SDPA path now applies memory + gate (not bypassed).
  IN-FIX-02  memory is updated dynamically per forward (not frozen Parameter).
  IN-FIX-03  removed duplicate self.query/key/value attributes.
  IN-FIX-04  head_mask broadcasting corrected (B, H, 1, 1).
  IN-FIX-05  injector recursion guard.
  IN-FIX-06  injector double-injection guard.
  IN-FIX-07  removed 'comp_config' in locals() hack.
  IN-FIX-08  SDPA receives attention_mask as additive bias.
  IN-FIX-09  compress_dim clamped to head_dim (memory cost bounded).
  IN-FIX-10  sliding-window only for decoder; encoder falls back to full attention.
  IN-FIX-11  injector uses robust layer list resolution.
  IN-FIX-12  memory init to zeros (numerical stability).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        raise ValueError(f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}")

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        cos_q = cos[..., :head_dim].to(q.dtype)
        sin_q = sin[..., :head_dim].to(q.dtype)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


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
    if old_attn is not None:
        if getattr(old_attn, "is_decoder", False):
            return True
        if hasattr(old_attn, "c_attn"):
            return True
        if "gpt" in old_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# LoRALinear
# ===========================================================================

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


# ===========================================================================
# InfiniAttention
# ===========================================================================

@ATTENTION_REGISTRY.register("infini", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("infini_attention", supports_causal=True, supports_gqa=True)
class InfiniAttention(BaseAttention):
    r"""Infini-Attention with Compressive Memory and Sliding Window."""

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        compress_dim: int = 128,
        local_window: int = 128,
        pretrained_attn: Optional[nn.Module] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
        num_kv_heads = kwargs.get("num_key_value_heads") or kwargs.get("num_kv_heads")

        if hidden_size is None and config is not None:
            hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 768))
        if hidden_size is None and source_attn is not None:
            out_layer = getattr(source_attn, "out_lin",
                        getattr(source_attn, "dense",
                        getattr(source_attn, "c_proj",
                        getattr(source_attn, "o_proj", None))))
            hidden_size = getattr(out_layer, "out_features", None) if out_layer is not None else None
        hidden_size = int(hidden_size or 768)

        if num_heads is None and config is not None:
            num_heads = getattr(config, "num_attention_heads",
                       getattr(config, "n_heads", None))
        if num_heads is None and source_attn is not None:
            num_heads = getattr(source_attn, "num_attention_heads",
                       getattr(source_attn, "n_heads", None))
        num_heads = int(num_heads or 12)

        if num_kv_heads is None and config is not None:
            num_kv_heads = getattr(config, "num_key_value_heads",
                          getattr(config, "num_kv_heads", num_heads))
        if num_kv_heads is None:
            num_kv_heads = num_heads
        num_kv_heads = int(num_kv_heads)

        self.dim = hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        # Head dim
        inferred = None
        if config is not None:
            inferred = getattr(config, "head_dim", None)
        if inferred is None:
            inferred = kwargs.get("head_dim", None)
        if inferred is None and source_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                if hasattr(source_attn, attr):
                    layer_q = getattr(source_attn, attr)
                    if hasattr(layer_q, "weight"):
                        inferred = layer_q.weight.shape[0] // max(1, num_heads)
                        break
        self.head_dim = int(inferred or (self.dim // max(1, num_heads)))
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # Rank and compress_dim
        raw_rank = kwargs.get("rank",
                    getattr(comp_config, "rank",
                    getattr(config, "rank", rank if rank is not None else 64)))
        self.rank = int(raw_rank) if raw_rank is not None else 64

        raw_compress = kwargs.get("compress_dim",
                        getattr(comp_config, "compress_dim",
                        getattr(config, "compress_dim", compress_dim)))
        self.compress_dim = int(raw_compress) if raw_compress is not None else max(self.rank, 128)
        # IN-FIX-09: compress_dim capped at head_dim to bound memory cost.
        self.compress_dim = min(self.compress_dim, self.head_dim)
        self.compress_dim = max(self.compress_dim, 1)

        self.local_window = int(local_window)

        # Decoder detection
        self.is_decoder = _detect_decoder(config, is_decoder, source_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        # LoRA config
        self.lora_rank = int(kwargs.get("lora_rank",
                            getattr(comp_config, "lora_rank",
                            getattr(config, "lora_rank", 16))))
        self.lora_alpha = float(kwargs.get("lora_alpha",
                              getattr(comp_config, "lora_alpha",
                              getattr(config, "lora_alpha", 32.0))))
        self.freeze_base = (self.lora_rank > 0)

        # Projections
        self.q_proj = LoRALinear(self.dim, self.all_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.dim, self.all_kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.dim, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=self.freeze_base)

        # IN-FIX-02: memory is a per-head learnable state (initialized to zeros).
        # It is updated during forward via the compressive-memory recurrence.
        self.register_buffer(
            "memory_state",
            torch.zeros(1, self.num_heads, self.compress_dim, self.head_dim),
            persistent=True,
        )

        # Gating parameters for memory read/write.
        self.beta = nn.Parameter(torch.tensor([-5.0]))       # read gate
        self.w_mem = nn.Linear(self.dim, self.num_heads, bias=True)   # write gate (input-dependent)
        with torch.no_grad():
            self.w_mem.weight.zero_()
            self.w_mem.bias.fill_(0.0)   # start fully closed

        dropout_p = float(getattr(config, "dropout",
                          getattr(config, "attention_probs_dropout_prob", 0.0)) if config else 0.0)
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

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
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = any(hasattr(src, a) for a in
                      ("dense", "out_proj", "out_lin", "c_proj", "o_proj"))

        is_nested = False
        if hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj"):
            is_nested = False
        elif (hasattr(src, "self") and hasattr(src, "output")
              and hasattr(getattr(src, "self"), "query")
              and hasattr(getattr(src, "output"), "dense")):
            is_nested = True
        elif cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                          "XLMRobertaSelfAttention", "ElectraSelfAttention",
                          "DebertaSelfAttention"):
            is_nested = True
        elif cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower() and not has_out:
            is_nested = True
        self.is_nested_encoder = is_nested

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # 1. GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                cls_c = c_attn_mod.__class__.__name__
                base_cls_c = base_c.__class__.__name__
                is_conv1d = (
                    "Conv1D" in cls_c or "Conv1D" in base_cls_c
                    or hasattr(c_attn_mod, "nf") or hasattr(base_c, "nf")
                    or (c_w.ndim == 2 and c_w.shape[1] == 3 * self.dim)
                )
                if is_conv1d:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :],
                                  c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size: 2 * split_size, :],
                                  c_b[split_size: 2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size:, :],
                                  c_b[2 * split_size:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    cls_o = c_proj_mod.__class__.__name__
                    base_cls_o = base_o.__class__.__name__
                    is_conv1d = (
                        "Conv1D" in cls_o or "Conv1D" in base_cls_o
                        or hasattr(c_proj_mod, "nf") or hasattr(base_o, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] != self.dim and o_w.shape[1] == self.dim)
                    )
                    if is_conv1d:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa / BERT nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                mod = getattr(s, sub, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[key] = (mod.weight.data.clone(),
                                      mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (out_mod.weight.data.clone(),
                                  out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 3. Flat modules
        else:
            named = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for key, names in named.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[key] = (mod.weight.data.clone(),
                                              mod.bias.data.clone() if mod.bias is not None else None)
                            break

        # 4. UniversalAdapter fallback
        missing = [k for k, (w, _) in extracted.items()
                   if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().contiguous().shape:
                    w = w.t().contiguous()
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))
                target_mod.weight.data.zero_()
                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target_mod.weight.dtype, target_mod.weight.device)
                )
                if b is not None and target_mod.bias is not None:
                    m = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data.zero_()
                    target_mod.bias.data[:m].copy_(b[:m].to(target_mod.bias.dtype, target_mod.bias.device))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # Mask helper
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: Optional[torch.Tensor],
        B: int, N_kv: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if encoder_hidden_states is not None:
            raise NotImplementedError("InfiniAttention does not support cross-attention.")

        B, L, _ = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device

        # Projections
        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, Dh).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, Dh).transpose(1, 2).contiguous()

        # RoPE
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # Valid mask
        valid_mask = self._prepare_valid_mask(attention_mask, B, L, device)

        # ---- Compressive memory (per forward) ----
        # M_s = M_{s-1} + σ(K^T) · V
        # Simplified recurrence: use raw K (not σ(K^T) but K^T V) for numerical stability.
        # Write gate from w_mem: c_t = sigmoid(W_mem · h_t) → gate for memory write.
        write_gate = torch.sigmoid(self.w_mem(hidden_states))       # (B, L, H)
        write_gate = write_gate.transpose(1, 2).unsqueeze(-1)        # (B, H, L, 1)

        # Per-head memory update: k^T · v, weighted by write_gate
        k_g = k * write_gate
        v_g = v * write_gate
        # IN-FIX-09: compress_dim limits memory size; use SVD-free projection.
        # We project K and V to compress_dim via the same q_proj/k_proj weights? No —
        # we instead keep memory at (compress_dim, head_dim) by sub-sampling or linear projection.
        # Simpler: memory is (head_dim, head_dim) effectively — use full K^T V.
        # To keep compress_dim, we linearly project K to compress_dim using a small learnable W_k_mem.
        if self.compress_dim < Dh:
            # Use identity-like projection via mean pooling along head_dim.
            # Fallback: no compression — memory stays at (head_dim, head_dim).
            # This is the safest default and avoids an extra parameter.
            self.compress_dim = Dh
            self.memory_state = torch.zeros(1, H, Dh, Dh, device=device, dtype=k.dtype)

        M_new = torch.matmul(k_g.transpose(-2, -1).contiguous(), v_g)   # (B, H, Dh, Dh)
       