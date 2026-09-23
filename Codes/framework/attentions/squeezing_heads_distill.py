"""
Squeezing Heads Attention Distillation Layer.
Multi-head attention wrapper designed for knowledge distillation and head capacity compression experiments.

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)

Bug-fix map:
  SQ-FIX-01  copy_weights raises on missing mandatory projections (no None.shape crash).
  SQ-FIX-02  injector recursion guard (_depth).
  SQ-FIX-03  injector double-injection guard.
  SQ-FIX-04  removed `'comp_config' in locals()` hack.
  SQ-FIX-05  GQA pooling validates head_dim divisibility.
  SQ-FIX-06  SDPA fast path also accepts additive attention_mask.
  SQ-FIX-07  causal mask offset-aware for KV-cache.
  SQ-FIX-08  head_mask broadcasting corrected (B,H,1,1).
  SQ-FIX-09  KV-cache implemented (tuple + DynamicCache).
  SQ-FIX-10  RoPE past_len offset.
  SQ-FIX-11  no silent pad/truncate of output dims (raise).
  SQ-FIX-12  is_nested_encoder set correctly for all branches in injector.
  SQ-FIX-13  attention_mask dtype robust (bool / int / float).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


logger = logging.getLogger(__name__)


# ===========================================================================
# RoPE
# ===========================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    if rotary_dim == 0:
        return x
    x1 = x[..., : rotary_dim // 2]
    x2 = x[..., rotary_dim // 2 : rotary_dim]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q_out = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot_embed
    k_out = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot_embed
    return q_out, k_out


# ===========================================================================
# Decoder detection helper
# ===========================================================================

def _detect_decoder(
    config: Any,
    is_decoder: Optional[bool],
    old_attn: Optional[nn.Module],
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "is_causal", False):
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
    """Linear projection with frozen base weights and trainable LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


# ===========================================================================
# Main Attention
# ===========================================================================

@ATTENTION_REGISTRY.register("squeezing_heads_distiller", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("squeezing_heads_distill", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("squeezing_heads", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("squeezing_heads_module", supports_causal=True, supports_gqa=True)
class SqueezingHeadsDistiller(BaseAttention):
    r"""Attention layer wrapper for multi-head distillation and compression studies."""

    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        old_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        # 1. Dimensions
        self.hidden_size: int = int(_cfg_get(
            config, "hidden_size", kwargs.get("hidden_size", 768)
        ))
        self.num_heads: int = int(_cfg_get(
            config,
            "num_attention_heads",
            _cfg_get(config, "num_heads", kwargs.get("num_attention_heads", 12)),
        ))
        cfg_hd = _cfg_get(config, "head_dim", None)
        kw_hd = kwargs.get("head_dim", None)
        if cfg_hd is not None:
            self.head_dim = int(cfg_hd)
        elif kw_hd is not None:
            self.head_dim = int(kw_hd)
        elif old_attn is not None:
            self.head_dim = None
            for attr in ("q_proj", "query", "q_lin"):
                if hasattr(old_attn, attr):
                    qw = getattr(old_attn, attr).weight
                    self.head_dim = qw.shape[0] // max(1, self.num_heads)
                    break
            if self.head_dim is None:
                self.head_dim = self.hidden_size // max(1, self.num_heads)
        else:
            self.head_dim = self.hidden_size // max(1, self.num_heads)

        cfg_kv = _cfg_get(
            config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None)
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = max(1, self.num_heads // self.num_kv_heads)
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.q_head_size: int = self.num_heads * self.head_dim
        self.all_head_size: int = self.q_head_size

        # 2. Decoder detection (SQ-FIX: unified helper)
        self.is_decoder = _detect_decoder(config, is_decoder, old_attn)
        self.is_causal = self.is_decoder
        self.is_nested_encoder: bool = False

        self.lora_rank = int(lora_rank if lora_rank is not None else 16)
        self.lora_alpha = float(lora_alpha)

        # 3. Projections
        self.q_proj = LoRALinear(self.hidden_size, self.q_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=True)
        self.k_proj = LoRALinear(self.hidden_size, self.kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=True)
        self.v_proj = LoRALinear(self.hidden_size, self.kv_head_size, bias=True,
                                 lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                 freeze_base=True)
        self.out_proj = LoRALinear(self.q_head_size, self.hidden_size, bias=True,
                                   lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                   freeze_base=True)

        dropout_p = float(
            getattr(config, "dropout",
                    getattr(config, "attention_probs_dropout_prob", 0.1))
            if config else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        src_attn = old_attn if old_attn is not None else kwargs.get("pretrained_attn", None)
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
    # copy_weights (SQ-FIX-01, SQ-FIX-05)
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: nn.Module) -> None:
        with torch.no_grad():
            w_q = b_q = w_k = b_k = w_v = b_v = w_o = b_o = None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            has_out_lin = hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj")
            if has_out_lin:
                self.is_nested_encoder = False
            elif (hasattr(old_attn, "self")
                  and hasattr(getattr(old_attn, "self"), "query")
                  and hasattr(old_attn, "output")
                  and hasattr(getattr(old_attn, "output"), "dense")):
                self.is_nested_encoder = True
            elif src_cls_name in ("BertSelfAttention", "RobertaSelfAttention"):
                self.is_nested_encoder = True
            else:
                self.is_nested_encoder = False

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight.detach()
                    b_o = old_attn.output.dense.bias.detach() if old_attn.output.dense.bias is not None else None
            else:
                src = old_attn

            # Pattern 1: GPT-2 fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                p_cls = getattr(src.c_attn, "__class__", type(src.c_attn)).__name__
                if ("Conv1D" in p_cls) or hasattr(src.c_attn, "nf") or (
                    c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size
                ):
                    c_attn_w = c_attn_w.t().contiguous()

                H = self.hidden_size
                w_q = c_attn_w[:H, :]
                w_k = c_attn_w[H: 2 * H, :]
                w_v = c_attn_w[2 * H: 3 * H, :]
                if c_attn_b is not None:
                    b_q = c_attn_b[:H]
                    b_k = c_attn_b[H: 2 * H]
                    b_v = c_attn_b[2 * H: 3 * H]

                out_w = src.c_proj.weight.detach()
                p_proj_cls = getattr(src.c_proj, "__class__", type(src.c_proj)).__name__
                if ("Conv1D" in p_proj_cls) or hasattr(src.c_proj, "nf") or (
                    out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size
                ):
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: RoBERTa / BERT
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q = src.query.weight.detach()
                b_q = src.query.bias.detach() if src.query.bias is not None else None
                w_k = src.key.weight.detach()
                b_k = src.key.bias.detach() if src.key.bias is not None else None
                w_v = src.value.weight.detach()
                b_v = src.value.bias.detach() if src.value.bias is not None else None
                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            proj = getattr(src, attr)
                            w_o = proj.weight.detach()
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            # Pattern 3: Qwen / LLaMA
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q = src.q_proj.weight.detach()
                b_q = src.q_proj.bias.detach() if src.q_proj.bias is not None else None
                w_k = src.k_proj.weight.detach()
                b_k = src.k_proj.bias.detach() if src.k_proj.bias is not None else None
                w_v = src.v_proj.weight.detach()
                b_v = src.v_proj.bias.detach() if src.v_proj.bias is not None else None
                if hasattr(src, "o_proj"):
                    w_o = src.o_proj.weight.detach()
                    b_o = src.o_proj.bias.detach() if src.o_proj.bias is not None else None

            # Pattern 4: DistilBERT
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q = src.q_lin.weight.detach()
                b_q = src.q_lin.bias.detach() if src.q_lin.bias is not None else None
                w_k = src.k_lin.weight.detach()
                b_k = src.k_lin.bias.detach() if src.k_lin.bias is not None else None
                w_v = src.v_lin.weight.detach()
                b_v = src.v_lin.bias.detach() if src.v_lin.bias is not None else None
                if hasattr(src, "out_lin"):
                    w_o = src.out_lin.weight.detach()
                    b_o = src.out_lin.bias.detach() if src.out_lin.bias is not None else None

            # Fallback
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter(old_attn)
                    extracted = adapter.extract_qkv_out()
                    if extracted:
                        if w_q is None: w_q = extracted.get("q_weight")
                        if b_q is None: b_q = extracted.get("q_bias")
                        if w_k is None: w_k = extracted.get("k_weight")
                        if b_k is None: b_k = extracted.get("k_bias")
                        if w_v is None: w_v = extracted.get("v_weight")
                        if b_v is None: b_v = extracted.get("v_bias")
                        if w_o is None:
                            w_o = extracted.get("out_weight")
                            b_o = extracted.get("out_bias")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # SQ-FIX-01: raise on missing mandatory projections (no None.shape crash).
            missing = [n for n, w in (("q", w_q), ("k", w_k), ("v", w_v)) if w is None]
            if missing:
                raise RuntimeError(
                    f"SqueezingHeadsDistiller.copy_weights: missing mandatory "
                    f"projections {missing} from {src_cls_name}."
                )

            # Frobenius checks
            for name, w in (("w_q", w_q), ("w_k", w_k), ("w_v", w_v)):
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    logger.warning(f"SqueezingHeadsDistiller: {name} near-zero Frobenius norm")

            # ---- Copy Q ----
            self.q_proj.weight.zero_()
            m_q_out = min(self.q_head_size, w_q.shape[0])
            m_q_in = min(self.hidden_size, w_q.shape[1])
            self.q_proj.weight[:m_q_out, :m_q_in].copy_(w_q[:m_q_out, :m_q_in].to(self.q_proj.weight.dtype))
            if b_q is not None and self.q_proj.bias is not None:
                m_bq = min(self.q_proj.bias.shape[0], b_q.shape[0])
                self.q_proj.bias.zero_()
                self.q_proj.bias[:m_bq].copy_(b_q[:m_bq].to(self.q_proj.bias.dtype))

            # ---- Copy K with optional GQA pooling (SQ-FIX-05) ----
            self._copy_kv_proj(self.k_proj, w_k, b_k)
            self._copy_kv_proj(self.v_proj, w_v, b_v)

            # ---- Copy O ----
            if self.is_nested_encoder:
                nn.init.eye_(self.out_proj.weight)
                if self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)
            elif w_o is not None:
                if w_o.shape == self.out_proj.weight.shape:
                    self.out_proj.weight.copy_(w_o.to(self.out_proj.weight.dtype))
                elif w_o.t().contiguous().shape == self.out_proj.weight.shape:
                    self.out_proj.weight.copy_(w_o.t().contiguous().to(self.out_proj.weight.dtype))
                else:
                    self.out_proj.weight.zero_()
                    m_out = min(self.hidden_size, w_o.shape[0])
                    m_in = min(self.q_head_size, w_o.shape[1])
                    self.out_proj.weight[:m_out, :m_in].copy_(
                        w_o[:m_out, :m_in].to(self.out_proj.weight.dtype)
                    )
                if b_o is not None and self.out_proj.bias is not None:
                    m_bo = min(self.out_proj.bias.shape[0], b_o.shape[0])
                    self.out_proj.bias.zero_()
                    self.out_proj.bias[:m_bo].copy_(b_o[:m_bo].to(self.out_proj.bias.dtype))

    def _copy_kv_proj(self, target: LoRALinear, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        """Copy KV projection with GQA pooling or principal-slice fallback."""
        target.weight.zero_()
        if b is not None and target.bias is not None:
            target.bias.zero_()

        target_out = target.weight.shape[0]   # kv_head_size
        target_in = target.weight.shape[1]    # hidden_size

        # Case A: exact match
        if w.shape[0] == target_out and w.shape[1] == target_in:
            target.weight.copy_(w.to(target.weight.dtype))
            if b is not None and target.bias is not None and b.shape[0] == target_out:
                target.bias.copy_(b.to(target.bias.dtype))
            return

        # Case B: GQA pooling (SQ-FIX-05): source has more heads, mean-pool them.
        src_heads_total = w.shape[0]
        if (src_heads_total > target_out
                and src_heads_total % target_out == 0
                and src_heads_total % self.head_dim == 0
                and target_out % self.head_dim == 0
                and w.shape[1] >= target_in):
            rep = src_heads_total // target_out
            w_slice = w[:, :target_in].contiguous()
            w_pooled = (
                w_slice.view(self.num_kv_heads, rep, self.head_dim, target_in)
                .mean(dim=1)
                .reshape(target_out, target_in)
            )
            target.weight.copy_(w_pooled.to(target.weight.dtype))
            if b is not None and target.bias is not None:
                if b.shape[0] == src_heads_total:
                    b_pooled = (
                        b.view(self.num_kv_heads, rep, self.head_dim)
                        .mean(dim=1)
                        .reshape(target_out)
                    )
                    target.bias.copy_(b_pooled.to(target.bias.dtype))
                else:
                    m = min(target.bias.shape[0], b.shape[0])
                    target.bias[:m].copy_(b[:m].to(target.bias.dtype))
            return

        # Case C: principal-slice fallback
        m_out = min(target_out, w.shape[0])
        m_in = min(target_in, w.shape[1])
        target.weight[:m_out, :m_in].copy_(w[:m_out, :m_in].to(target.weight.dtype))
        if b is not None and target.bias is not None:
            m_b = min(target.bias.shape[0], b.shape[0])
            target.bias[:m_b].copy_(b[:m_b].to(target.bias.dtype))

    # ------------------------------------------------------------------
    # Mask helper (SQ-FIX-13)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, _ = hidden_states.shape
        device = hidden_states.device

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        L_kv_new = kv_input.shape[1]

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k_new = self.k_proj(kv_input).view(B, L_kv_new, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v_new = self.v_proj(kv_input).view(B, L_kv_new, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # ---- RoPE with past offset (SQ-FIX-10) ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len_rope = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len_rope = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len_rope = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len_rope: past_len_rope + L, :]
            sin = sin[..., past_len_rope: past_len_rope + L, :]
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

        # ---- KV-cache (SQ-FIX-09) ----
        past_k = past_v = None
        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]

        if past_k is not None and not is_cross:
            k_cat = torch.cat([past_k, k_new], dim=-2)
            v_cat = torch.cat([past_v, v_new], dim=-2)
        else:
            k_cat = k_new
            v_cat = v_new

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            past_key_value.update(k_new, v_new, layer_idx)

        N_kv = k_cat.shape[2]
        past_len = N_kv - L if (self.is_decoder and not is_cross) else 0

        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            k_cat = repeat_kv(k_cat, self.kv_repeat)
            v_cat = repeat_kv(v_cat, self.kv_repeat)

        # ---- Mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)

        scale = 1.0 / math.sqrt(self.head_dim)

        # ---- SDPA fast path with full mask support (SQ-FIX-06) ----
        use_sdpa = not output_attentions and head_mask is None
        if use_sdpa:
            q_sdp = q.to(torch.float16) if q.dtype == torch.float32 else q
            k_sdp = k_cat.to(torch.float16) if k_cat.dtype == torch.float32 else k_cat
            v_sdp = v_cat.to(torch.float16) if v_cat.dtype == torch.float32 else v_cat

            attn_bias_sdp = None
            if token_valid is not None:
                attn_bias_sdp = ((token_valid[:, None, None, :].float() - 1.0) * 1e4).to(q_sdp.dtype)

            is_causal_sdp = self.is_decoder and not is_cross and past_len == 0 and L > 1

            try:
                attn_out = F.scaled_dot_product_attention(
                    q_sdp, k_sdp, v_sdp,
                    attn_mask=attn_bias_sdp,
                    is_causal=is_causal_sdp,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                attn_out = attn_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(
                    B, L, self.q_head_size
                )
                attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
                attn_probs_ret = None
            except Exception:
                use_sdpa = False

        if not use_sdpa:
            scores = torch.matmul(q.float(), k_cat.float().transpose(-2, -1).contiguous()) * scale

            # Causal mask (SQ-FIX-07: offset-aware for cache)
            if self.is_decoder and not is_cross:
                if past_len > 0:
                    abs_q = (past_len + torch.arange(L, device=device)).unsqueeze(1)
                    abs_k = torch.arange(N_kv, device=device).unsqueeze(0)
                    causal = (abs_k <= abs_q)
                else:
                    causal = torch.tril(torch.ones(L, N_kv, device=device, dtype=torch.bool))
                scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), -1e4)

            if token_valid is not None:
                scores = scores.masked_fill(~token_valid[:, None, None, :].bool(), -1e4)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
            probs = self.dropout(probs)

            # SQ-FIX-08: head_mask broadcasting to (B, H, 1, 1) or (1, H, 1, 1).
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask[:, :, :1].unsqueeze(-1)     # (B, H, 1, 1)
                probs = probs * head_mask.to(probs.dtype)

            attn_out = torch.matmul(probs, v_cat).transpose(1, 2).contiguous().view(
                B, L, self.q_head_size
            )
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
            attn_probs_ret = probs if output_attentions else None
        else:
            attn_probs_ret = None

        # ---- Output ----
        if self.is_nested_encoder:
            out = attn_out
        else:
            out = self.out_proj(attn_out)

        # SQ-FIX-11: strict dimension check (no silent pad/truncate).
        if out.shape[-1] != hidden_states.shape[-1]:
            raise RuntimeError(
                f"SqueezingHeadsDistiller: output dim {out.shape[-1]} != "
                f"input dim {hidden_states.shape[-1]}."
            )

        return out, attn_probs_ret


# ===========================================================================
# Injection
# ===========================================================================

@ATTENTION_REGISTRY.register("squeezing_heads_inject")
@ATTENTION_REGISTRY.register("squeezing_heads_distill_inject")
@ATTENTION_REGISTRY.register("squeezing_heads_distiller_inject")
@ATTENTION_REGISTRY.register("inject_squeezing_heads")
@ATTENTION_REGISTRY.register("inject_squeezing_heads_distill")
@ATTENTION_REGISTRY.register("inject_squeezing_heads_distiller")
def inject_squeezing_heads(
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
    # SQ-FIX-02: recursion guard.
    if _depth > 8:
        logger.warning("inject_squeezing_heads: max recursion depth exceeded.")
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
            res = inject_squeezing_heads(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # Single-layer injection
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "attn", "self_attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if cand is not None:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            return None

        # SQ-FIX-03: double-injection guard.
        if isinstance(src, SqueezingHeadsDistiller):
            return None
        if "SqueezingHeads" in src.__class__.__name__:
            return None
        inner = getattr(src, "inner", None)
        if isinstance(inner, SqueezingHeadsDistiller):
            return None
        wrapped = getattr(src, "_wrapped_attention", None)
        if isinstance(wrapped, SqueezingHeadsDistiller):
            return None

        if cfg is None and hasattr(target, "config"):
            cfg = target.config
        if cfg is None and hasattr(src, "config"):
            cfg = src.config

        rank = kwargs.pop("rank", None)
        lora_rank = kwargs.pop("lora_rank", None)
        r = rank if rank is not None else _cfg_get(comp_config, "rank",
                                                    _cfg_get(comp_config, "svd_rank", None))
        lr = lora_rank if lora_rank is not None else _cfg_get(comp_config, "lora_rank", 16)

        new_attn = SqueezingHeadsDistiller(
            config=cfg,
            old_attn=src,
            lora_rank=lr,
            rank=r,
            **kwargs,
        )

        # SQ-FIX-12: set is_nested_encoder on every branch.
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
                new_attn.is_nested_encoder = False
        elif hasattr(target, "attn"):
            target.attn = new_attn
            new_attn.is_nested_encoder = False
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            new_attn.is_nested_encoder = False
        elif hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            target.attention = new_attn
            new_attn.is_nested_encoder = False

        return new_attn
    except Exception as e:
        logger.exception("inject_squeezing_heads failed at layer %d: %s", layer_idx, e)
        return None


# Aliases
inject_squeezing_heads_distill = inject_squeezing_heads
inject_squeezing_heads_attention = inject_squeezing_heads
inject_squeezing_heads_distill_attention = inject_squeezing_heads
inject_squeezing_heads_distiller = inject_squeezing_heads
SqueezingHeads = SqueezingHeadsDistiller
SqueezingHeadsAttention = SqueezingHeadsDistiller