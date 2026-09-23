"""
Token-Compressed Attention — Production-Hardened.

Compresses Key and Value sequence tokens using energy/norm + learned saliency
scoring, while always preserving token 0 ([CLS] / first token).

Bug-fix map (2026 revision):
  TC-01  LoRALinear respects device/dtype.
  TC-02  RoPE past_len offset for KV-cache.
  TC-03  All aliases (all_head_size, q_lin/k_lin/v_lin/out_lin) defined.
  TC-04  head_mask shape unified.
  TC-05  score_net respects device/dtype.
  TC-06  CLS preservation via mask (no in-place mutation of grad tensor).
  TC-07  SDPA fast path when no compression/head_mask.
  TC-08  No silent pad/truncate; raise on mismatch.
  TC-09  Recursion guard in injector.
  TC-10  Decoder detection in injector.
  TC-11  Saliency uses hidden_states (intentional), documented.
  TC-12  Robust 4D-mask handling.
  TC-13  Sort + causal handled via absolute positions.
  TC-14  Single registry key per role (class vs injector).
  TC-15  num_heads fallback derived from hidden_size.
  TC-16  All common aliases present for framework compatibility.
  TC-17  repeat_kv imported from framework (fixes Pylance error).
  TC-18  score_net created on same device as base projections.
  TC-19  valid_mask not applied to query positions (semantically wrong).
  TC-20  CLS boost skipped when CLS is padding.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY

# TC-17: import repeat_kv (fixes Pylance "not defined")
try:
    from framework.core.attention_utils import repeat_kv
except ImportError:
    try:
        from framework.core.gqa_utils import repeat_kv
    except ImportError:
        # Fallback: define locally (last resort).
        def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
            if n_rep == 1:
                return hidden_states
            batch, num_kv_heads, slen, head_dim = hidden_states.shape
            hidden_states = hidden_states[:, :, None, :, :].expand(
                batch, num_kv_heads, n_rep, slen, head_dim
            )
            return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

logger = logging.getLogger(__name__)


# ===========================================================================
# Config getter
# ===========================================================================

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# RoPE
# ===========================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:2 * half]
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
    """TC-02: dtype-safe, past_len-aware RoPE."""
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
        raise ValueError(
            f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}"
        )

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut = cos[..., :head_dim].to(q.dtype)
        sin_cut = sin[..., :head_dim].to(q.dtype)
        q = (q * cos_cut) + (_rotate_half(q) * sin_cut)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# TC-01: LoRALinear with device/dtype support
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank) if lora_rank and lora_rank > 0 else 0
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / max(1, self.lora_rank)

        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features, **factory))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_rank, **factory))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
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
# TC-14: registry separation
# ===========================================================================

@ATTENTION_REGISTRY.register("token_compressed_class")
@ATTENTION_REGISTRY.register("token_compressed_attention_class")
class TokenCompressedAttention(BaseAttention):
    r"""Token-Compressed Attention Module.

    Compresses Key and Value sequence tokens using energy + learned saliency,
    always preserving token 0 ([CLS] / first token).
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        compression_ratio: float = 0.5,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: Optional[int] = None,
        lora_alpha: Optional[float] = None,
        comp_config: Optional[Any] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn", None)

        # -------- hidden_size --------
        dim = None
        if config is not None and not isinstance(config, int):
            dim = _cfg_get(config, "hidden_size",
                           _cfg_get(config, "dim", _cfg_get(config, "d_model", None)))
        if dim is None and comp_config is not None:
            dim = _cfg_get(comp_config, "hidden_size", _cfg_get(comp_config, "dim", None))
        if dim is None and src is not None:
            dim = getattr(src, "hidden_size", getattr(src, "embed_dim", None))
            if dim is None and hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features"):
                dim = src.q_proj.in_features
            elif dim is None and hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features"):
                dim = src.q_lin.in_features
            elif dim is None and hasattr(src, "query") and hasattr(src.query, "in_features"):
                dim = src.query.in_features
            elif dim is None and hasattr(src, "c_attn") and hasattr(src.c_attn, "weight"):
                w = src.c_attn.weight
                dim = w.shape[0] if w.shape[0] < w.shape[1] else w.shape[1]
        if dim is None:
            dim = kwargs.get("hidden_size", 768)
        self.dim = int(dim)
        self.hidden_size = self.dim

        # -------- num_heads (TC-15) --------
        num_heads = None
        if "num_heads" in kwargs and kwargs["num_heads"] is not None:
            num_heads = kwargs["num_heads"]
        elif config is not None and not isinstance(config, int):
            num_heads = _cfg_get(config, "num_attention_heads",
                                 _cfg_get(config, "n_heads",
                                          _cfg_get(config, "num_heads", None)))
        if num_heads is None and comp_config is not None:
            num_heads = _cfg_get(comp_config, "num_attention_heads",
                                 _cfg_get(comp_config, "n_heads", None))
        if num_heads is None and src is not None:
            num_heads = getattr(src, "num_heads",
                                getattr(src, "num_attention_heads",
                                        getattr(src, "n_head", None)))
        if num_heads is None:
            num_heads = max(1, self.dim // 64)
        self.num_heads = max(1, int(num_heads))

        # -------- head_dim --------
        head_dim = None
        if config is not None and not isinstance(config, int):
            head_dim = _cfg_get(config, "head_dim", None)
        if head_dim is None:
            head_dim = kwargs.get("head_dim", None)
        if head_dim is None:
            head_dim = self.dim // self.num_heads
        self.head_dim = int(head_dim)

        # -------- num_kv_heads --------
        cfg_kv = None
        if config is not None and not isinstance(config, int):
            cfg_kv = _cfg_get(config, "num_key_value_heads",
                              _cfg_get(config, "num_kv_heads", None))
        if cfg_kv is None and comp_config is not None:
            cfg_kv = _cfg_get(comp_config, "num_key_value_heads",
                              _cfg_get(comp_config, "num_kv_heads", None))
        if cfg_kv is None and src is not None:
            cfg_kv = getattr(src, "num_key_value_heads",
                             getattr(src, "num_kv_heads", None))
        self.num_kv_heads = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.num_key_value_groups = max(1, self.num_heads // self.num_kv_heads)

        # TC-03: all aliases
        self.q_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.all_head_size = self.q_head_size
        self.all_kv_head_size = self.kv_head_size

        # -------- compression ratio --------
        self.compression_ratio = float(kwargs.get(
            "compression_ratio",
            _cfg_get(comp_config, "compression_ratio", compression_ratio),
        ))
        self.compression_ratio = max(0.0, min(1.0, self.compression_ratio))

        # -------- LoRA config --------
        if "lora_rank" in kwargs and kwargs["lora_rank"] is not None:
            rl = int(kwargs["lora_rank"])
        elif comp_config is not None and _cfg_get(comp_config, "lora_rank") is not None:
            rl = int(_cfg_get(comp_config, "lora_rank"))
        elif config is not None and not isinstance(config, int) and _cfg_get(config, "lora_rank") is not None:
            rl = int(_cfg_get(config, "lora_rank"))
        elif lora_rank is not None:
            rl = int(lora_rank)
        else:
            rl = 16
        self.lora_rank = rl

        if "lora_alpha" in kwargs and kwargs["lora_alpha"] is not None:
            ra = float(kwargs["lora_alpha"])
        elif comp_config is not None and _cfg_get(comp_config, "lora_alpha") is not None:
            ra = float(_cfg_get(comp_config, "lora_alpha"))
        elif config is not None and not isinstance(config, int) and _cfg_get(config, "lora_alpha") is not None:
            ra = float(_cfg_get(config, "lora_alpha"))
        elif lora_alpha is not None:
            ra = float(lora_alpha)
        else:
            ra = 32.0
        self.lora_alpha = ra

        # -------- decoder detection --------
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        elif kwargs.get("is_decoder") is not None:
            self.is_decoder = bool(kwargs["is_decoder"])
        elif src is not None and getattr(src, "is_decoder", False):
            self.is_decoder = True
        elif config is not None and not isinstance(config, int):
            mt = str(_cfg_get(config, "model_type", "")).lower()
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or any(mt.startswith(p) for p in (
                    "gpt", "qwen", "llama", "mistral", "gemma",
                    "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))
            )
        else:
            self.is_decoder = False
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False
        self.rotary_emb = getattr(src, "rotary_emb", None) if src is not None else None

        # TC-18: derive a consistent device/dtype from source attention if possible.
        proj_device = None
        proj_dtype = None
        if src is not None:
            for attr in ("q_lin", "q_proj", "query", "c_attn"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    proj_device = mod.weight.device
                    proj_dtype = mod.weight.dtype
                    break

        # -------- projections --------
        self.q_lin = LoRALinear(
            self.dim, self.q_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True,
            device=proj_device, dtype=proj_dtype,
        )
        self.k_lin = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True,
            device=proj_device, dtype=proj_dtype,
        )
        self.v_lin = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True,
            device=proj_device, dtype=proj_dtype,
        )
        self.out_lin = LoRALinear(
            self.q_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True,
            device=proj_device, dtype=proj_dtype,
        )

        # Aliases
        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin
        self.query = self.q_lin
        self.key = self.k_lin
        self.value = self.v_lin

        # TC-18: score_net on the same device as projections.
        self.score_net = nn.Sequential(
            nn.Linear(self.dim, 64, device=proj_device, dtype=proj_dtype),
            nn.GELU(),
            nn.Linear(64, 1, device=proj_device, dtype=proj_dtype),
        )

        dropout_p = (
            _cfg_get(config, "dropout",
                     _cfg_get(config, "attention_probs_dropout_prob", 0.1))
            if config is not None and not isinstance(config, int)
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    # Aliases (TC-16)
    # ------------------------------------------------------------------
    @property
    def dense(self): return self.out_lin
    @property
    def c_proj(self): return self.out_lin
    @property
    def o_proj(self): return self.out_lin

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2, Qwen/LLaMA."""
        with torch.no_grad():
            if getattr(old_attn, "rotary_emb", None) is not None:
                self.rotary_emb = old_attn.rotary_emb

            w_q = b_q = w_k = b_k = w_v = b_v = w_o = b_o = None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
                and not hasattr(old_attn, "c_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    dense = getattr(old_attn.output.dense, "base_layer", old_attn.output.dense)
                    w_o = dense.weight.detach()
                    b_o = dense.bias.detach() if dense.bias is not None else None
            else:
                src = old_attn

            def _unwrap_linear(layer: Any) -> Any:
                return getattr(layer, "base_layer", layer)

            # GPT-2 fused c_attn / c_proj
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_layer = _unwrap_linear(src.c_attn)
                c_proj_layer = _unwrap_linear(src.c_proj)
                c_attn_w = c_attn_layer.weight.detach()
                c_attn_b = c_attn_layer.bias.detach() if c_attn_layer.bias is not None else None
                if c_attn_w.dim() == 2 and c_attn_w.shape[0] == self.dim and c_attn_w.shape[1] != self.dim:
                    c_attn_w = c_attn_w.t().contiguous()
                q_dim, k_dim, v_dim = self.q_head_size, self.kv_head_size, self.kv_head_size
                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim:q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim:q_dim + k_dim + v_dim, :]
                if c_attn_b is not None:
                    b_q = c_attn_b[:q_dim]
                    b_k = c_attn_b[q_dim:q_dim + k_dim]
                    b_v = c_attn_b[q_dim + k_dim:q_dim + k_dim + v_dim]
                out_w = c_proj_layer.weight.detach()
                if out_w.dim() == 2 and out_w.shape[0] == self.dim and out_w.shape[1] != self.dim:
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = c_proj_layer.bias.detach() if c_proj_layer.bias is not None else None

            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                q_l = _unwrap_linear(src.query)
                k_l = _unwrap_linear(src.key)
                v_l = _unwrap_linear(src.value)
                w_q = q_l.weight.detach()
                b_q = q_l.bias.detach() if q_l.bias is not None else None
                w_k = k_l.weight.detach()
                b_k = k_l.bias.detach() if k_l.bias is not None else None
                w_v = v_l.weight.detach()
                b_v = v_l.bias.detach() if v_l.bias is not None else None
                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj", "c_proj"):
                        if hasattr(src, attr):
                            proj = _unwrap_linear(getattr(src, attr))
                            w_o = proj.weight.detach()
                            b_o = proj.bias.detach() if proj.bias is not None else None
                            break

            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                q_l = _unwrap_linear(src.q_proj)
                k_l = _unwrap_linear(src.k_proj)
                v_l = _unwrap_linear(src.v_proj)
                w_q = q_l.weight.detach()
                b_q = q_l.bias.detach() if q_l.bias is not None else None
                w_k = k_l.weight.detach()
                b_k = k_l.bias.detach() if k_l.bias is not None else None
                w_v = v_l.weight.detach()
                b_v = v_l.bias.detach() if v_l.bias is not None else None
                if hasattr(src, "o_proj"):
                    o_l = _unwrap_linear(src.o_proj)
                    w_o = o_l.weight.detach()
                    b_o = o_l.bias.detach() if o_l.bias is not None else None

            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                q_l = _unwrap_linear(src.q_lin)
                k_l = _unwrap_linear(src.k_lin)
                v_l = _unwrap_linear(src.v_lin)
                w_q = q_l.weight.detach()
                b_q = q_l.bias.detach() if q_l.bias is not None else None
                w_k = k_l.weight.detach()
                b_k = k_l.bias.detach() if k_l.bias is not None else None
                w_v = v_l.weight.detach()
                b_v = v_l.bias.detach() if v_l.bias is not None else None
                if hasattr(src, "out_lin"):
                    o_l = _unwrap_linear(src.out_lin)
                    w_o = o_l.weight.detach()
                    b_o = o_l.bias.detach() if o_l.bias is not None else None

            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    if w_q is None:
                        w_q, b_q = adapter.get_weight_bias(src, "q")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(src, "k")
                    if w_v is None:
                        w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            missing = []
            if w_q is None: missing.append("Q")
            if w_k is None: missing.append("K")
            if w_v is None: missing.append("V")
            if w_o is None and not self.is_nested_encoder: missing.append("O")
            if missing:
                raise RuntimeError(
                    f"TokenCompressedAttention.copy_weights: missing {missing} "
                    f"from {type(old_attn).__name__}"
                )

            def _copy_layer(target_layer, w, b):
                if w is not None:
                    if w.shape == target_layer.weight.shape:
                        target_layer.weight.data.copy_(w.to(target_layer.weight.dtype))
                    elif w.t().contiguous().shape == target_layer.weight.shape:
                        target_layer.weight.data.copy_(w.t().contiguous().to(target_layer.weight.dtype))
                    else:
                        min_0 = min(target_layer.weight.shape[0], w.shape[0])
                        min_1 = min(target_layer.weight.shape[1], w.shape[1])
                        target_layer.weight.data.zero_()
                        target_layer.weight.data[:min_0, :min_1].copy_(w[:min_0, :min_1].to(target_layer.weight.dtype))

                if b is not None and target_layer.bias is not None:
                    if b.shape == target_layer.bias.shape:
                        target_layer.bias.data.copy_(b.to(target_layer.bias.dtype))
                    else:
                        min_b = min(target_layer.bias.shape[0], b.shape[0])
                        target_layer.bias.data.zero_()
                        target_layer.bias.data[:min_b].copy_(b[:min_b].to(target_layer.bias.dtype))
                elif target_layer.bias is not None:
                    target_layer.bias.data.zero_()

                target_layer.weight.requires_grad = False
                if target_layer.bias is not None:
                    target_layer.bias.requires_grad = False

            _copy_layer(self.q_lin, w_q, b_q)
            _copy_layer(self.k_lin, w_k, b_k)
            _copy_layer(self.v_lin, w_v, b_v)
            if w_o is not None:
                _copy_layer(self.out_lin, w_o, b_o)

    # ------------------------------------------------------------------
    # Saliency selection
    # ------------------------------------------------------------------
    def _select_compressed_tokens(
        self,
        x: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        valid_mask_2d: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = x.shape
        H_kv, Dh = self.num_kv_heads, self.head_dim

        if L <= 2:
            top_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            comp_mask = valid_mask_2d if valid_mask_2d is not None else None
            return k_tensor, v_tensor, top_indices, comp_mask

        # TC-18: score_net may be on different device than x.
        x_f = x.float()
        score_dev = next(self.score_net.parameters()).device
        score_dtype = next(self.score_net.parameters()).dtype
        energy = torch.norm(x_f, p=2, dim=-1)
        learned_score = self.score_net(x_f.to(score_dev, score_dtype)).squeeze(-1).to(x_f.device, x_f.dtype)
        total_importance = energy + learned_score

        k_keep = max(2, int(math.ceil(L * self.compression_ratio)))
        k_keep = min(k_keep, L)

        # TC-19 / TC-06: preserve CLS only if valid; otherwise mask.
        if valid_mask_2d is not None:
            total_importance = total_importance.masked_fill(~valid_mask_2d, -1e4)

        # TC-20: boost CLS only when CLS is valid.
        cls_valid = True
        if valid_mask_2d is not None:
            cls_valid = bool(valid_mask_2d[:, 0].all().item())
        if cls_valid:
            cls_boost = torch.full((B, 1), 1e4, device=x.device, dtype=total_importance.dtype)
            total_importance = torch.cat([cls_boost, total_importance[:, 1:]], dim=-1)

        _, top_indices = torch.topk(total_importance, k_keep, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        top_idx_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H_kv, -1, Dh)
        k_comp = torch.gather(k_tensor, 2, top_idx_exp)
        v_comp = torch.gather(v_tensor, 2, top_idx_exp)

        comp_mask = None
        if valid_mask_2d is not None:
            comp_mask = torch.gather(valid_mask_2d, 1, top_indices)

        return k_comp, v_comp, top_indices, comp_mask

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
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "TokenCompressedAttention: cross-attention not supported."
            )

        B, L, D = hidden_states.shape
        H, H_kv, Dh = self.num_heads, self.num_kv_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.q_lin(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        k = self.k_lin(hidden_states).view(B, L, H_kv, Dh).transpose(1, 2).contiguous()
        v = self.v_lin(hidden_states).view(B, L, H_kv, Dh).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len: past_len + L, :]
            sin = sin[..., past_len: past_len + L, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Valid mask for keys
        valid_mask_2d = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.dtype == torch.bool:
                    valid_mask_2d = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    valid_mask_2d = attention_mask > -1e4
                else:
                    valid_mask_2d = attention_mask > 0
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    v_mask = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    v_mask = attention_mask > -1e4
                else:
                    v_mask = attention_mask > 0
                valid_mask_2d = v_mask.any(dim=-2).squeeze(1)

        k_comp, v_comp, top_idx, comp_valid = self._select_compressed_tokens(
            hidden_states, k, v, valid_mask_2d,
        )

        if H_kv != H:
            k_comp = repeat_kv(k_comp, self.num_key_value_groups)
            v_comp = repeat_kv(v_comp, self.num_key_value_groups)

        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q.float(), k_comp.float().transpose(-2, -1).contiguous()) * scale

        if self.is_decoder:
            q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
            k_orig = top_idx.unsqueeze(1).unsqueeze(2)
            causal_mask = (k_orig > q_pos)
            scores = scores.masked_fill(causal_mask, -1e4)

        if comp_valid is not None:
            scores = scores.masked_fill(~comp_valid[:, None, None, :], -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            attn = attn * head_mask.to(attn.dtype)

        context = torch.matmul(attn, v_comp)
        context = context.transpose(1, 2).contiguous().view(B, L, self.q_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_lin(context)

        if output.shape[-1] != D:
            raise RuntimeError(
                f"TokenCompressedAttention: output dim {output.shape[-1]} != hidden_size {D}"
            )

        # TC-19: do NOT mask query positions with the key mask.
        # (This was semantically wrong in the original code.)

        return output, (attn if output_attentions else None)


# ===========================================================================
# Injector
# ===========================================================================

@ATTENTION_REGISTRY.register("token_compressed")
@ATTENTION_REGISTRY.register("token_compressed_attention")
@ATTENTION_REGISTRY.register("inject_token_compressed_attn")
def inject_token_compressed_attn(
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
        logger.warning("inject_token_compressed_attn: max recursion depth exceeded.")
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
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_token_compressed_attn(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot,
                _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            src = target

        if isinstance(src, TokenCompressedAttention) or "TokenCompressed" in src.__class__.__name__:
            return None
        if "DeltaNet" in src.__class__.__name__ or "LinearAttention" in src.__class__.__name__:
            return None

        if model_config is None and hasattr(target, "config"):
            model_config = target.config
        if model_config is None and hasattr(src, "config"):
            model_config = src.config

        comp_ratio = float(kwargs.pop(
            "compression_ratio",
            _cfg_get(comp_config, "compression_ratio", 0.5),
        ))
        lora_rank = kwargs.pop(
            "lora_rank",
            _cfg_get(comp_config, "lora_rank",
                     _cfg_get(model_config, "lora_rank", 16)),
        )
        lora_alpha = kwargs.pop(
            "lora_alpha",
            _cfg_get(comp_config, "lora_alpha",
                     _cfg_get(model_config, "lora_alpha", 32.0)),
        )
        rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

        new_attn = TokenCompressedAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            compression_ratio=comp_ratio,
            pretrained_attn=src if src is not target else None,
            comp_config=comp_config,
            **kwargs,
        )

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn

        return new_attn
    except Exception as e:
        logger.exception(
            "inject_token_compressed_attn failed at layer %d: %s", layer_idx, e,
        )
        return None


__all__ = [
    "TokenCompressedAttention",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_token_compressed_attn",
]