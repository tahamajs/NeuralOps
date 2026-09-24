"""HGRN-2 Outer-Product Recurrent Linear Attention Module.

Implements multi-head bidirectional/causal outer-product associative matrix memory updates
with log-space bounded decay gating, denominator normalization, and pretrained weight transfer.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import apply_attention_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) with partial dimension support."""
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


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
            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        """Freeze base weight and bias; keep LoRA adapters trainable."""
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("hgrn2_outer", supports_causal=True, supports_gqa=True)
class HGRN2OuterBlock(BaseAttention):
    r"""HGRN-2 Outer-Product Recurrent Linear Attention Module.

    Maintains multi-head outer-product associative matrix recurrence with input-dependent decay gating.
    """

    supports_gqa: bool = True
    supports_causal: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        dim: Optional[Union[int, Any]] = None,
        num_heads: Optional[int] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        # Resolve dim
        if dim is None:
            if isinstance(config, int):
                dim = config
            elif config is not None:
                dim = getattr(
                    config, "hidden_size", getattr(config, "d_model", getattr(config, "n_embd", getattr(config, "dim", None)))
                )
        if dim is None:
            dim = kwargs.get("hidden_size", kwargs.get("d_model", kwargs.get("dim", 768)))

        if hasattr(dim, "hidden_size"):
            dim = dim.hidden_size

        # Resolve num_heads
        if num_heads is None:
            if config is not None and not isinstance(config, int):
                num_heads = getattr(
                    config,
                    "num_attention_heads",
                    getattr(config, "num_heads", getattr(config, "n_heads", getattr(config, "n_head", None))),
                )
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))

        self.dim: int = int(dim)
        self.num_heads: int = int(num_heads)

        while self.num_heads > 1 and (self.dim % self.num_heads != 0):
            self.num_heads -= 1

        self.head_dim: int = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        cfg_kv = (
            getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
            if config is not None
            else None
        )
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_key_value_groups: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # Decoder / Causal flag
        _mt = str(getattr(config, "model_type", "") or "").lower()
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None:
            self.is_decoder = bool(getattr(config, "is_decoder", False) or any(_mt.startswith(p) for p in _decoder_prefixes))
        else:
            self.is_decoder = bool(kwargs.get("is_causal", False) or kwargs.get("is_decoder", False))

        self.is_nested_encoder: bool = False

        self.q_proj = LoRALinear(
            self.dim,
            self.q_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_proj = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_proj = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=False,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.W_f = nn.Linear(self.dim, self.num_heads, bias=True)
        self.state_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = LoRALinear(
            self.q_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        nn.init.normal_(self.W_f.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.W_f.bias, -2.5)

        old_attn = kwargs.get("old_attn", kwargs.get("pretrained_attn", None))
        if old_attn is not None:
            self.copy_weights(old_attn)

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Check nested encoder container
            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    w_o = old_attn.output.dense.weight
                    b_o = old_attn.output.dense.bias
            else:
                src = old_attn

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w = src.c_attn.weight.detach()
                c_attn_b = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                cls_c = getattr(src.c_attn, "__class__", type(src.c_attn)).__name__
                if hasattr(src.c_attn, "nf") or "Conv1D" in cls_c or (c_attn_w.shape[0] == self.dim and c_attn_w.shape[1] != self.dim):
                    c_attn_w = c_attn_w.t()

                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size

                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]

                if c_attn_b is not None:
                    b_q = c_attn_b[:q_dim]
                    b_k = c_attn_b[q_dim : q_dim + k_dim]
                    b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]

                out_w = src.c_proj.weight.detach()
                cls_p = getattr(src.c_proj, "__class__", type(src.c_proj)).__name__
                if hasattr(src.c_proj, "nf") or "Conv1D" in cls_p or (out_w.ndim == 2 and out_w.shape[0] != self.dim):
                    out_w = out_w.t()
                w_o = out_w
                b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: RoBERTa / BERT separate query, key, value
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

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
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

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
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

            # Pattern 5: UniversalAdapter fallback
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    w_q, b_q = adapter.get_weight_bias(src, "q")
                    w_k, b_k = adapter.get_weight_bias(src, "k")
                    w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Strict validation
            missing = []
            if w_q is None:
                missing.append("Q projection")
            if w_k is None:
                missing.append("K projection")
            if w_v is None:
                missing.append("V projection")
            if w_o is None and not self.is_nested_encoder:
                missing.append("Out projection")

            if missing:
                logger.warning(
                    "HGRN2OuterBlock: Missing weights: %s from %s",
                    ", ".join(missing),
                    type(old_attn),
                )
                return

            for target_mod, w_src, b_src in [
                (self.q_proj, w_q, b_q),
                (self.k_proj, w_k, b_k),
                (self.v_proj, w_v, b_v),
            ]:
                tw = target_mod.weight
                w_cast = w_src.to(tw.dtype)
                if w_cast.shape == tw.shape:
                    tw.data.copy_(w_cast)
                else:
                    min_r = min(tw.shape[0], w_cast.shape[0])
                    min_c = min(tw.shape[1], w_cast.shape[1])
                    tw.data[:min_r, :min_c].copy_(w_cast[:min_r, :min_c])

                if b_src is not None and getattr(target_mod, "bias", None) is not None:
                    tb = target_mod.bias
                    b_cast = b_src.to(tb.dtype)
                    min_b = min(tb.numel(), b_cast.numel())
                    tb.data[:min_b].copy_(b_cast[:min_b])

            f_norm_q = float(torch.linalg.norm(self.q_proj.weight.float()))
            if f_norm_q < 1e-4:
                logger.warning("HGRN2OuterBlock: Q weight norm is near zero: %g", f_norm_q)

            if w_o is not None:
                tw_o = self.out_proj.weight
                w_o_cast = w_o.to(tw_o.dtype)
                if w_o_cast.shape == tw_o.shape:
                    tw_o.data.copy_(w_o_cast)
                else:
                    min_r = min(tw_o.shape[0], w_o_cast.shape[0])
                    min_c = min(tw_o.shape[1], w_o_cast.shape[1])
                    tw_o.data[:min_r, :min_c].copy_(w_o_cast[:min_r, :min_c])

                if b_o is not None and self.out_proj.bias is not None:
                    tb_o = self.out_proj.bias
                    b_o_cast = b_o.to(tb_o.dtype)
                    min_b = min(tb_o.numel(), b_o_cast.numel())
                    tb_o.data[:min_b].copy_(b_o_cast[:min_b])
                f_norm_o = float(torch.linalg.norm(self.out_proj.weight.float()))
                if f_norm_o < 1e-4:
                    logger.warning("HGRN2OuterBlock: Out projection norm is near zero: %g", f_norm_o)

    @staticmethod
    def _scan_fwd_fn(q_f: torch.Tensor, k_f: torch.Tensor, v_f: torch.Tensor, f_f: torch.Tensor, mask_float: Optional[torch.Tensor]) -> torch.Tensor:
        B, H, L, d = q_f.shape
        S = torch.zeros(B, H, d, d, device=q_f.device, dtype=torch.float32)
        z = torch.zeros(B, H, d, device=q_f.device, dtype=torch.float32)
        y_fwd_list = []
        for t in range(L):
            ft = f_f[:, :, t, None, None]
            ft_z = f_f[:, :, t, None]
            kt = k_f[:, :, t, :]
            vt = v_f[:, :, t, :]
            qt = q_f[:, :, t, :]
            if mask_float is not None:
                m_t = mask_float[:, None, t, None]
                kt = kt * m_t
                vt = vt * m_t
            kv = kt.contiguous().unsqueeze(-1) @ vt.contiguous().unsqueeze(-2)
            S = ft * S + (1.0 - ft) * kv
            z = z + kt
            num = (qt.contiguous().unsqueeze(-2) @ S).squeeze(-2)
            denom = (qt * z).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            y_t = num / denom
            y_fwd_list.append(y_t)
        return torch.stack(y_fwd_list, dim=2)

    @staticmethod
    def _scan_bwd_fn(q_f: torch.Tensor, k_f: torch.Tensor, v_f: torch.Tensor, f_f: torch.Tensor, mask_float: Optional[torch.Tensor]) -> torch.Tensor:
        B, H, L, d = q_f.shape
        S_bwd = torch.zeros(B, H, d, d, device=q_f.device, dtype=torch.float32)
        z_bwd = torch.zeros(B, H, d, device=q_f.device, dtype=torch.float32)
        y_bwd_list = []
        for t in reversed(range(L)):
            ft = f_f[:, :, t, None, None]
            ft_z = f_f[:, :, t, None]
            kt = k_f[:, :, t, :]
            vt = v_f[:, :, t, :]
            qt = q_f[:, :, t, :]
            if mask_float is not None:
                m_t = mask_float[:, None, t, None]
                kt = kt * m_t
                vt = vt * m_t
            kv = kt.contiguous().unsqueeze(-1) @ vt.contiguous().unsqueeze(-2)
            S_bwd = ft * S_bwd + (1.0 - ft) * kv
            z_bwd = z_bwd + kt
            num_b = (qt.unsqueeze(-2) @ S_bwd).squeeze(-2)
            denom_b = (qt * z_bwd).sum(dim=-1, keepdim=True).clamp(min=1e-6)
            y_t_b = num_b / denom_b
            y_bwd_list.append(y_t_b)
        return torch.stack(list(reversed(y_bwd_list)), dim=2)

    def _outer_scan_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Outer-product associative matrix recurrence with running states in FP32."""
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        f_f = f.float()  # [B, H, L]

        mask_float = None
        if mask is not None:
            if mask.dim() == 2:
                mask_float = mask.float()
            elif mask.dim() == 4:
                v = (mask > -1e4) if mask.dtype.is_floating_point else mask.bool()
                v = v.any(dim=1)
                if v.dim() == 3:
                    mask_float = v.any(dim=1).float()
                else:
                    mask_float = v.float()
            else:
                mask_float = mask.float()

        if self.training:
            import torch.utils.checkpoint as cp
            y_fwd = cp.checkpoint(self._scan_fwd_fn, q_f, k_f, v_f, f_f, mask_float, use_reentrant=False).to(q.dtype)
        else:
            y_fwd = self._scan_fwd_fn(q_f, k_f, v_f, f_f, mask_float).to(q.dtype)

        if self.is_decoder:
            return y_fwd, None

        if self.training:
            import torch.utils.checkpoint as cp
            y_bwd = cp.checkpoint(self._scan_bwd_fn, q_f, k_f, v_f, f_f, mask_float, use_reentrant=False).to(q.dtype)
        else:
            y_bwd = self._scan_bwd_fn(q_f, k_f, v_f, f_f, mask_float).to(q.dtype)
        return y_fwd, y_bwd

    @property
    def query(self) -> nn.Module:
        return self.q_proj

    @property
    def key(self) -> nn.Module:
        return self.k_proj

    @property
    def value(self) -> nn.Module:
        return self.v_proj

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
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """Forward pass for HGRN2OuterBlock."""
        if encoder_hidden_states is not None:
            raise NotImplementedError("Cross-attention is not supported in HGRN2OuterBlock")

        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2).contiguous()

        # Partial Rotary Position Embeddings (RoPE)
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Grouped-Query Attention expansion
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # Non-negative kernel mapping to prevent state sign-flipping and negative denominators
        q = F.elu(q) + 1.0
        k = (F.elu(k) + 1.0) / math.sqrt(d)

        # Bounded forget gating to prevent saturation
        f = torch.exp((-F.softplus(self.W_f(hidden_states))).clamp(min=-3.0, max=-1e-4)).transpose(1, 2)  # [B, H, L]

        mask_bool = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_bool = attention_mask.bool()
            elif attention_mask.dim() == 4:
                mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).bool()

        y_fwd, y_bwd = self._outer_scan_parallel(q, k, v, f, mask_bool)

        if self.is_decoder or y_bwd is None:
            y = y_fwd
        else:
            y = 0.5 * (y_fwd + y_bwd)

        y = self.state_norm(y)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            y = y * head_mask

        y = y.transpose(1, 2).contiguous().view(B, L, H * d)
        if self.is_nested_encoder:
            out = y
        else:
            out = self.out_proj(y)

        if attention_mask is not None and mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return (out, None)


@ATTENTION_REGISTRY.register("hgrn2_outer_inject")
@ATTENTION_REGISTRY.register("inject_hgrn2_outer")
def inject_hgrn2_outer(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
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
            res = inject_hgrn2_outer(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""Safe injection hook for HGRN2OuterBlock."""
    try:
        src = old_attn
        is_nested_target = False
        if src is None and layer is not None:
            if hasattr(layer, "attention") and hasattr(layer.attention, "self"):
                src = layer.attention.self
                is_nested_target = True
            elif hasattr(layer, "attention"):
                src = layer.attention
            elif hasattr(layer, "self"):
                src = layer.self
                is_nested_target = True
            elif hasattr(layer, "attn"):
                src = layer.attn
            elif hasattr(layer, "self_attn"):
                src = layer.self_attn

        if src is None:
            return None

        dim = 768
        num_heads = 12
        if model_config is not None:
            dim = getattr(model_config, "hidden_size", getattr(model_config, "dim", 768))
            num_heads = getattr(model_config, "num_attention_heads", 12)
        elif hasattr(layer, "config"):
            dim = getattr(layer.config, "hidden_size", 768)
            num_heads = getattr(layer.config, "num_attention_heads", 12)

        new_block = HGRN2OuterBlock(dim=dim, num_heads=num_heads, config=model_config, **kwargs)
        new_block.copy_weights(src)

        if is_nested_target and hasattr(layer, "attention") and hasattr(layer.attention, "self"):
            layer.attention.self = new_block
            return new_block
        elif hasattr(layer, "attention"):
            layer.attention = new_block
            return new_block
        elif hasattr(layer, "self"):
            layer.self = new_block
            return new_block
        elif hasattr(layer, "attn"):
            layer.attn = new_block
            return new_block
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_block
            return new_block

        return new_block
    except Exception as e:
        logger.error("inject_hgrn2_outer failed at layer %d: %s", layer_idx, e)
        return None
