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
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None:
            self.is_decoder = bool(getattr(config, "is_decoder", False))
        else:
            self.is_decoder = False

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
        self.out_proj = LoRALinear(
            self.q_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        nn.init.constant_(self.W_f.bias, -3.0)

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
                if c_attn_w.shape[0] == self.dim and c_attn_w.shape[1] != self.dim:
                    c_attn_w = c_attn_w.t()

                q_dim = self.q_head_size
                k_dim = self.kv_head_size
                v_dim = self.kv_head_size

                w_q = c_attn_w[:q_dim, :]
                w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]

                out_w = src.c_proj.weight.detach()
                if out_w.shape[0] == self.dim and out_w.shape[1] != self.dim:
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
                raise RuntimeError(
                    f"HGRN2OuterBlock: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )

            self.q_proj.weight.copy_(w_q.to(self.q_proj.weight.dtype))
            self.k_proj.weight.copy_(w_k.to(self.k_proj.weight.dtype))
            self.v_proj.weight.copy_(w_v.to(self.v_proj.weight.dtype))

            f_norm_q = float(torch.linalg.norm(self.q_proj.weight))
            if f_norm_q < 1e-4:
                raise RuntimeError(f"HGRN2OuterBlock: Q weight norm is near zero: {f_norm_q}")

            if w_o is not None:
                self.out_proj.weight.copy_(w_o.to(self.out_proj.weight.dtype))
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o.to(self.out_proj.bias.dtype))
                f_norm_o = float(torch.linalg.norm(self.out_proj.weight))
                if f_norm_o < 1e-4:
                    raise RuntimeError(f"HGRN2OuterBlock: Out projection norm is near zero: {f_norm_o}")

    def _outer_scan_parallel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        f: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Parallel outer-product attention scan (zero Python loops)."""
        B, H, L, d = q.shape

        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        f_f = f.float()

        log_f = torch.log(f_f.clamp(min=1e-4, max=1.0 - 1e-4))  # [B, H, L]
        C = torch.cumsum(log_f, dim=-1)                         # [B, H, L]
        C_diff = C.unsqueeze(-1) - C.unsqueeze(-2)              # [B, H, L, L]

        tril_mask = torch.tril(torch.ones(L, L, device=q.device, dtype=torch.bool))
        D_causal = torch.exp(C_diff.clamp(min=-30.0, max=0.0)) * tril_mask.float()  # [B, H, L, L]

        attn_scores = q_f @ k_f.transpose(-1, -2)   # [B, H, L, L]
        weight_in = (1.0 - f_f).unsqueeze(-1)        # [B, H, L, 1]
        v_scaled = v_f * weight_in                   # [B, H, L, d]

        if mask is not None:
            mask_float = mask.float() if mask.dim() == 2 else mask[:, 0, 0, :].float()
            mask_2d = mask_float.unsqueeze(1).unsqueeze(-1) * mask_float.unsqueeze(1).unsqueeze(-2)
            D_causal = D_causal * mask_2d

        num_fwd = (attn_scores * D_causal) @ v_scaled
        denom_fwd = (attn_scores * D_causal) @ weight_in
        y_fwd = (num_fwd / (denom_fwd + 1e-4).clamp(min=1e-4)).to(q.dtype)

        if not self.is_decoder:
            triu_mask = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool))
            D_anti = torch.exp((-C_diff).clamp(min=-30.0, max=0.0)) * triu_mask.float()
            if mask is not None:
                D_anti = D_anti * mask_2d
            num_bwd = (attn_scores * D_anti) @ v_scaled
            denom_bwd = (attn_scores * D_anti) @ weight_in
            y_bwd = (num_bwd / (denom_bwd + 1e-4).clamp(min=1e-4)).to(q.dtype)
        else:
            y_bwd = None

        return y_fwd, y_bwd

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
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, d).transpose(1, 2)

        # Partial Rotary Position Embeddings (RoPE)
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Grouped-Query Attention expansion
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1) / math.sqrt(d)

        f = torch.exp(-F.softplus(self.W_f(hidden_states))).transpose(1, 2)  # [B, H, L]

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
def inject_hgrn2_outer(
    layer: nn.Module,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 6,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[HGRN2OuterBlock]:
    r"""Safe injection hook for HGRN2OuterBlock."""
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break

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

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_block)
                return new_block

        return new_block
    except Exception as e:
        logger.error("inject_hgrn2_outer failed at layer %d: %s", layer_idx, e)
        return None
