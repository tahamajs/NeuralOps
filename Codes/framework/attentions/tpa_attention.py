r"""
TPAttention: Tensor Product Attention with Factorised Query-Key Projections.
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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions: [-x2, x1], safe for odd dimensions."""
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    if rotary_dim == 0:
        return x
    x_rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
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
    """Applies Rotary Position Embedding (RoPE) with partial dimension and independent seq-len support."""
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

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
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


@ATTENTION_REGISTRY.register("tpa", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("tpa_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("tp_attention", supports_causal=True, supports_gqa=True)
class TPAttention(BaseAttention):
    r"""Tensor Product Attention with factorised low-rank query-key projection."""

    supports_gqa: bool = True
    supports_causal: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 8,
        pretrained_attn: Optional[nn.Module] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn", None)

        dim = None
        if config is not None and not isinstance(config, int):
            dim = _cfg_get(config, "hidden_size", _cfg_get(config, "dim", _cfg_get(config, "d_model", None)))
        if dim is None and src is not None:
            dim = getattr(src, "hidden_size", getattr(src, "embed_dim", None))
            if dim is None and hasattr(src, "q_proj") and hasattr(src.q_proj, "in_features"):
                dim = src.q_proj.in_features
            elif dim is None and hasattr(src, "q_lin") and hasattr(src.q_lin, "in_features"):
                dim = src.q_lin.in_features
            elif dim is None and hasattr(src, "query") and hasattr(src.query, "in_features"):
                dim = src.query.in_features
            elif dim is None and hasattr(src, "c_attn") and hasattr(src.c_attn, "weight"):
                dim = src.c_attn.weight.shape[0] if src.c_attn.weight.shape[0] != 3 * src.c_attn.weight.shape[1] else src.c_attn.weight.shape[1]
        if dim is None:
            dim = kwargs.get("hidden_size", 768)

        num_heads = None
        if config is not None and not isinstance(config, int):
            num_heads = _cfg_get(config, "num_attention_heads", _cfg_get(config, "n_heads", _cfg_get(config, "num_heads", None)))
        if num_heads is None and src is not None:
            num_heads = getattr(src, "num_heads", getattr(src, "num_attention_heads", getattr(src, "n_head", None)))
        if num_heads is None:
            num_heads = 12

        self.dim: int = int(dim)
        self.num_heads: int = max(1, int(num_heads))
        self.head_dim: int = getattr(
            config, "head_dim", kwargs.get("head_dim", self.dim // self.num_heads)
        ) if config is not None else self.dim // self.num_heads

        cfg_kv = (
            _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
            if config is not None
            else None
        )
        if cfg_kv is None and src is not None:
            cfg_kv = getattr(src, "num_key_value_heads", getattr(src, "num_kv_heads", None))
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_key_value_groups: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.rank: int = kwargs.get("rank", _cfg_get(config, "rank", rank))
        self.lora_rank: int = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha: float = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        # Decoder / Causal flag
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None:
            self.is_decoder = bool(_cfg_get(config, "is_decoder", False))
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False

        # Factorised projections for Q and K
        self.q_proj1 = LoRALinear(
            self.dim, self.num_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.q_proj2 = LoRALinear(
            self.dim, self.num_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.k_proj1 = LoRALinear(
            self.dim, self.num_kv_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.k_proj2 = LoRALinear(
            self.dim, self.num_kv_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )

        self.v_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.out_proj = LoRALinear(
            self.q_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )

        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

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
                    f"TPAttention: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )

            # SVD factorized initialization for Q and K factor pairs
            def _init_tp(W: torch.Tensor, proj1: LoRALinear, proj2: LoRALinear, num_heads: int, r: int, src_sub=None) -> None:
                W_f = W.float()
                U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
                eff_r = min(num_heads * r, U.size(1), Vt.size(0))
                S_sqrt = torch.diag(torch.sqrt(S[:eff_r].clamp(min=1e-6)))

                proj1.weight.zero_()
                proj2.weight.zero_()

                p1 = (U[:num_heads * r, :eff_r] @ S_sqrt)
                p2 = (S_sqrt @ Vt[:eff_r, :])

                min_p1_0 = min(proj1.weight.shape[0], p1.size(0))
                min_p1_1 = min(proj1.weight.shape[1], p1.size(1))
                proj1.weight[:min_p1_0, :min_p1_1].copy_(p1[:min_p1_0, :min_p1_1].to(proj1.weight.dtype))

                min_p2_0 = min(proj2.weight.shape[0], p2.size(0))
                min_p2_1 = min(proj2.weight.shape[1], p2.size(1))
                proj2.weight[:min_p2_0, :min_p2_1].copy_(p2[:min_p2_0, :min_p2_1].to(proj2.weight.dtype))

                if float(torch.linalg.norm(proj1.weight)) < 1e-4:
                    nn.init.normal_(proj1.weight, std=0.02)
                if float(torch.linalg.norm(proj2.weight)) < 1e-4:
                    nn.init.normal_(proj2.weight, std=0.02)

                if src_sub is not None and hasattr(src_sub, "lora_A") and hasattr(src_sub, "lora_B"):
                    if src_sub.lora_A is not None and src_sub.lora_B is not None:
                        s_lA = getattr(src_sub.lora_A, "weight", src_sub.lora_A)
                        s_lB = getattr(src_sub.lora_B, "weight", src_sub.lora_B)
                        if hasattr(s_lA, "data") and hasattr(s_lB, "data"):
                            if proj1.lora_A is not None and proj1.lora_B is not None:
                                r_min = min(proj1.lora_A.shape[0], s_lA.shape[0])
                                in_min = min(proj1.lora_A.shape[1], s_lA.shape[1])
                                proj1.lora_A.data[:r_min, :in_min].copy_(s_lA.data[:r_min, :in_min])
                                proj1.lora_A.requires_grad = True

                                out_min = min(proj1.lora_B.shape[0], s_lB.shape[0])
                                r_min_b = min(proj1.lora_B.shape[1], s_lB.shape[1])
                                proj1.lora_B.data[:out_min, :r_min_b].copy_(s_lB.data[:out_min, :r_min_b])
                                proj1.lora_B.requires_grad = True

            def _copy_layer(target_layer, w, b, src_sub=None):
                if w is not None:
                    if w.shape == target_layer.weight.shape:
                        target_layer.weight.copy_(w.to(target_layer.weight.dtype))
                    elif w.t().shape == target_layer.weight.shape:
                        target_layer.weight.copy_(w.t().to(target_layer.weight.dtype))
                    else:
                        min_0 = min(target_layer.weight.shape[0], w.shape[0])
                        min_1 = min(target_layer.weight.shape[1], w.shape[1])
                        target_layer.weight.zero_()
                        target_layer.weight[:min_0, :min_1].copy_(w[:min_0, :min_1].to(target_layer.weight.dtype))
                if b is not None and target_layer.bias is not None:
                    if b.shape == target_layer.bias.shape:
                        target_layer.bias.copy_(b.to(target_layer.bias.dtype))
                    else:
                        min_b = min(target_layer.bias.shape[0], b.shape[0])
                        target_layer.bias.zero_()
                        target_layer.bias[:min_b].copy_(b[:min_b].to(target_layer.bias.dtype))
                elif target_layer.bias is not None:
                    target_layer.bias.zero_()

                if src_sub is not None and hasattr(src_sub, "lora_A") and hasattr(src_sub, "lora_B"):
                    if src_sub.lora_A is not None and src_sub.lora_B is not None and target_layer.lora_A is not None and target_layer.lora_B is not None:
                        src_lA = getattr(src_sub.lora_A, "weight", src_sub.lora_A)
                        src_lB = getattr(src_sub.lora_B, "weight", src_sub.lora_B)
                        if hasattr(s_lA, "data") and hasattr(s_lB, "data"):
                            r_min = min(target_layer.lora_A.shape[0], src_lA.shape[0])
                            in_min = min(target_layer.lora_A.shape[1], src_lA.shape[1])
                            target_layer.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                            target_layer.lora_A.requires_grad = True

                            out_min = min(target_layer.lora_B.shape[0], src_lB.shape[0])
                            r_min_b = min(target_layer.lora_B.shape[1], src_lB.shape[1])
                            target_layer.lora_B.data[:out_min, :r_min_b].copy_(src_lB.data[:out_min, :r_min_b])
                            target_layer.lora_B.requires_grad = True

            sub_q = getattr(src, "q_proj", getattr(src, "q_lin", getattr(src, "query", None)))
            sub_k = getattr(src, "k_proj", getattr(src, "k_lin", getattr(src, "key", None)))
            sub_v = getattr(src, "v_proj", getattr(src, "v_lin", getattr(src, "value", None)))
            sub_o = getattr(src, "o_proj", getattr(src, "out_lin", getattr(src, "dense", getattr(src, "out_proj", None))))

            _init_tp(w_q, self.q_proj1, self.q_proj2, self.num_heads, self.rank, sub_q)
            _init_tp(w_k, self.k_proj1, self.k_proj2, self.num_kv_heads, self.rank, sub_k)

            _copy_layer(self.v_proj, w_v, b_v, sub_v)
            f_norm_v = float(torch.linalg.norm(self.v_proj.weight))
            if f_norm_v < 1e-4:
                raise RuntimeError(f"TPAttention: V weight norm is near zero: {f_norm_v}")

            if w_o is not None:
                _copy_layer(self.out_proj, w_o, b_o, sub_o)
                f_norm_o = float(torch.linalg.norm(self.out_proj.weight))
                if f_norm_o < 1e-4:
                    raise RuntimeError(f"TPAttention: Out projection norm is near zero: {f_norm_o}")

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
        B, L, D = hidden_states.shape
        H, H_kv, r, hd = self.num_heads, self.num_kv_heads, self.rank, self.head_dim

        q1 = self.q_proj1(hidden_states).view(B, L, H, r)
        q2 = self.q_proj2(hidden_states).view(B, L, H, r)
        k1 = self.k_proj1(hidden_states).view(B, L, H_kv, r)
        k2 = self.k_proj2(hidden_states).view(B, L, H_kv, r)
        v = self.v_proj(hidden_states).view(B, L, H_kv, hd)

        # Tensor product: Q = q1 ⊗ q2, shape [B, L, H, r*r]
        Q = torch.einsum("blhi,blhj->blhij", q1, q2).reshape(B, L, H, r * r).permute(0, 2, 1, 3)
        K = torch.einsum("blhi,blhj->blhij", k1, k2).reshape(B, L, H_kv, r * r).permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)  # (B, H_kv, L, hd)

        # Partial Rotary Position Embeddings (RoPE)
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # GQA expansion
        if H_kv != H:
            K = repeat_kv(K, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        scale = 1.0 / math.sqrt(r * r)
        scores = torch.matmul(Q.float(), K.float().transpose(-1, -2)) * scale

        # Causal mask for decoders
        if self.is_decoder and L > 1:
            causal_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=scores.device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        mask = attention_mask if attention_mask is not None else kwargs.get("attn_mask", None)
        if mask is not None:
            if mask.dim() == 2:
                mask_t = mask[:, None, None, :].to(scores.dtype)
                scores = scores + (1.0 - mask_t) * -10000.0
            elif mask.dim() == 3:
                scores = scores + mask.unsqueeze(1).to(scores.dtype)
            elif mask.dim() == 4:
                scores = scores + mask.to(scores.dtype)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights.to(v.dtype))

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, H * hd)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        if output.shape[-1] != hidden_states.shape[-1]:
            output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1])) if output.shape[-1] < hidden_states.shape[-1] else output[..., :hidden_states.shape[-1]]

        return (output, (attn_weights if output_attentions else None))


@ATTENTION_REGISTRY.register("tpa_inject")
@ATTENTION_REGISTRY.register("tp_attention_inject")
@ATTENTION_REGISTRY.register("inject_tpa")
def inject_tpa(
    layer: Any,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    **kwargs: Any,
) -> Optional[TPAttention]:
    """Safe injection hook for TPAttention."""
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(layer, attr):
                    src = getattr(layer, attr)
                    break

        if src is None:
            return None

        if model_config is None and hasattr(layer, "config"):
            model_config = layer.config
        if model_config is None and hasattr(src, "config"):
            model_config = src.config

        rank = kwargs.pop("rank", _cfg_get(comp_config, "tpa_rank", _cfg_get(comp_config, "rank", 8)))
        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))

        new_attn = TPAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=src,
            **kwargs,
        )

        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(layer, attr):
                setattr(layer, attr, new_attn)
                return new_attn

        layer.attention = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_tpa failed at layer %d: %s", layer_idx, e)
        return None
