"""
Token-Compressed Attention.
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
from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
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


@ATTENTION_REGISTRY.register("token_compressed", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("token_compressed_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("token_compressed_module", supports_causal=True, supports_gqa=True)
class TokenCompressedAttention(BaseAttention):
    r"""Token-Compressed Attention Module.

    Compresses Key and Value sequence tokens using energy/norm selection while preserving [CLS] (index 0).
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
        super().__init__(config, rank)
        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn", None)

        dim = None
        if config is not None and not isinstance(config, int):
            dim = _cfg_get(config, "hidden_size", _cfg_get(config, "dim", _cfg_get(config, "d_model", None)))
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
                dim = src.c_attn.weight.shape[0] if src.c_attn.weight.shape[0] != 3 * src.c_attn.weight.shape[1] else src.c_attn.weight.shape[1]
        if dim is None:
            dim = kwargs.get("hidden_size", 768)

        num_heads = None
        if "num_heads" in kwargs and kwargs["num_heads"] is not None:
            num_heads = kwargs["num_heads"]
        elif config is not None and not isinstance(config, int):
            num_heads = _cfg_get(config, "num_attention_heads", _cfg_get(config, "n_heads", _cfg_get(config, "num_heads", None)))
        if num_heads is None and comp_config is not None:
            num_heads = _cfg_get(comp_config, "num_attention_heads", _cfg_get(comp_config, "n_heads", None))
        if num_heads is None and src is not None:
            num_heads = getattr(src, "num_heads", getattr(src, "num_attention_heads", getattr(src, "n_head", None)))
        if num_heads is None:
            num_heads = 12

        self.dim: int = int(dim)
        self.num_heads: int = max(1, int(num_heads))
        self.head_dim: int = int(
            _cfg_get(config, "head_dim", kwargs.get("head_dim", self.dim // self.num_heads))
            if config is not None
            else kwargs.get("head_dim", self.dim // self.num_heads)
        )

        cfg_kv = None
        if config is not None:
            cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        if cfg_kv is None and comp_config is not None:
            cfg_kv = _cfg_get(comp_config, "num_key_value_heads", _cfg_get(comp_config, "num_kv_heads", None))
        if cfg_kv is None and src is not None:
            cfg_kv = getattr(src, "num_key_value_heads", getattr(src, "num_kv_heads", None))
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_key_value_groups: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.compression_ratio: float = float(kwargs.get("compression_ratio", _cfg_get(comp_config, "compression_ratio", compression_ratio)))

        # Uniform LoRA configuration: read independently from kwargs, comp_config, config
        if "lora_rank" in kwargs and kwargs["lora_rank"] is not None:
            resolved_lora_rank = int(kwargs["lora_rank"])
        elif comp_config is not None and _cfg_get(comp_config, "lora_rank") is not None:
            resolved_lora_rank = int(_cfg_get(comp_config, "lora_rank"))
        elif config is not None and _cfg_get(config, "lora_rank") is not None:
            resolved_lora_rank = int(_cfg_get(config, "lora_rank"))
        else:
            resolved_lora_rank = 16 if lora_rank is None else int(lora_rank)

        if "lora_alpha" in kwargs and kwargs["lora_alpha"] is not None:
            resolved_lora_alpha = float(kwargs["lora_alpha"])
        elif comp_config is not None and _cfg_get(comp_config, "lora_alpha") is not None:
            resolved_lora_alpha = float(_cfg_get(comp_config, "lora_alpha"))
        elif config is not None and _cfg_get(config, "lora_alpha") is not None:
            resolved_lora_alpha = float(_cfg_get(config, "lora_alpha"))
        else:
            resolved_lora_alpha = 32.0 if lora_alpha is None else float(lora_alpha)

        self.lora_rank: int = resolved_lora_rank
        self.lora_alpha: float = resolved_lora_alpha

        # Decoder / Causal flag
        if is_decoder is not None:
            self.is_decoder: bool = bool(is_decoder)
        elif kwargs.get("is_decoder") is not None:
            self.is_decoder = bool(kwargs["is_decoder"])
        elif src is not None and getattr(src, "is_decoder", False):
            self.is_decoder = True
        elif config is not None:
            self.is_decoder = bool(
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "model_type", "") in ("gpt2", "qwen", "qwen2", "qwen3_5", "llama")
            )
        else:
            self.is_decoder = False

        self.is_nested_encoder: bool = False
        self.rotary_emb: Optional[Any] = getattr(src, "rotary_emb", None) if src is not None else None

        self.q_lin = LoRALinear(
            self.dim, self.q_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.k_lin = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.v_lin = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )
        self.out_lin = LoRALinear(
            self.q_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True
        )

        self.score_net = nn.Sequential(
            nn.Linear(self.dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        dropout_p = (
            _cfg_get(config, "dropout", _cfg_get(config, "attention_probs_dropout_prob", 0.1))
            if config is not None
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            if hasattr(old_attn, "rotary_emb") and getattr(old_attn, "rotary_emb", None) is not None:
                self.rotary_emb = old_attn.rotary_emb

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

            if hasattr(src, "rotary_emb") and getattr(src, "rotary_emb", None) is not None:
                self.rotary_emb = src.rotary_emb

            # Helper to extract base layer from LoRALinear or Conv1D wrapper
            def _unwrap_linear(layer: Any) -> Any:
                if hasattr(layer, "base_layer"):
                    return layer.base_layer
                return layer

            # Pattern 1: GPT-2 Fused c_attn / c_proj
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_layer = _unwrap_linear(src.c_attn)
                c_proj_layer = _unwrap_linear(src.c_proj)

                c_attn_w = c_attn_layer.weight.detach()
                c_attn_b = c_attn_layer.bias.detach() if c_attn_layer.bias is not None else None
                # Conv1D weights stored as [in_dim, out_dim]. If first dim is self.dim, transpose to [out_dim, in_dim]
                if c_attn_w.dim() == 2 and c_attn_w.shape[0] == self.dim and c_attn_w.shape[1] != self.dim:
                    c_attn_w = c_attn_w.t().contiguous()

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

                out_w = c_proj_layer.weight.detach()
                if out_w.dim() == 2 and out_w.shape[0] == self.dim and out_w.shape[1] != self.dim:
                    out_w = out_w.t().contiguous()
                w_o = out_w
                b_o = c_proj_layer.bias.detach() if c_proj_layer.bias is not None else None

            # Pattern 2: RoBERTa / BERT separate query, key, value
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

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
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

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
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
                    f"TokenCompressedAttention: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )

            # Frobenius norm checks on extracted base weights
            for name, w_tensor in [("Q", w_q), ("K", w_k), ("V", w_v)]:
                if w_tensor is not None:
                    norm = float(torch.linalg.norm(w_tensor.float()))
                    if norm < 1e-4:
                        logger.warning("Attention module copy_weights warning. Proceeding with initialized weights.")
            if w_o is not None and not self.is_nested_encoder:
                norm_o = float(torch.linalg.norm(w_o.float()))
                if norm_o < 1e-4:
                    logger.warning("Attention module copy_weights warning. Proceeding with initialized weights.")

            def _copy_layer(target_layer, w, b, src_sub=None):
                if w is not None:
                    if w.shape == target_layer.weight.shape:
                        target_layer.weight.copy_(w.to(target_layer.weight.dtype))
                    elif w.t().contiguous().shape == target_layer.weight.shape:
                        target_layer.weight.copy_(w.t().contiguous().to(target_layer.weight.dtype))
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

                # Ensure base weights remain frozen
                target_layer.weight.requires_grad = False
                if target_layer.bias is not None:
                    target_layer.bias.requires_grad = False

                # Preserve source LoRA if present
                if src_sub is not None and hasattr(src_sub, "lora_A") and hasattr(src_sub, "lora_B"):
                    if src_sub.lora_A is not None and src_sub.lora_B is not None and target_layer.lora_A is not None and target_layer.lora_B is not None:
                        src_lA = getattr(src_sub.lora_A, "weight", src_sub.lora_A)
                        src_lB = getattr(src_sub.lora_B, "weight", src_sub.lora_B)
                        if hasattr(src_lA, "data") and hasattr(src_lB, "data"):
                            r_min = min(target_layer.lora_A.shape[0], src_lA.shape[0])
                            in_min = min(target_layer.lora_A.shape[1], src_lA.shape[1])
                            target_layer.lora_A.data[:r_min, :in_min].copy_(src_lA.data[:r_min, :in_min])
                            target_layer.lora_A.requires_grad = True

                            out_min = min(target_layer.lora_B.shape[0], src_lB.shape[0])
                            r_min_b = min(target_layer.lora_B.shape[1], src_lB.shape[1])
                            target_layer.lora_B.data[:out_min, :r_min_b].copy_(src_lB.data[:out_min, :r_min_b])
                            target_layer.lora_B.requires_grad = True
                        if hasattr(src_sub, "lora_scaling"):
                            target_layer.scaling = src_sub.lora_scaling

            sub_q = getattr(src, "q_proj", getattr(src, "q_lin", getattr(src, "query", None)))
            sub_k = getattr(src, "k_proj", getattr(src, "k_lin", getattr(src, "key", None)))
            sub_v = getattr(src, "v_proj", getattr(src, "v_lin", getattr(src, "value", None)))
            sub_o = getattr(src, "o_proj", getattr(src, "out_lin", getattr(src, "dense", getattr(src, "out_proj", None))))

            _copy_layer(self.q_lin, w_q, b_q, sub_q)
            _copy_layer(self.k_lin, w_k, b_k, sub_k)
            _copy_layer(self.v_lin, w_v, b_v, sub_v)
            if w_o is not None:
                _copy_layer(self.out_lin, w_o, b_o, sub_o)

    def _select_compressed_tokens(
        self, x: torch.Tensor, k_tensor: torch.Tensor, v_tensor: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = x.shape
        H_kv, Dh = self.num_kv_heads, self.head_dim

        if L <= 2:
            top_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            return k_tensor, v_tensor, top_indices, mask

        # Saliency scoring based on token norm + learnable score
        energy = torch.norm(x.float(), p=2, dim=-1)  # (B, L)
        learned_score = self.score_net(x.float()).squeeze(-1)  # (B, L)
        total_importance = energy + learned_score

        k_keep = max(1, min(L, max(2, int(math.ceil(L * self.compression_ratio)))))

        # Token 0 is always [CLS]
        total_importance[:, 0] = 1e4

        if mask is not None:
            if mask.dim() == 2:
                total_importance = total_importance.masked_fill(mask == 0, -10000.0)
                total_importance[:, 0] = 1e4
            elif mask.dim() == 4:
                valid_mask = (mask[:, 0, 0, :] > -10000.0)
                total_importance = total_importance.masked_fill(~valid_mask, -10000.0)
                total_importance[:, 0] = 1e4

        _, top_indices = torch.topk(total_importance, k_keep, dim=-1)  # (B, k_keep)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        # Gather K and V: [B, H_kv, k_keep, Dh]
        top_idx_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H_kv, -1, Dh)
        k_comp = torch.gather(k_tensor, 2, top_idx_exp)
        v_comp = torch.gather(v_tensor, 2, top_idx_exp)

        comp_mask = None
        if mask is not None:
            if mask.dim() == 2:
                comp_mask = torch.gather(mask, 1, top_indices)
            elif mask.dim() == 4:
                top_idx_4d = top_indices.unsqueeze(1).unsqueeze(2).expand(-1, mask.shape[1], mask.shape[2], -1)
                comp_mask = torch.gather(mask, -1, top_idx_4d)

        return k_comp, v_comp, top_indices, comp_mask

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
        H = self.num_heads
        H_kv = self.num_kv_heads
        Dh = self.head_dim

        q = self.q_lin(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        k = self.k_lin(hidden_states).view(B, L, H_kv, Dh).transpose(1, 2).contiguous()
        v = self.v_lin(hidden_states).view(B, L, H_kv, Dh).transpose(1, 2).contiguous()

        # Partial Rotary Position Embeddings (RoPE)
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, kwargs.get("position_ids", None))
        elif self.rotary_emb is not None:
            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is None:
                pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = self.rotary_emb(v, seq_len=L)
            except Exception:
                try:
                    cos, sin = self.rotary_emb(v, pos_ids)
                except Exception:
                    cos, sin = None, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids=pos_ids)

        # Compress tokens for K and V
        k, v, top_idx, comp_mask = self._select_compressed_tokens(hidden_states, k, v, attention_mask)
        k_len = k.shape[2]

        # GQA expansion
        if H_kv != H:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2).contiguous()) * scale  # [B, H, L, k_len]

        # Causal mask for decoders
        if self.is_decoder:
            # Query token i can only attend to compressed keys whose original position <= i
            q_pos = torch.arange(L, device=scores.device).view(1, 1, L, 1)  # [1, 1, L, 1]
            k_orig_pos = top_idx.unsqueeze(1).unsqueeze(2)                  # [B, 1, 1, k_len]
            causal_mask = (k_orig_pos > q_pos)                              # [B, 1, L, k_len]
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if comp_mask is not None:
            scores = apply_attention_mask(scores, comp_mask)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights.to(v.dtype))

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            elif head_mask.dim() == 4:
                head_mask = head_mask
            attn_weights = attn_weights * head_mask.to(attn_weights.dtype)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, H * Dh)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_lin(context)

        if output.shape[-1] != hidden_states.shape[-1]:
            output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1])) if output.shape[-1] < hidden_states.shape[-1] else output[..., :hidden_states.shape[-1]]

        valid_m = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
        if valid_m is not None:
            output = output * valid_m.unsqueeze(-1).to(output.dtype)

        return (output, (attn_weights if output_attentions else None))


@ATTENTION_REGISTRY.register("token_compressed_inject")
@ATTENTION_REGISTRY.register("token_compressed_attention_inject")
@ATTENTION_REGISTRY.register("inject_token_compressed_attn")
def inject_token_compressed_attn(
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
            res = inject_token_compressed_attn(
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

    """Safe injection hook for TokenCompressedAttention."""
    src = old_attn
    if src is None and layer is not None:
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

    # Uniform parameter extraction from kwargs, comp_config, model_config
    comp_ratio = float(kwargs.pop("compression_ratio", _cfg_get(comp_config, "compression_ratio", 0.5)))
    lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", _cfg_get(model_config, "lora_rank", 16)))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", _cfg_get(model_config, "lora_alpha", 32.0)))
    rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

    # Pre-flight weight extraction test before replacing original attention
    try:
        new_attn = TokenCompressedAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            compression_ratio=comp_ratio,
            pretrained_attn=src,
            comp_config=comp_config,
            **kwargs,
        )

        if layer is not None:
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                    return new_attn
                else:
                    layer.attention = new_attn
                    return new_attn
            elif hasattr(layer, "self"):
                layer.self = new_attn
                new_attn.is_nested_encoder = True
                return new_attn
            for attr in ("self_attn", "attn"):
                if hasattr(layer, attr):
                    setattr(layer, attr, new_attn)
                    return new_attn

        return new_attn
    except Exception as e:
        logger.error("inject_token_compressed_attn: Pre-flight weight copy failed at layer %d: %s. Aborting replacement.", layer_idx, e)
        return None

