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

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    HAS_SDPA_KERNEL = True
except ImportError:
    HAS_SDPA_KERNEL = False

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

    cos_sub = cos[..., : q.shape[-2], :rotary_dim]
    sin_sub = sin[..., : q.shape[-2], :rotary_dim]
    if cos_sub.device != q.device or cos_sub.dtype != q.dtype:
        cos_q = cos_sub.to(device=q.device, dtype=q.dtype)
        sin_q = sin_sub.to(device=q.device, dtype=q.dtype)
    else:
        cos_q, sin_q = cos_sub, sin_sub

    if cos_sub.device != k.device or cos_sub.dtype != k.dtype:
        cos_k = cos_sub.to(device=k.device, dtype=k.dtype)
        sin_k = sin_sub.to(device=k.device, dtype=k.dtype)
    else:
        cos_k, sin_k = cos_sub, sin_sub

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_out[..., :rotary_dim] = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_out[..., :rotary_dim] = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
    if q_pass.numel() > 0:
        q_out[..., rotary_dim:] = q_pass
        k_out[..., rotary_dim:] = k_pass
    return q_out, k_out


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
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, lora_rank)

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
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
    is_custom_attention: bool = True

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

        cfg_hd = _cfg_get(config, "head_dim", None) if config is not None else None
        kw_hd = kwargs.get("head_dim", None)
        if cfg_hd is not None:
            self.head_dim: int = int(cfg_hd)
        elif kw_hd is not None:
            self.head_dim: int = int(kw_hd)
        else:
            self.head_dim: int = self.dim // self.num_heads

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

        # Rank adjustment: ensure r * r matches head_dim if possible
        raw_r = kwargs.get("rank", _cfg_get(config, "rank", rank))
        self.rank = int(raw_r)

        self.lora_rank: int = kwargs.get("lora_rank", _cfg_get(config, "lora_rank", lora_rank))
        self.lora_alpha: float = kwargs.get("lora_alpha", _cfg_get(config, "lora_alpha", lora_alpha))

        # Projection if r * r != head_dim
        if self.rank * self.rank != self.head_dim:
            self.qk_proj: Optional[nn.Linear] = nn.Linear(self.rank * self.rank, self.head_dim, bias=False)
        else:
            self.qk_proj = None

        # Decoder / Causal flag
        _mt = str(_cfg_get(config, "model_type", "") or "").lower() if config is not None else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if is_decoder is not None:
            self.is_decoder: bool = is_decoder
        elif config is not None and (
            _cfg_get(config, "is_decoder", False)
            or _cfg_get(config, "causal", False)
            or any(_mt.startswith(p) for p in _decoder_prefixes)
        ):
            self.is_decoder = True
        else:
            self.is_decoder = bool(kwargs.get("is_decoder", False))

        self.is_nested_encoder: bool = False
        self.gradient_checkpointing: bool = bool(kwargs.get("gradient_checkpointing", True))

        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and src is not None:
            target_dtype = next(src.parameters(), torch.tensor([])).dtype
            if not isinstance(target_dtype, torch.dtype):
                target_dtype = torch.float32

        # Factorised projections for Q and K
        self.q_proj1 = LoRALinear(
            self.dim, self.num_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.q_proj2 = LoRALinear(
            self.dim, self.num_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.k_proj1 = LoRALinear(
            self.dim, self.num_kv_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.k_proj2 = LoRALinear(
            self.dim, self.num_kv_heads * self.rank, bias=False,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )

        self.v_proj = LoRALinear(
            self.dim, self.kv_head_size, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )
        self.out_proj = LoRALinear(
            self.q_head_size, self.dim, bias=True,
            lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True, dtype=target_dtype
        )

        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

        if target_dtype is not None:
            for proj in (self.q_proj1, self.q_proj2, self.k_proj1, self.k_proj2, self.v_proj, self.out_proj):
                proj.to(target_dtype)

    def copy_weights(self, old_attn: Any) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Check nested encoder container
            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            has_no_out_proj = not any(
                hasattr(old_attn, attr)
                for attr in ("dense", "out_proj", "o_proj", "c_proj", "out_lin", "proj")
            )
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query") and has_no_out_proj
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

            # Pattern 1: GPT-2 Conv1D (c_attn fused QKV)
            if hasattr(src, "c_attn"):
                W = src.c_attn.weight.detach()
                B_vec = src.c_attn.bias.detach() if src.c_attn.bias is not None else None
                if W.shape[0] != 3 * self.dim and W.shape[1] == 3 * self.dim:
                    W = W.t().contiguous()
                if W.shape[0] == 3 * self.dim:
                    w_q, w_k, w_v = W.chunk(3, dim=0)
                    if B_vec is not None:
                        b_q, b_k, b_v = B_vec.chunk(3, dim=0)

                if hasattr(src, "c_proj"):
                    w_o = src.c_proj.weight.detach()
                    if w_o.shape[0] != self.dim and w_o.shape[1] == self.dim:
                        w_o = w_o.t().contiguous()
                    b_o = src.c_proj.bias.detach() if src.c_proj.bias is not None else None

            # Pattern 2: BERT / RoBERTa (query, key, value, dense)
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
                    "TPAttention: Failed to extract weights: %s from %s", missing, type(old_attn).__name__
                )
                return

            def _init_tp(W: torch.Tensor, proj1: LoRALinear, proj2: LoRALinear, num_h: int, r: int, src_sub=None) -> None:
                import gc
                W_f = W.detach().float().cpu()
                U = S = Vt = None
                S_sqrt = p1 = p2 = None
                try:
                    U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
                    eff_r = min(num_h * r, U.size(1), Vt.size(0))
                    S_sqrt = torch.sqrt(S[:eff_r].clamp(min=1e-6))
                    p1 = U[:num_h * r, :eff_r] * S_sqrt.unsqueeze(0)
                    p2 = S_sqrt.unsqueeze(1) * Vt[:eff_r, :]

                    proj1.weight.data.zero_()
                    proj2.weight.data.zero_()
                    min_p1_0 = min(proj1.weight.shape[0], p1.size(0))
                    min_p1_1 = min(proj1.weight.shape[1], p1.size(1))
                    proj1.weight.data[:min_p1_0, :min_p1_1].copy_(
                        p1[:min_p1_0, :min_p1_1].to(proj1.weight.dtype)
                    )
                    min_p2_0 = min(proj2.weight.shape[0], p2.size(0))
                    min_p2_1 = min(proj2.weight.shape[1], p2.size(1))
                    proj2.weight.data[:min_p2_0, :min_p2_1].copy_(
                        p2[:min_p2_0, :min_p2_1].to(proj2.weight.dtype)
                    )
                except Exception:
                    nn.init.orthogonal_(proj1.weight)
                    nn.init.orthogonal_(proj2.weight)
                finally:
                    del W_f, U, S, Vt, S_sqrt, p1, p2
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if float(torch.linalg.norm(proj1.weight)) < 1e-4:
                    nn.init.orthogonal_(proj1.weight)
                if float(torch.linalg.norm(proj2.weight)) < 1e-4:
                    nn.init.orthogonal_(proj2.weight)

                if proj1.bias is not None:
                    proj1.bias.zero_()
                if proj2.bias is not None:
                    proj2.bias.zero_()
                proj1.weight.requires_grad = False
                proj2.weight.requires_grad = False

                # LoRA init with standard kaiming uniform / zeros (no copy from src_sub)
                if proj1.lora_A is not None and proj1.lora_B is not None:
                    nn.init.kaiming_uniform_(proj1.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(proj1.lora_B)
                    proj1.lora_A.requires_grad = True
                    proj1.lora_B.requires_grad = True
                if proj2.lora_A is not None and proj2.lora_B is not None:
                    nn.init.kaiming_uniform_(proj2.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(proj2.lora_B)
                    proj2.lora_A.requires_grad = True
                    proj2.lora_B.requires_grad = True

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

            sub_q = getattr(src, "q_proj", getattr(src, "q_lin", getattr(src, "query", None)))
            sub_k = getattr(src, "k_proj", getattr(src, "k_lin", getattr(src, "key", None)))
            sub_v = getattr(src, "v_proj", getattr(src, "v_lin", getattr(src, "value", None)))
            sub_o = getattr(src, "o_proj", getattr(src, "out_lin", getattr(src, "dense", getattr(src, "out_proj", None))))

            _init_tp(w_q, self.q_proj1, self.q_proj2, self.num_heads, self.rank, sub_q)
            _init_tp(w_k, self.k_proj1, self.k_proj2, self.num_kv_heads, self.rank, sub_k)

            _copy_layer(self.v_proj, w_v, b_v, sub_v)
            f_norm_v = float(torch.linalg.norm(self.v_proj.weight))
            if f_norm_v < 1e-4:
                logger.warning(f"TPAttention: V weight norm is near zero: {f_norm_v}")

            if w_o is not None:
                _copy_layer(self.out_proj, w_o, b_o, sub_o)
                f_norm_o = float(torch.linalg.norm(self.out_proj.weight))
                if f_norm_o < 1e-4:
                    logger.warning(f"TPAttention: Out projection norm is near zero: {f_norm_o}")

            if hasattr(old_attn, "is_decoder"):
                self.is_decoder = bool(old_attn.is_decoder)
            elif hasattr(old_attn, "c_attn") or any(x in old_attn.__class__.__name__.lower() for x in ("gpt", "qwen", "llama", "decoder")):
                self.is_decoder = True

            del w_q, w_k, w_v, w_o, b_q, b_k, b_v, b_o, sub_q, sub_k, sub_v, sub_o
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        # Projections with outer product
        def _compute_q(x: torch.Tensor) -> torch.Tensor:
            q1 = self.q_proj1(x).view(B, L, H, r)
            q2 = self.q_proj2(x).view(B, L, H, r)
            out_q = (q1.unsqueeze(-1) * q2.unsqueeze(-2)).flatten(-2, -1)
            if self.qk_proj is not None:
                out_q = self.qk_proj(out_q)
            return out_q

        def _compute_k(x: torch.Tensor) -> torch.Tensor:
            k1 = self.k_proj1(x).view(B, L, H_kv, r)
            k2 = self.k_proj2(x).view(B, L, H_kv, r)
            out_k = (k1.unsqueeze(-1) * k2.unsqueeze(-2)).flatten(-2, -1)
            if self.qk_proj is not None:
                out_k = self.qk_proj(out_k)
            return out_k

        if self.training and getattr(self, "gradient_checkpointing", True):
            try:
                from torch.utils.checkpoint import checkpoint
                Q = checkpoint(_compute_q, hidden_states, use_reentrant=False)
                K = checkpoint(_compute_k, hidden_states, use_reentrant=False)
            except Exception:
                Q = _compute_q(hidden_states)
                K = _compute_k(hidden_states)
        else:
            Q = _compute_q(hidden_states)
            K = _compute_k(hidden_states)

        v = self.v_proj(hidden_states).view(B, L, H_kv, hd)

        Q = Q.transpose(1, 2).contiguous()
        K = K.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # Partial Rotary Position Embeddings (RoPE)
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # GQA expansion
        if H_kv != H:
            K = repeat_kv(K, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        # 2. SDPA fast-path
        is_causal_mask = self.is_decoder and L > 1
        custom_mask = attention_mask if attention_mask is not None else kwargs.get("attn_mask", None)

        # Optimize attention mask for memory efficiency
        if custom_mask is not None:
            if custom_mask.dtype == torch.bool:
                if custom_mask.all():
                    custom_mask = None
                elif custom_mask.dim() == 2:
                    custom_mask = custom_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
                elif custom_mask.dim() == 3:
                    custom_mask = custom_mask.unsqueeze(1)  # [B, 1, L, S]
            elif torch.is_floating_point(custom_mask):
                if (custom_mask == 0.0).all():
                    custom_mask = None
                elif (custom_mask <= -1.0).any() and ((custom_mask == 0.0) | (custom_mask <= -1.0)).all():
                    # Extended HF mask (0.0 for keep, <= -1.0 for padding) -> convert to boolean for FlashAttention
                    custom_mask = (custom_mask > -1.0)
                    if custom_mask.dim() == 2:
                        custom_mask = custom_mask.unsqueeze(1).unsqueeze(2)
                    elif custom_mask.dim() == 3:
                        custom_mask = custom_mask.unsqueeze(1)
                else:
                    if custom_mask.dim() == 2:
                        custom_mask = custom_mask.unsqueeze(1).unsqueeze(2)
                    elif custom_mask.dim() == 3:
                        custom_mask = custom_mask.unsqueeze(1)
                    if custom_mask.dtype != Q.dtype:
                        custom_mask = custom_mask.to(dtype=Q.dtype)

        # SDPA (no .contiguous() calls to avoid memory duplication)
        if not output_attentions and head_mask is None:
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    context = F.scaled_dot_product_attention(
                        query=Q,
                        key=K,
                        value=v,
                        attn_mask=custom_mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=is_causal_mask if custom_mask is None else False,
                    )
            except (ImportError, RuntimeError, TypeError):
                try:
                    from torch.nn.attention import sdpa_kernel, SDPBackend
                    with sdpa_kernel([SDPBackend.MATH]):
                        context = F.scaled_dot_product_attention(
                            query=Q,
                            key=K,
                            value=v,
                            attn_mask=custom_mask,
                            dropout_p=self.dropout.p if self.training else 0.0,
                            is_causal=is_causal_mask if custom_mask is None else False,
                        )
                except Exception:
                    context = F.scaled_dot_product_attention(
                        query=Q,
                        key=K,
                        value=v,
                        attn_mask=custom_mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                        is_causal=is_causal_mask if custom_mask is None else False,
                    )
            attn_weights = None
        else:
            # Fallback path if output_attentions or head_mask are explicitly requested
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(Q, K.transpose(-1, -2).contiguous()) * scale
            if is_causal_mask and custom_mask is None:
                causal_mask = torch.triu(
                    torch.full((L, L), float("-inf"), device=scores.device, dtype=scores.dtype),
                    diagonal=1,
                )
                scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            elif custom_mask is not None:
                scores = apply_attention_mask(scores, custom_mask)

            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            attn_weights = self.dropout(attn_weights)
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.view(head_mask.size(0), head_mask.size(1), 1, 1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                attn_weights = attn_weights * head_mask.to(attn_weights.dtype)
            context = torch.matmul(attn_weights, v)

        del Q, K, v  # Free large tensors immediately

        if self.is_nested_encoder:
            output = context.transpose(1, 2).contiguous().view(B, L, H * hd)
            del context
        else:
            output = self.out_proj(context.transpose(1, 2).contiguous().view(B, L, H * hd))
            del context

        if output.shape[-1] != hidden_states.shape[-1]:
            output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1])) if output.shape[-1] < hidden_states.shape[-1] else output[..., :hidden_states.shape[-1]]

        valid_m = get_token_valid_mask(attention_mask, B, L, hidden_states.device)
        if valid_m is not None:
            output = output * valid_m.unsqueeze(-1).to(output.dtype)

        return (output, (attn_weights if output_attentions else None))


@ATTENTION_REGISTRY.register("tpa_inject")
@ATTENTION_REGISTRY.register("tp_attention_inject")
@ATTENTION_REGISTRY.register("inject_tpa")
def inject_tpa(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    **kwargs: Any,
) -> Optional[Union[TPAttention, nn.Module]]:
    """Safe injection hook for TPAttention across diverse transformer architectures."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # 1. Whole model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
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
            res = inject_tpa(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # 2. Single layer injection
    try:
        src = old_attn
        if src is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if hasattr(cand, "self"):
                        src = cand.self
                    else:
                        src = cand
                    break

        if src is None and target is not None and hasattr(target, "forward"):
            src = target

        if src is None:
            return None

        # Double-injection guard
        if src is not None:
            if isinstance(src, TPAttention):
                return None
            if "TPAttention" in src.__class__.__name__ or "TPA" in src.__class__.__name__:
                return None
            if hasattr(src, "inner") and isinstance(getattr(src, "inner"), TPAttention):
                return None
            if hasattr(src, "_wrapped_attention") and isinstance(getattr(src, "_wrapped_attention"), TPAttention):
                return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, TPAttention):
                    return None
                if "TPAttention" in cand.__class__.__name__ or "TPA" in cand.__class__.__name__:
                    return None
                if hasattr(cand, "inner") and isinstance(getattr(cand, "inner"), TPAttention):
                    return None
                if hasattr(cand, "_wrapped_attention") and isinstance(getattr(cand, "_wrapped_attention"), TPAttention):
                    return None

        # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
        if src is not None:
            c_name = src.__class__.__name__
            if "DeltaNet" in c_name or "LinearAttention" in c_name:
                return None

        if cfg is None and hasattr(src, "config"):
            cfg = src.config

        rank = kwargs.pop("rank", _cfg_get(comp_config, "tpa_rank", _cfg_get(comp_config, "rank", 8)))
        lora_rank = kwargs.pop("lora_rank", _cfg_get(comp_config, "lora_rank", 16))

        new_attn = TPAttention(
            config=cfg,
            rank=rank,
            lora_rank=lora_rank,
            pretrained_attn=src,
            **kwargs,
        )

        p = next(src.parameters(), None) if src is not None else None
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

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
        else:
            new_attn.is_nested_encoder = False
            target.attention = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_tpa failed at layer %d: %s", layer_idx, e)
        return None


tp_inject = inject_tpa
TPA = TPAttention

