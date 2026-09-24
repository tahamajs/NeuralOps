r"""
Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.
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
    """Safely retrieve configuration value from dict, object, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class FactorizedLinear(nn.Module):
    def __init__(self, down: nn.Linear, up: nn.Linear) -> None:
        super().__init__()
        self.down = down
        self.up = up

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.up, "bias", None)

    @property
    def in_features(self) -> int:
        return self.down.in_features

    @property
    def out_features(self) -> int:
        return self.up.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


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
    head_dim = q.shape[-1]

    if rotary_dim > head_dim:
        cos = cos[..., :head_dim]
        sin = sin[..., :head_dim]
        rotary_dim = head_dim

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


class FactorizedLoRAUp(nn.Module):
    """Up-projection with frozen base weight and trainable LoRA adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
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
            self.scaling = lora_alpha / lora_rank
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.scaling = 1.0
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out


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
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register("soa_module", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("soa_attn")
@ATTENTION_REGISTRY.register("soa_fused_module")
@ATTENTION_REGISTRY.register("soafusedattention")
@ATTENTION_REGISTRY.register("soa_fused_attn")
class SOAAttention(BaseAttention):
    r"""Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.

    Factorizes dense Q, K, and V linear projections into low-rank bottleneck pairs
    initialized via truncated SVD from pretrained transformer weights.
    """

    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        model_config: Optional[Any] = None,
        rank: Optional[int] = None,
        k_rank: int = 64,
        dropout: float = 0.1,
        pretrained_attn: Optional[Any] = None,
        old_attn: Optional[Any] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or model_config or kwargs.get("config", None)
        src_attn = pretrained_attn if pretrained_attn is not None else (old_attn if old_attn is not None else kwargs.get("source_attn", None))
        super().__init__(cfg, rank=k_rank, **kwargs)

        # 1. Architecture detection & dimensions
        self.hidden_size: int = getattr(
            cfg, "hidden_size", getattr(cfg, "dim", kwargs.get("hidden_size", kwargs.get("dim", 768)))
        ) if cfg is not None else kwargs.get("hidden_size", kwargs.get("dim", 768))

        self.num_heads: int = getattr(
            cfg,
            "num_attention_heads",
            getattr(cfg, "n_heads", getattr(cfg, "num_heads", kwargs.get("num_attention_heads", kwargs.get("num_heads", 12)))),
        ) if cfg is not None else kwargs.get("num_attention_heads", kwargs.get("num_heads", 12))
        self.num_heads = int(self.num_heads or 12)

        # Probe head_dim from source attention if available
        head_dim = kwargs.get("head_dim", None)
        if not head_dim and src_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                mod = getattr(src_attn, attr, None)
                if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
                    if mod.weight.ndim == 2:
                        head_dim = mod.weight.shape[0] // max(1, self.num_heads)
                        break
            if not head_dim:
                head_dim = getattr(src_attn, "head_dim", None)
        if not head_dim and cfg is not None:
            head_dim = getattr(cfg, "head_dim", None)
        if not head_dim or head_dim <= 0:
            head_dim = self.hidden_size // max(1, self.num_heads)
        self.head_dim = int(head_dim)

        cfg_kv = (
            getattr(cfg, "num_key_value_heads", getattr(cfg, "num_kv_heads", kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", None))))
            if cfg is not None
            else kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", None))
        )
        self.num_kv_heads: int = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_key_value_groups: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.kv_repeat: int = self.num_key_value_groups

        self.q_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # Adaptive k_rank
        comp_config = getattr(cfg, "compression_config", None)
        if k_rank is None or k_rank == 64:
            effective_rank = getattr(comp_config, "k_rank", getattr(comp_config, "soa_rank", rank if rank is not None else max(128, self.hidden_size // 4)))
        else:
            effective_rank = min(k_rank, self.hidden_size)
        self.k_rank: int = effective_rank
        self.rank = effective_rank
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        # Decoder / Causal flag
        _mt = str(getattr(cfg, "model_type", "") or "").lower() if cfg else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                             "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        is_dec = (
            bool(_cfg_get(cfg, "is_decoder", False))
            or bool(_cfg_get(cfg, "is_causal", False))
            or any(_mt.startswith(p) for p in _decoder_prefixes)
            or (src_attn is not None and hasattr(src_attn, "c_attn"))
            or (src_attn is not None and getattr(src_attn, "is_causal", False))
            or (src_attn is not None and getattr(src_attn, "is_decoder", False))
            or bool(kwargs.get("is_causal", False))
            or bool(kwargs.get("is_decoder", False))
        )
        self.is_decoder: bool = is_dec
        self.is_causal: bool = is_dec
        self.is_nested_encoder: bool = False

        # 2. Factorized SVD linear projections for Q, K, V
        self.q_down = nn.Linear(self.hidden_size, self.k_rank, bias=False)
        self.q_up = FactorizedLoRAUp(self.k_rank, self.q_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)

        self.k_down = nn.Linear(self.hidden_size, self.k_rank, bias=False)
        self.k_up = FactorizedLoRAUp(self.k_rank, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)

        self.v_down = nn.Linear(self.hidden_size, self.k_rank, bias=False)
        self.v_up = FactorizedLoRAUp(self.k_rank, self.kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)

        # Output projection with trainable LoRA adapter
        self.out_proj = LoRALinear(
            self.q_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        dropout_p = (
            getattr(cfg, "dropout", getattr(cfg, "attention_probs_dropout_prob", dropout))
            if cfg is not None
            else dropout
        )
        self.dropout = nn.Dropout(dropout_p)

        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and src_attn is not None:
            target_dtype = next(src_attn.parameters(), torch.tensor([])).dtype
            if not isinstance(target_dtype, torch.dtype):
                target_dtype = torch.float32

        # 3. Pretrained weight transfer
        if src_attn is not None:
            self.copy_weights(src_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    @property
    def q_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.q_down, self.q_up)

    @property
    def k_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.k_down, self.k_up)

    @property
    def v_proj(self) -> FactorizedLinear:
        return FactorizedLinear(self.v_down, self.v_up)

    @property
    def query(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorizedLinear:
        return self.v_proj

    def copy_weights(self, old_attn: nn.Module) -> None:
        """Robust weight extraction and SVD factorization across all architectures."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Check nested encoder container
            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            has_out_lin = hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj") or hasattr(old_attn, "dense")
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not has_out_lin
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
                self.is_decoder = True
                self.is_causal = True

                c_attn_mod = src.c_attn
                base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
                c_attn_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None)).detach()
                c_attn_b = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_attn_b = c_attn_b.detach() if c_attn_b is not None else None

                c_cls = getattr(base_c, "__class__", type(base_c)).__name__
                if ("Conv1D" in c_cls) or hasattr(base_c, "nf") or ("Conv1D" in getattr(c_attn_mod, "__class__", type(c_attn_mod)).__name__) or hasattr(c_attn_mod, "nf") or (c_attn_w.dim() == 2 and c_attn_w.shape[0] < c_attn_w.shape[1]):
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

                c_proj_mod = src.c_proj
                base_p = getattr(c_proj_mod, "base_layer", c_proj_mod)
                out_w = getattr(base_p, "weight", getattr(c_proj_mod, "weight", None)).detach()
                b_o = getattr(base_p, "bias", getattr(c_proj_mod, "bias", None))
                b_o = b_o.detach() if b_o is not None else None

                p_cls = getattr(base_p, "__class__", type(base_p)).__name__
                if ("Conv1D" in p_cls) or hasattr(base_p, "nf") or ("Conv1D" in getattr(c_proj_mod, "__class__", type(c_proj_mod)).__name__) or hasattr(c_proj_mod, "nf") or (out_w.dim() == 2 and out_w.shape[0] < out_w.shape[1]):
                    out_w = out_w.t().contiguous()
                w_o = out_w

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
                except Exception:
                    pass

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
                    f"SOAAttention: Failed to extract weights: {', '.join(missing)} from {type(old_attn)}"
                )

            # SVD Factorization helper
            def _factorize_svd(W: torch.Tensor, down: nn.Linear, up: nn.Linear, bias: Optional[torch.Tensor]) -> None:
                W_f = W.float()
                U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
                S = S.clamp(min=1e-6)
                r = min(self.k_rank, U.size(1), Vt.size(0))

                down.weight.zero_()
                up.weight.zero_()
                sqrt_s = torch.sqrt(S[:r])
                down.weight[:r, :].copy_((sqrt_s.unsqueeze(1) * Vt[:r, :]).to(down.weight.dtype))
                up.weight[:, :r].copy_((U[:, :r] * sqrt_s).to(up.weight.dtype))

                if bias is not None and up.bias is not None:
                    up.bias.copy_(bias.to(up.bias.dtype))
                elif up.bias is not None:
                    up.bias.zero_()

                # Frobenius check
                rec = up.weight @ down.weight
                f_norm = float(torch.linalg.norm(rec))
                if f_norm < 1e-4:
                    logger.warning(f"SOAAttention: SVD reconstructed weight norm is near zero: {f_norm}")

            _factorize_svd(w_q, self.q_down, self.q_up, b_q)
            _factorize_svd(w_k, self.k_down, self.k_up, b_k)
            _factorize_svd(w_v, self.v_down, self.v_up, b_v)

            if w_o is not None and not self.is_nested_encoder:
                self.out_proj.weight.copy_(w_o.to(self.out_proj.weight.dtype))
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(b_o.to(self.out_proj.bias.dtype))
                f_norm_o = float(torch.linalg.norm(self.out_proj.weight))
                if f_norm_o < 1e-4:
                    logger.warning(f"SOAAttention: Out projection norm is near zero: {f_norm_o}")
            elif self.is_nested_encoder:
                if hasattr(self.out_proj, "weight") and self.out_proj.weight.ndim == 2:
                    nn.init.eye_(self.out_proj.weight)
                if hasattr(self.out_proj, "bias") and self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)

            # Freeze base weights
            for mod in [self.q_down, self.k_down, self.v_down]:
                mod.weight.requires_grad = False
                if getattr(mod, "bias", None) is not None:
                    mod.bias.requires_grad = False
            for mod in [self.q_up, self.k_up, self.v_up]:
                if hasattr(mod, "freeze_base_weights"):
                    mod.freeze_base_weights()
                else:
                    mod.weight.requires_grad = False
                    if getattr(mod, "bias", None) is not None:
                        mod.bias.requires_grad = False
                    if getattr(mod, "lora_A", None) is not None:
                        mod.lora_A.requires_grad = True
                    if getattr(mod, "lora_B", None) is not None:
                        mod.lora_B.requires_grad = True
            if hasattr(self.out_proj, "freeze_base_weights"):
                self.out_proj.freeze_base_weights()
            else:
                self.out_proj.weight.requires_grad = False
                if getattr(self.out_proj, "bias", None) is not None:
                    self.out_proj.bias.requires_grad = False
                if getattr(self.out_proj, "lora_A", None) is not None:
                    self.out_proj.lora_A.requires_grad = True
                if getattr(self.out_proj, "lora_B", None) is not None:
                    self.out_proj.lora_B.requires_grad = True

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "SOAAttention":
        return cls(model_config=model_config, pretrained_attn=old_attn, **kwargs)

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
        B, N, _ = hidden_states.shape

        # Factorized projections
        z_q = self.q_down(hidden_states)
        z_k = self.k_down(hidden_states)
        z_v = self.v_down(hidden_states)

        q = self.q_up(z_q)
        k = self.k_up(z_k)
        v = self.v_up(z_v)

        H = self.num_heads
        H_kv = self.num_kv_heads
        D = self.head_dim

        q = q.view(B, N, H, D).transpose(1, 2).contiguous()
        k = k.view(B, N, H_kv, D).transpose(1, 2).contiguous()
        v = v.view(B, N, H_kv, D).transpose(1, 2).contiguous()

        # Partial Rotary Position Embeddings (RoPE)
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, tuple) and len(pos_emb) == 2:
            cos, sin = pos_emb
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Grouped-Query Attention expansion
        if H_kv != H:
            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

        scale = 1.0 / math.sqrt(D)
        # Memory-efficient: use SDPA to avoid OOM on large seqs

        try:

            _qs = q.half() if q.dtype == torch.float32 else q

            _ks = k.half() if k.dtype == torch.float32 else k

            _vs = v.half() if v.dtype == torch.float32 else v

            _sdpa_out = torch.nn.functional.scaled_dot_product_attention(_qs, _ks, _vs, is_causal=getattr(self, 'is_decoder', False) and N > 1, dropout_p=self.dropout.p if self.training else 0.0)

            _sdpa_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(B, N, self.all_head_size)

            _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)

            return (self.out_proj(_sdpa_out) if not getattr(self, 'is_nested_encoder', False) else _sdpa_out), None

        except Exception:

            pass

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        # Causal mask for decoders
        is_causal = (self.is_decoder or getattr(self, "is_causal", False) or kwargs.get("is_causal", False)) and (encoder_hidden_states is None)
        if is_causal and N > 1:
            causal_mask = torch.triu(
                torch.full((N, N), float("-inf"), device=scores.device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

        # Standard Attention Mask
        mask = attention_mask if attention_mask is not None else kwargs.get("attn_mask", None)
        if mask is not None:
            if mask.dim() == 4 and mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -10000.0)
            else:
                scores = apply_attention_mask(scores, mask)

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
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)

        if self.is_nested_encoder:
            return (context, attn_weights if output_attentions else None)

        output = self.out_proj(context)
        return (output, attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("soa_attn_inject")
@ATTENTION_REGISTRY.register("soa_inject")
@ATTENTION_REGISTRY.register("soa_module_inject")
@ATTENTION_REGISTRY.register("inject_soa_attn")
@ATTENTION_REGISTRY.register("inject_soa")
@ATTENTION_REGISTRY.register("inject_soa_module")
def inject_soa_attn(
    model: Optional[nn.Module] = None,
    layer: Optional[nn.Module] = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    target_layer_idx: Optional[int] = None,
    **kwargs: Any,
) -> Optional[Union[SOAAttention, nn.Module]]:
    """Safely replace attention module with hardened SOAAttention across diverse transformer architectures."""
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
            if target_layer_idx is not None and idx != target_layer_idx:
                continue
            res = inject_soa_attn(
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
        src_attn = old_attn
        if src_attn is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if hasattr(cand, "self"):
                        src_attn = cand.self
                    else:
                        src_attn = cand
                    break

        if src_attn is None and target is not None and hasattr(target, "forward"):
            src_attn = target

        if src_attn is None:
            return None

        # Double-injection guard
        if src_attn is not None:
            if isinstance(src_attn, SOAAttention):
                return None
            if "SOA" in src_attn.__class__.__name__:
                return None
            if hasattr(src_attn, "inner") and isinstance(getattr(src_attn, "inner"), SOAAttention):
                return None
            if hasattr(src_attn, "_wrapped_attention") and isinstance(getattr(src_attn, "_wrapped_attention"), SOAAttention):
                return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, SOAAttention):
                    return None
                if "SOA" in cand.__class__.__name__:
                    return None
                if hasattr(cand, "inner") and isinstance(getattr(cand, "inner"), SOAAttention):
                    return None
                if hasattr(cand, "_wrapped_attention") and isinstance(getattr(cand, "_wrapped_attention"), SOAAttention):
                    return None

        # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
        if src_attn is not None:
            c_name = src_attn.__class__.__name__
            if "DeltaNet" in c_name or "LinearAttention" in c_name:
                return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        k_rank = getattr(comp_config, "soa_rank", getattr(comp_config, "rank", kwargs.pop("k_rank", 128)))
        dropout = getattr(comp_config, "dropout", kwargs.pop("dropout", 0.1))

        new_attn = SOAAttention(
            config=cfg,
            k_rank=k_rank,
            dropout=dropout,
            pretrained_attn=src_attn,
            **kwargs,
        )

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                else:
                    target.attention = new_attn
                return new_attn
            for attr in ("self_attn", "attn", "self"):
                if hasattr(target, attr):
                    setattr(target, attr, new_attn)
                    return new_attn
            target.attention = new_attn
            return new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_soa_attn failed at layer %d: %s", layer_idx, e)
        return None


inject_soa = inject_soa_attn
inject_soa_attention = inject_soa_attn
inject_soa_module = inject_soa_attn
inject_soafusedattention = inject_soa_attn
inject_soa_fused_module = inject_soa_attn
inject_soa_fused_attn = inject_soa_attn
inject_soa_fused = inject_soa_attn
SOAFusedAttention = SOAAttention