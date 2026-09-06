"""
SVD Factorized Linear Attention & Kernelized Linear Attention Modules with Exponential Decay Gating.
Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

from __future__ import annotations

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

try:
    from framework.attentions.fused_svd_kernel import FusedSVDLinear, fused_svd_linear
    HAS_FUSED_KERNEL = True
except ImportError:
    HAS_FUSED_KERNEL = False

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        if (module.__class__.__name__ == "Conv1D") or (base.__class__.__name__ == "Conv1D") or (w.ndim == 2 and w.shape[0] < w.shape[1]):
            w = w.t()
    if b is not None:
        b = b.data.clone()
    return w, b


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
    """Applies Rotary Position Embedding (RoPE) with partial dimension and odd-dimension safety."""
    rotary_dim = cos.shape[-1]
    safe_rotary_dim = (rotary_dim // 2) * 2

    q_rot, q_pass = q[..., :safe_rotary_dim], q[..., safe_rotary_dim:]
    k_rot, k_pass = k[..., :safe_rotary_dim], k[..., safe_rotary_dim:]

    cos_safe = cos[..., :safe_rotary_dim]
    sin_safe = sin[..., :safe_rotary_dim]

    if cos_safe.dim() == 2 and cos_safe.shape[0] == q_rot.shape[-2]:
        cos_safe = cos_safe.unsqueeze(0).unsqueeze(1)
        sin_safe = sin_safe.unsqueeze(0).unsqueeze(1)
    else:
        while cos_safe.dim() < q_rot.dim():
            cos_safe = cos_safe.unsqueeze(unsqueeze_dim)
            sin_safe = sin_safe.unsqueeze(unsqueeze_dim)

    q_rot = (q_rot * cos_safe) + (_rotate_half(q_rot) * sin_safe)
    k_rot = (k_rot * cos_safe) + (_rotate_half(k_rot) * sin_safe)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


def compute_exponential_decay(
    decay_logits: torch.Tensor, activation: str = "softplus"
) -> torch.Tensor:
    """Computes strictly bounded, numerically stable retention factor lambda in (0, 1)."""
    if activation == "softplus":
        decay = torch.exp(-F.softplus(decay_logits))
        return torch.clamp(decay, min=1e-5, max=1.0)
    elif activation == "sigmoid":
        decay = torch.sigmoid(decay_logits)
        return torch.clamp(decay, min=1e-5, max=1.0 - 1e-5)
    else:
        raise ValueError(
            f"Unsupported decay activation '{activation}'. Must be 'softplus' or 'sigmoid'."
        )


class SVDLinear(nn.Module):
    r"""Low-Rank Factorized Linear Layer with SVD bases and trainable LoRA adapters:
    $$y = (x V^\top) U^\top + b + \frac{\alpha}{r_{\text{lora}}} (x A^\top) B^\top$$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        freeze_svd: bool = False,
        lora_rank: Optional[int] = 16,
        lora_alpha: float = 32.0,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.freeze_svd = freeze_svd
        self.lora_rank = lora_rank if lora_rank is not None else 16
        self.lora_alpha = lora_alpha
        self.use_fused_kernel = use_fused_kernel and HAS_FUSED_KERNEL
        self.use_rms_norm = use_rms_norm
        self.eps = eps

        # SVD Base Parameters
        self.V = nn.Parameter(torch.empty(rank, in_features, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(out_features, rank, dtype=dtype, device=device))

        if use_rms_norm:
            self.norm_weight = nn.Parameter(torch.ones(rank, dtype=dtype, device=device))
        else:
            self.register_parameter("norm_weight", None)

        if freeze_svd:
            self.U.requires_grad = False
            self.V.requires_grad = False

        # Trainable LoRA Residual Adapter
        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features, dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_rank, dtype=dtype, device=device))
            self.lora_scaling = lora_alpha / max(1, self.lora_rank)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_scaling = 1.0

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective dense weight matrix."""
        W = torch.matmul(self.U, self.V)
        if self.lora_A is not None and self.lora_B is not None:
            W = W + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return W

    def reset_parameters(self) -> None:
        for param in (self.U, self.V):
            orig_dtype = param.data.dtype
            if orig_dtype in (torch.float16, torch.bfloat16):
                param.data = param.data.float()
                nn.init.orthogonal_(param)
                param.data = param.data.to(orig_dtype)
            else:
                nn.init.orthogonal_(param)
        if self.norm_weight is not None:
            nn.init.ones_(self.norm_weight)
        if self.lora_A is not None and self.lora_B is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fused_kernel and HAS_FUSED_KERNEL:
            out = fused_svd_linear(
                x=x,
                V=self.V,
                U=self.U,
                norm_weight=self.norm_weight if self.use_rms_norm else None,
                bias=self.bias,
                eps=self.eps,
                use_rms_norm=self.use_rms_norm,
            )
        else:
            Vx = F.linear(x, self.V)
            if self.use_rms_norm and self.norm_weight is not None:
                norm = torch.rsqrt(Vx.pow(2).mean(-1, keepdim=True) + self.eps)
                Vx = Vx * norm * self.norm_weight
            out = F.linear(Vx, self.U, self.bias)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out

        return out

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        with torch.no_grad():
            r = min(self.rank, U_factor.size(1), V_factor.size(0))
            self.U.data.zero_()
            self.V.data.zero_()
            self.U.data[:, :r].copy_(U_factor[:, :r].to(device=self.U.device, dtype=self.U.dtype))
            self.V.data[:r, :].copy_(V_factor[:r, :].to(device=self.V.device, dtype=self.V.dtype))

    def get_orthogonality_loss(self) -> torch.Tensor:
        U_orth = torch.matmul(self.U.t(), self.U) - torch.eye(self.rank, device=self.U.device, dtype=self.U.dtype)
        V_orth = torch.matmul(self.V, self.V.t()) - torch.eye(self.rank, device=self.V.device, dtype=self.V.dtype)
        return torch.norm(U_orth, p="fro") + torch.norm(V_orth, p="fro")


@ATTENTION_REGISTRY.register("svd_linear_attn", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_linear_attention")
@ATTENTION_REGISTRY.register("svd_linear")
class SVDLinearAttention(BaseAttention):
    r"""SVD Factorized Linear Attention & Kernelized Linear Attention with Exponential Decay Gating.
    Hardened for production use across diverse transformer architectures.
    """

    supports_gqa: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        old_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        mode: str = "exact",
        freeze_svd: bool = False,
        lora_rank: Optional[int] = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        use_decay_gating: bool = False,
        decay_activation: str = "softplus",
        decay_channel_wise: bool = False,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        actual_rank = rank or _cfg_get(config, "svd_rank", _cfg_get(config, "rank", 32))
        super().__init__(config, rank=actual_rank)

        # 1. Dimensions & Architecture
        self.hidden_size: int = _cfg_get(
            config, "hidden_size", getattr(old_attn, "embed_dim", kwargs.get("hidden_size", 768))
        )
        self.num_heads: int = _cfg_get(
            config,
            "num_attention_heads",
            _cfg_get(config, "num_heads", getattr(old_attn, "num_heads", kwargs.get("num_attention_heads", 12))),
        )
        self.head_dim: int = _cfg_get(
            config, "head_dim", getattr(old_attn, "head_dim", kwargs.get("head_dim", self.hidden_size // self.num_heads))
        )

        cfg_kv = _cfg_get(
            config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None)
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        elif hasattr(old_attn, "num_key_value_heads"):
            self.num_kv_heads = int(old_attn.num_key_value_heads)
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.q_head_size: int = self.num_heads * self.head_dim
        self.all_head_size: int = self.q_head_size

        self.rank: int = actual_rank
        self.mode: str = mode.lower()
        self.dropout_rate: float = _cfg_get(
            config, "attention_probs_dropout_prob", _cfg_get(config, "dropout", 0.1)
        )
        self.eps: float = _cfg_get(config, "linear_attention_eps", 1e-6)

        # 2. Causal decoder flag detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = (
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "model_type", "").lower() in ("gpt2", "qwen", "qwen2", "llama")
                or (old_attn is not None and hasattr(old_attn, "c_attn"))
            )

        self.is_nested_encoder: bool = False
        self.lora_rank = lora_rank if lora_rank is not None else 16
        self.lora_alpha = lora_alpha

        self.use_decay_gating: bool = (
            use_decay_gating
            or _cfg_get(config, "use_decay_gating", False)
            or (self.mode in ["decay_linear", "linear_decay", "decay", "gated_linear"])
        )
        self.decay_activation: str = _cfg_get(config, "decay_activation", decay_activation)
        self.decay_channel_wise: bool = _cfg_get(config, "decay_channel_wise", decay_channel_wise)
        self.use_fused_kernel: bool = use_fused_kernel or _cfg_get(config, "use_fused_kernel", False)
        self.use_rms_norm: bool = use_rms_norm or _cfg_get(config, "use_rms_norm", False)

        # 3. Factorized projections
        self.q_proj = SVDLinear(
            self.hidden_size,
            self.q_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.k_proj = SVDLinear(
            self.hidden_size,
            self.kv_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.v_proj = SVDLinear(
            self.hidden_size,
            self.kv_head_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )
        self.out_proj = SVDLinear(
            self.q_head_size,
            self.hidden_size,
            rank=self.rank,
            freeze_svd=freeze_svd,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            use_fused_kernel=self.use_fused_kernel,
            use_rms_norm=self.use_rms_norm,
            eps=self.eps,
        )

        gate_out_dim = self.all_head_size if self.decay_channel_wise else self.num_heads
        self.decay_gate = nn.Linear(self.hidden_size, gate_out_dim, bias=True)
        self.dropout = nn.Dropout(self.dropout_rate)
        self._init_decay_gate()

        if old_attn is not None:
            self.copy_weights(old_attn)

    def _init_decay_gate(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.decay_gate.weight, a=math.sqrt(5))
            if self.decay_channel_wise:
                decay_times = torch.exp(torch.linspace(math.log(1.0), math.log(1024.0), self.num_heads))
                decay_times = decay_times.repeat_interleave(self.head_dim)
            else:
                decay_times = torch.exp(torch.linspace(math.log(1.0), math.log(1024.0), self.num_heads))
            bias_target = -torch.log(decay_times)
            self.decay_gate.bias.copy_(bias_target)

    def _factorize_and_set(
        self, target_linear: SVDLinear, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(weight.detach().cpu().float(), full_matrices=False)
            r = min(target_linear.rank, U.size(1), Vt.size(0))
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-5))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(device=weight.device, dtype=weight.dtype)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(device=weight.device, dtype=weight.dtype)
            target_linear.set_low_rank_factors(U_r, V_r)
            if bias is not None and target_linear.bias is not None:
                target_linear.bias.data.copy_(bias.to(target_linear.bias.device, target_linear.bias.dtype))

    def copy_weights(self, old_attn: nn.Module) -> None:
        """Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA."""
        with torch.no_grad():
            w_q, b_q = None, None
            w_k, b_k = None, None
            w_v, b_v = None, None
            w_o, b_o = None, None

            # Detect nested encoder container or inner self-attention
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

            # Pattern 0: PyTorch MultiheadAttention (in_proj_weight, in_proj_bias)
            if hasattr(src, "in_proj_weight") and src.in_proj_weight is not None:
                in_proj_w, in_proj_b = _unpack_weight_bias(src)
                if in_proj_w is None:
                    in_proj_w = getattr(src, "in_proj_weight", None)
                    in_proj_b = getattr(src, "in_proj_bias", None)
                if in_proj_w is not None:
                    chunks_w = in_proj_w.chunk(3, dim=0)
                    w_q, w_k, w_v = chunks_w[0], chunks_w[1], chunks_w[2]
                if in_proj_b is not None:
                    chunks_b = in_proj_b.chunk(3, dim=0)
                    b_q, b_k, b_v = chunks_b[0], chunks_b[1], chunks_b[2]
                if hasattr(src, "out_proj"):
                    w_o, b_o = _unpack_weight_bias(src.out_proj)

            # Pattern 1: GPT-2 Fused c_attn
            elif hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
                if c_attn_w is not None:
                    if c_attn_w.shape[0] == self.hidden_size and c_attn_w.shape[1] != self.hidden_size:
                        c_attn_w = c_attn_w.t()

                    q_dim = self.q_head_size
                    k_dim = self.kv_head_size if hasattr(self, "kv_head_size") else self.hidden_size
                    v_dim = self.kv_head_size if hasattr(self, "kv_head_size") else self.hidden_size

                    w_q = c_attn_w[:q_dim, :]
                    w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                    w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]

                    if c_attn_b is not None:
                        b_q = c_attn_b[:q_dim]
                        b_k = c_attn_b[q_dim : q_dim + k_dim]
                        b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]

                if hasattr(src, "c_proj"):
                    out_w, out_b = _unpack_weight_bias(src.c_proj)
                    if out_w is not None and out_w.shape[0] == self.hidden_size and out_w.shape[1] != self.hidden_size:
                        out_w = out_w.t()
                    w_o, b_o = out_w, out_b

            # Pattern 2: RoBERTa / BERT separate query, key, value
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                w_q, b_q = _unpack_weight_bias(src.query)
                w_k, b_k = _unpack_weight_bias(src.key)
                w_v, b_v = _unpack_weight_bias(src.value)

                if w_o is None:
                    for attr in ("dense", "out_proj", "o_proj"):
                        if hasattr(src, attr):
                            w_o, b_o = _unpack_weight_bias(getattr(src, attr))
                            break

            # Pattern 3: Qwen / LLaMA (q_proj, k_proj, v_proj, o_proj)
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                w_q, b_q = _unpack_weight_bias(src.q_proj)
                w_k, b_k = _unpack_weight_bias(src.k_proj)
                w_v, b_v = _unpack_weight_bias(src.v_proj)
                if hasattr(src, "o_proj"):
                    w_o, b_o = _unpack_weight_bias(src.o_proj)

            # Pattern 4: DistilBERT (q_lin, k_lin, v_lin, out_lin)
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                w_q, b_q = _unpack_weight_bias(src.q_lin)
                w_k, b_k = _unpack_weight_bias(src.k_lin)
                w_v, b_v = _unpack_weight_bias(src.v_lin)
                if hasattr(src, "out_lin"):
                    w_o, b_o = _unpack_weight_bias(src.out_lin)

            # Fallback: UniversalAdapter
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter(old_attn)
                    extracted = adapter.extract_qkv_out()
                    if extracted:
                        w_q = extracted.get("q_weight")
                        b_q = extracted.get("q_bias")
                        w_k = extracted.get("k_weight")
                        b_k = extracted.get("k_bias")
                        w_v = extracted.get("v_weight")
                        b_v = extracted.get("v_bias")
                        if w_o is None:
                            w_o = extracted.get("out_weight")
                            b_o = extracted.get("out_bias")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Verification of essential projections
            missing = []
            if w_q is None:
                missing.append("query/q_proj")
            if w_k is None:
                missing.append("key/k_proj")
            if w_v is None:
                missing.append("value/v_proj")
            if missing:
                raise RuntimeError(f"Failed to extract required projections: {missing}")

            # Frobenius norm checks
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    raise RuntimeError(f"Copied weight {name} has near-zero Frobenius norm (< 1e-4)")

            # Set Q
            self._factorize_and_set(self.q_proj, w_q, b_q)

            # Set K & V with GQA pooling if needed
            if w_k.shape[0] == self.kv_head_size:
                self._factorize_and_set(self.k_proj, w_k, b_k)
            else:
                rep = w_k.shape[0] // self.kv_head_size
                w_k_pooled = (
                    w_k.view(self.num_kv_heads, rep, self.head_dim, -1)
                    .mean(dim=1)
                    .view(self.kv_head_size, -1)
                )
                b_k_pooled = (
                    b_k.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    if b_k is not None
                    else None
                )
                self._factorize_and_set(self.k_proj, w_k_pooled, b_k_pooled)

            if w_v.shape[0] == self.kv_head_size:
                self._factorize_and_set(self.v_proj, w_v, b_v)
            else:
                rep = w_v.shape[0] // self.kv_head_size
                w_v_pooled = (
                    w_v.view(self.num_kv_heads, rep, self.head_dim, -1)
                    .mean(dim=1)
                    .view(self.kv_head_size, -1)
                )
                b_v_pooled = (
                    b_v.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                    if b_v is not None
                    else None
                )
                self._factorize_and_set(self.v_proj, w_v_pooled, b_v_pooled)

            # Set out_proj if not nested encoder
            if w_o is not None and not self.is_nested_encoder:
                if torch.norm(w_o.float(), p="fro").item() < 1e-4:
                    raise RuntimeError("Copied out_proj weight has near-zero Frobenius norm (< 1e-4)")
                self._factorize_and_set(self.out_proj, w_o, b_o)

    def _forward_decay_gated_linear_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        phi_q = F.elu(q.float()) + 1.0  # [B, H, N, D_k]
        phi_k = F.elu(k.float()) + 1.0  # [B, H, N, D_k]
        v_f = v.float()

        if mask is not None:
            if mask.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                mask_binary = (mask > -10000.0).to(torch.float32) if (mask < 0).any() else mask.to(torch.float32)
            else:
                mask_binary = mask.to(torch.float32)

            if mask_binary.dim() == 2:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            elif mask_binary.dim() == 3:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            elif mask_binary.dim() == 4:
                if mask_binary.shape[-2] == 1 and mask_binary.shape[-1] == N:
                    mask_expanded = mask_binary.permute(0, 1, 3, 2)
                elif mask_binary.shape[-2] == N and mask_binary.shape[-1] == 1:
                    mask_expanded = mask_binary
                elif mask_binary.shape[-2] == N and mask_binary.shape[-1] == N:
                    mask_expanded = mask_binary.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
                else:
                    mask_expanded = mask_binary.view(B, 1, N, 1)
            else:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            phi_k = phi_k * mask_expanded

        # Compute Data-Dependent Decay
        decay_logits = self.decay_gate(hidden_states)
        if self.decay_channel_wise:
            decay_logits = decay_logits.view(B, N, H, D_k).permute(0, 2, 1, 3)
            lambda_decay = compute_exponential_decay(decay_logits, activation=self.decay_activation).unsqueeze(-1)
        else:
            decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            lambda_decay = compute_exponential_decay(decay_logits, activation=self.decay_activation)

        # Fast parallel vectorized scan for non-streaming batch mode
        if past_state is None and not self.decay_channel_wise and N <= 2048:
            # log-space cumulative sum decay kernel
            # lambda_decay is [B, H, N, 1, 1]
            log_decay = torch.log(torch.clamp(lambda_decay.squeeze(-1).squeeze(-1), min=1e-6))  # [B, H, N]
            log_gamma = torch.cumsum(log_decay, dim=-1)  # [B, H, N]
            # Pairwise log decay matrix: log_M[t, j] = log_gamma[t] - log_gamma[j] for j <= t
            log_M = log_gamma.unsqueeze(-1) - log_gamma.unsqueeze(-2)  # [B, H, N, N]
            # Mask out future positions
            causal_bool = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool))
            M = torch.exp(torch.clamp(log_M, max=0.0)) * causal_bool.unsqueeze(0).unsqueeze(0)

            # Affinity: A = phi_q @ phi_k^T  [B, H, N, N]
            aff = torch.matmul(phi_q, phi_k.transpose(-2, -1)) * M  # [B, H, N, N]
            num = torch.matmul(aff, v_f)  # [B, H, N, D_v]
            den = aff.sum(dim=-1, keepdim=True) + self.eps  # [B, H, N, 1]
            context = (num / den).to(q.dtype)

            # Exact final recurrent state
            log_w_final = log_gamma[:, :, -1:] - log_gamma  # [B, H, N]
            w_final = torch.exp(torch.clamp(log_w_final, max=0.0)).unsqueeze(-1)  # [B, H, N, 1]
            phi_k_w = phi_k * w_final  # [B, H, N, D_k]
            S = torch.matmul(phi_k_w.transpose(-1, -2), v_f)  # [B, H, D_k, D_v]
            Z = phi_k_w.sum(dim=-2, keepdim=True).transpose(-1, -2)  # [B, H, D_k, 1]
            return context, (S, Z)

        # Sequential associative memory recurrence loop
        if past_state is not None:
            S, Z = past_state
            S = S.to(device=q.device, dtype=torch.float32)
            Z = Z.to(device=q.device, dtype=torch.float32)
        else:
            S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=torch.float32)
            Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=torch.float32)

        outputs = []
        for t in range(N):
            qt = phi_q[:, :, t : t + 1, :]
            kt = phi_k[:, :, t : t + 1, :]
            vt = v_f[:, :, t : t + 1, :]
            lam_t = lambda_decay[:, :, t]

            kt_t = kt.transpose(-1, -2)
            kv_t = torch.matmul(kt_t, vt)
            S = lam_t * S + kv_t
            Z = lam_t * Z + kt_t

            num_t = torch.matmul(qt, S)
            den_t = torch.matmul(qt, Z) + self.eps
            y_t = num_t / den_t
            outputs.append(y_t)

        context = torch.cat(outputs, dim=2).to(q.dtype)
        return context, (S, Z)

    def _forward_linear_kernel(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, H, N, D_k = q.shape
        D_v = v.shape[-1]

        phi_q = F.elu(q.float()) + 1.0
        phi_k = F.elu(k.float()) + 1.0
        v_f = v.float()

        if mask is not None:
            if mask.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                mask_binary = (mask > -10000.0).to(torch.float32) if (mask < 0).any() else mask.to(torch.float32)
            else:
                mask_binary = mask.to(torch.float32)

            if mask_binary.dim() == 2:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            elif mask_binary.dim() == 3:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            elif mask_binary.dim() == 4:
                if mask_binary.shape[-2] == 1 and mask_binary.shape[-1] == N:
                    mask_expanded = mask_binary.permute(0, 1, 3, 2)
                elif mask_binary.shape[-2] == N and mask_binary.shape[-1] == 1:
                    mask_expanded = mask_binary
                else:
                    mask_expanded = mask_binary.view(B, 1, N, 1)
            else:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            phi_k = phi_k * mask_expanded

        # Causal mode for decoders
        if self.is_decoder and N > 1:
            aff = torch.matmul(phi_q, phi_k.transpose(-2, -1))  # [B, H, N, N]
            causal_mask = torch.tril(torch.ones(N, N, device=q.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            aff = aff * causal_mask
            num = torch.matmul(aff, v_f)
            den = aff.sum(dim=-1, keepdim=True) + self.eps
            context = (num / den).to(q.dtype)
        else:
            # Associative accumulation for bidirectional encoders
            kv_state = torch.matmul(phi_k.transpose(-1, -2), v_f)  # [B, H, D_k, D_v]
            numerator = torch.matmul(phi_q, kv_state)
            k_sum = phi_k.sum(dim=-2, keepdim=True)
            denominator = torch.matmul(phi_q, k_sum.transpose(-1, -2)) + self.eps
            context = (numerator / denominator).to(q.dtype)

        return context

    def _forward_exact_softmax(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale

        if self.is_decoder and q.shape[-2] > 1:
            causal_mask = torch.triu(
                torch.full((q.shape[-2], k.shape[-2]), float("-inf"), device=scores.device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

        if mask is not None:
            scores = apply_attention_mask(scores, mask)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        probs = F.softmax(scores, dim=-1).to(q.dtype)
        probs = self.dropout(probs)
        context = torch.matmul(probs, v)
        return context, probs

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
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        query = kwargs.get("query", None)
        key = kwargs.get("key", None)
        mask = kwargs.get("mask", None)
        return_state = kwargs.get("return_state", False)
        past_state = kwargs.get("past_state", None)

        if hidden_states is None:
            hidden_states = query if query is not None else key
        if attention_mask is None:
            attention_mask = mask

        B, N, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 2. RoPE support
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 3. GQA expansion
        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        updated_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # 4. Mode dispatch
        if self.mode == "exact":
            context, attn_probs = self._forward_exact_softmax(q, k, v, eff_mask)
        elif self.use_decay_gating or self.mode in ["decay_linear", "linear_decay", "decay", "gated_linear"]:
            context, updated_state = self._forward_decay_gated_linear_kernel(
                q=q,
                k=k,
                v=v,
                hidden_states=hidden_states,
                mask=eff_mask,
                past_state=past_state,
            )
            attn_probs = None
        else:
            context = self._forward_linear_kernel(q, k, v, eff_mask)
            attn_probs = None

        # 5. Safe head_mask application
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(1)
            if attn_probs is not None:
                attn_probs = attn_probs * head_mask
            context = context * head_mask

        # 6. Head merge and output projection
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        if output.shape[-1] != hidden_states.shape[-1]:
            if output.shape[-1] < hidden_states.shape[-1]:
                output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1]))
            else:
                output = output[..., :hidden_states.shape[-1]]

        attn_out = attn_probs if output_attentions else None

        if return_state and updated_state is not None:
            return output, attn_out, updated_state

        return output, attn_out

    def step_recurrent(
        self,
        x_t: torch.Tensor,
        past_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)

        B = x_t.shape[0]
        q = self.q_proj(x_t).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_t).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_t).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        phi_q = F.elu(q.float()) + 1.0
        phi_k = F.elu(k.float()) + 1.0

        _, H, _, D_k = q.shape
        D_v = v.shape[-1]

        decay_logits = self.decay_gate(x_t)
        if self.decay_channel_wise:
            decay_logits = decay_logits.view(B, 1, H, D_k).permute(0, 2, 1, 3)
            lam_t = compute_exponential_decay(decay_logits, activation=self.decay_activation).unsqueeze(-1).squeeze(2)
        else:
            decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            lam_t = compute_exponential_decay(decay_logits, activation=self.decay_activation).squeeze(2)

        if past_state is not None:
            S_prev, Z_prev = past_state
        else:
            S_prev = torch.zeros(B, H, D_k, D_v, device=x_t.device, dtype=torch.float32)
            Z_prev = torch.zeros(B, H, D_k, 1, device=x_t.device, dtype=torch.float32)

        kt_t = phi_k.transpose(-1, -2)
        kv_t = torch.matmul(kt_t, v.float())

        S_t = lam_t * S_prev + kv_t
        Z_t = lam_t * Z_prev + kt_t

        num_t = torch.matmul(phi_q, S_t)
        den_t = torch.matmul(phi_q, Z_t) + self.eps
        y_t = num_t / den_t

        context = y_t.transpose(1, 2).contiguous().view(B, 1, self.all_head_size).to(x_t.dtype)
        if self.is_nested_encoder:
            out_t = context
        else:
            out_t = self.out_proj(context)

        if out_t.shape[-1] != x_t.shape[-1]:
            if out_t.shape[-1] < x_t.shape[-1]:
                out_t = F.pad(out_t, (0, x_t.shape[-1] - out_t.shape[-1]))
            else:
                out_t = out_t[..., :x_t.shape[-1]]

        return out_t, (S_t, Z_t)

    def get_orthogonality_loss(self) -> torch.Tensor:
        base_loss = (
            self.q_proj.get_orthogonality_loss()
            + self.k_proj.get_orthogonality_loss()
            + self.v_proj.get_orthogonality_loss()
            + self.out_proj.get_orthogonality_loss()
        )
        return base_loss


@ATTENTION_REGISTRY.register("svd_linear_attn")
@ATTENTION_REGISTRY.register("svd_linear_attention")
@ATTENTION_REGISTRY.register("svd_linear_inject")
@ATTENTION_REGISTRY.register("inject_svd_linear_attention")
def inject_svd_linear_attention(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    lora_rank: int = 16,
    rank: Optional[int] = 32,
    mode: str = "exact",
    **kwargs: Any,
) -> Optional[Any]:
    """Safely constructs and injects SVDLinearAttention into a model or transformer layer."""
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config", None)

    # 1. Whole model traversal support
    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and not hasattr(target, "forward"))
    ):
        layer_list = None
        if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer

        if layer_list is not None:
            num_injected = 0
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                res = inject_svd_linear_attention(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    rank=rank,
                    mode=mode,
                    **kwargs,
                )
                if res is not None:
                    num_injected += 1
            return target if num_injected > 0 else None

    # 2. Single layer injection
    curr_attn = old_attn
    if curr_attn is None and target is not None:
        if hasattr(target, "attention"):
            curr_attn = target.attention
        elif hasattr(target, "self_attn"):
            curr_attn = target.self_attn
        elif hasattr(target, "attn"):
            curr_attn = target.attn

    if curr_attn is None:
        return None

    try:
        new_attn = SVDLinearAttention(
            config=cfg,
            old_attn=curr_attn,
            rank=rank,
            mode=mode,
            lora_rank=lora_rank,
            **kwargs,
        )
    except Exception as e:
        logger.warning(f"Failed to instantiate SVDLinearAttention: {e}")
        return None

    if target is not None:
        if hasattr(target, "attention"):
            target.attention = new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn

    return new_attn
