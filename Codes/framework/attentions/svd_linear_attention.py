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
from framework.core.triton_kernels import triton_dispatch

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
        cls_name = getattr(module, "__class__", type(module)).__name__
        base_cls_name = getattr(base, "__class__", type(base)).__name__
        if "Conv1D" in cls_name or "Conv1D" in base_cls_name or hasattr(module, "nf") or hasattr(base, "nf"):
            w = w.t().contiguous()
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
    if safe_rotary_dim == 0:
        return q, k

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

    def get_weight(self) -> torch.Tensor:
        """Reconstructed effective dense weight matrix without caching or materializing on property access."""
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

    triton_kernel: str = "low_rank"

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
        elif not self.use_rms_norm:
            out = triton_dispatch(self.triton_kernel, x, self.U, self.V, bias=self.bias)
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
            if r < self.rank:
                logger.debug(
                    "SVDLinear: factor rank %d is less than allocated rank %d. Zeroing remaining components.",
                    r,
                    self.rank,
                )
            out_d = min(self.out_features, U_factor.size(0))
            in_d = min(self.in_features, V_factor.size(1))
            self.U.data[:out_d, :].zero_()
            self.V.data[:, :in_d].zero_()
            self.U.data[:out_d, :r].copy_(U_factor[:out_d, :r].to(device=self.U.device, dtype=self.U.dtype))
            self.V.data[:r, :in_d].copy_(V_factor[:r, :in_d].to(device=self.V.device, dtype=self.V.dtype))

    def get_orthogonality_loss(self) -> torch.Tensor:
        U_orth = torch.matmul(self.U.t().contiguous(), self.U) - torch.eye(self.rank, device=self.U.device, dtype=self.U.dtype)
        V_orth = torch.matmul(self.V, self.V.t().contiguous()) - torch.eye(self.rank, device=self.V.device, dtype=self.V.dtype)
        return torch.norm(U_orth, p="fro") + torch.norm(V_orth, p="fro")


@ATTENTION_REGISTRY.register("svd_linear_attn", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_linear_attention")
@ATTENTION_REGISTRY.register("svd_linear")
class SVDLinearAttention(BaseAttention):
    r"""SVD Factorized Linear Attention & Kernelized Linear Attention with Exponential Decay Gating.
    Hardened for production use across diverse transformer architectures.
    """

    supports_gqa: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "linear_attn"

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
        actual_rank = rank or _cfg_get(config, "svd_rank", _cfg_get(config, "rank", 192))
        super().__init__(config, rank=actual_rank)

        # 1. Dimensions & Architecture
        # Detect from old_attn if available and not explicitly provided
        detected_hidden = None
        detected_heads = None
        detected_head_dim = None
        if old_attn is not None:
            for attr in ("all_head_size", "embed_dim", "hidden_size", "d_model"):
                if hasattr(old_attn, attr):
                    detected_hidden = getattr(old_attn, attr)
                    break
            if detected_hidden is None:
                # Inspect query/q_proj/q_lin/c_attn weights
                for proj_name in ("query", "q_proj", "q_lin"):
                    if hasattr(old_attn, proj_name):
                        proj = getattr(old_attn, proj_name)
                        if hasattr(proj, "in_features"):
                            detected_hidden = proj.in_features
                            break
                        elif hasattr(proj, "weight"):
                            detected_hidden = proj.weight.shape[-1]
                            break
            for attr in ("num_attention_heads", "num_heads", "n_head", "n_heads"):
                if hasattr(old_attn, attr):
                    detected_heads = getattr(old_attn, attr)
                    break
            for attr in ("attention_head_size", "head_dim", "d_head"):
                if hasattr(old_attn, attr):
                    detected_head_dim = getattr(old_attn, attr)
                    break

        self.hidden_size: int = _cfg_get(
            config,
            "hidden_size",
            kwargs.get("hidden_size", kwargs.get("embed_dim", detected_hidden or 768)),
        )
        self.num_heads: int = _cfg_get(
            config,
            "num_attention_heads",
            _cfg_get(
                config,
                "num_heads",
                kwargs.get("num_attention_heads", kwargs.get("num_heads", detected_heads or 12)),
            ),
        )
        self.head_dim: int = _cfg_get(
            config,
            "head_dim",
            kwargs.get("head_dim", detected_head_dim or (self.hidden_size // self.num_heads)),
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
        self.triton_kwargs = {"phi": "elu+1", "eps": self.eps}

        # 2. Causal decoder flag detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(_cfg_get(config, "model_type", "") or "").lower()
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or any(_mt.startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "falcon", "bloom", "opt"))
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
            dev = weight.device
            try:
                U, S, Vt = torch.linalg.svd(weight.detach().float(), full_matrices=False)
            except Exception:
                U, S, Vt = torch.linalg.svd(weight.detach().cpu().float(), full_matrices=False)
            r = min(target_linear.rank, U.size(1), Vt.size(0))
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-5))
            U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(device=dev, dtype=weight.dtype)
            V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(device=dev, dtype=weight.dtype)
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

            # Nested encoder container handling
            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                src = old_attn.self
                self.is_nested_encoder = True
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
                        c_attn_w = c_attn_w.t().contiguous()

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
                        out_w = out_w.t().contiguous()
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
            if w_q is None or w_k is None or w_v is None or (w_o is None and not self.is_nested_encoder):
                try:
                    adapter = UniversalAdapter()
                    if w_q is None:
                        w_q, b_q = adapter.get_weight_bias(old_attn, "q")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(old_attn, "k")
                    if w_v is None:
                        w_v, b_v = adapter.get_weight_bias(old_attn, "v")
                    if w_o is None and not self.is_nested_encoder:
                        w_o, b_o = adapter.get_weight_bias(old_attn, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            # Verification of essential projections - raise RuntimeError on missing projections
            missing = []
            if w_q is None:
                missing.append("query/q_proj")
            if w_k is None:
                missing.append("key/k_proj")
            if w_v is None:
                missing.append("value/v_proj")
            if missing:
                raise RuntimeError(
                    f"SVDLinearAttention: failed to extract required projections: {missing} from {type(old_attn).__name__}."
                )

            # Frobenius norm checks with None guard and raise on near-zero norm
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if w is None:
                    continue
                norm_val = torch.norm(w.float(), p="fro").item()
                if norm_val < 1e-4:
                    raise RuntimeError(
                        f"SVDLinearAttention: copied weight {name} has near-zero Frobenius norm ({norm_val:.6e} < 1e-4)."
                    )

            # Set Q
            self._factorize_and_set(self.q_proj, w_q, b_q)

            # Set K & V with GQA pooling or expansion if needed
            def _adapt_to_kv_size(
                weight: torch.Tensor,
                bias: Optional[torch.Tensor],
                mode: str = "slice",
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
                w = weight
                b = bias
                if w.shape[0] == self.kv_head_size:
                    return w, b
                if w.shape[0] > self.kv_head_size and w.shape[0] % self.kv_head_size == 0:
                    rep = w.shape[0] // self.kv_head_size
                    if mode == "slice":
                        # Group by num_kv_heads and slice the first head per group
                        w = w.contiguous().reshape(self.num_kv_heads, rep, -1)[:, 0, :].contiguous()
                        if b is not None:
                            b = b.contiguous().reshape(self.num_kv_heads, rep, -1)[:, 0, :].contiguous().reshape(-1)
                    elif mode == "first":
                        w = w[:self.kv_head_size, :].contiguous()
                        if b is not None:
                            b = b[:self.kv_head_size].contiguous()
                    else:  # "mean"
                        w = w.contiguous().reshape(self.num_kv_heads, rep, -1).mean(dim=1)
                        if b is not None:
                            b = b.contiguous().reshape(self.num_kv_heads, rep, -1).mean(dim=1).reshape(-1)
                elif w.shape[0] < self.kv_head_size and self.kv_head_size % w.shape[0] == 0:
                    rep = self.kv_head_size // w.shape[0]
                    w = w.repeat(rep, 1)
                    if b is not None:
                        b = b.repeat(rep)
                else:
                    padded_w = torch.zeros(self.kv_head_size, w.shape[1], device=w.device, dtype=w.dtype)
                    min_rows = min(self.kv_head_size, w.shape[0])
                    padded_w[:min_rows, :] = w[:min_rows, :]
                    w = padded_w
                    if b is not None:
                        padded_b = torch.zeros(self.kv_head_size, device=b.device, dtype=b.dtype)
                        padded_b[:min_rows] = b[:min_rows]
                        b = padded_b
                return w, b

            w_k_adapt, b_k_adapt = _adapt_to_kv_size(w_k, b_k, mode="slice")
            self._factorize_and_set(self.k_proj, w_k_adapt, b_k_adapt)

            w_v_adapt, b_v_adapt = _adapt_to_kv_size(w_v, b_v, mode="slice")
            self._factorize_and_set(self.v_proj, w_v_adapt, b_v_adapt)

            # Set out_proj if not nested encoder
            if w_o is not None and not self.is_nested_encoder:
                norm_o = torch.norm(w_o.float(), p="fro").item()
                if norm_o < 1e-4:
                    raise RuntimeError(
                        f"SVDLinearAttention: copied out_proj weight has near-zero Frobenius norm ({norm_o:.6e} < 1e-4)."
                    )
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

        # Dynamic compute dtype: BF16 if q is BF16, else FP32
        compute_dtype = torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float32

        phi_q = (F.elu(q.to(compute_dtype)) + 1.0)  # [B, H, N, D_k]
        phi_k = (F.elu(k.to(compute_dtype)) + 1.0)  # [B, H, N, D_k]
        v_f = v.to(compute_dtype)

        if mask is not None:
            if mask.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                mask_binary = (mask > -10000.0).to(compute_dtype) if (mask < 0).any() else mask.to(compute_dtype)
            else:
                mask_binary = mask.to(compute_dtype)

            if mask_binary.dim() == 2:
                mask_expanded = mask_binary.view(B, 1, N, 1)
            elif mask_binary.dim() == 3:
                # [B, H, N] -> [B, H, N, 1] or [B, N, 1] -> [B, 1, N, 1]
                if mask_binary.shape[1] == H:
                    mask_expanded = mask_binary.unsqueeze(-1)
                else:
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
            lambda_decay = compute_exponential_decay(decay_logits, activation=self.decay_activation).unsqueeze(-1).to(compute_dtype)
        else:
            decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            lambda_decay = compute_exponential_decay(decay_logits, activation=self.decay_activation).to(compute_dtype)

        # Ensure unobserved padding tokens do not decay memory state (lambda = 1.0)
        if mask is not None:
            mask_decay = mask_expanded
            while mask_decay.dim() < lambda_decay.dim():
                mask_decay = mask_decay.unsqueeze(-1)
            lambda_decay = torch.where(mask_decay > 0, lambda_decay, torch.ones_like(lambda_decay))

        # O(N) chunk-wise cumulative decay scan for non-streaming batch mode
        if past_state is None and not self.decay_channel_wise:
            chunk_size = 64
            num_chunks = (N + chunk_size - 1) // chunk_size
            S_chunk = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
            Z_chunk = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)
            outputs = []

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)
                qc = phi_q[:, :, start:end, :]
                kc = phi_k[:, :, start:end, :]
                vc = v_f[:, :, start:end, :]
                lam_c = lambda_decay[:, :, start:end].squeeze(-1).squeeze(-1)  # [B, H, Lc]
                Lc = end - start

                # Cumulative decay within chunk
                log_lam_c = torch.log(torch.clamp(lam_c, min=1e-6))
                log_gamma_c = torch.cumsum(log_lam_c, dim=-1)  # [B, H, Lc]

                # Intra-chunk decay matrix: M_intra[i, j] = exp(log_gamma_c[i] - log_gamma_c[j]) for j <= i
                log_M_intra = log_gamma_c.unsqueeze(-1) - log_gamma_c.unsqueeze(-2)  # [B, H, Lc, Lc]
                causal_bool = torch.tril(torch.ones(Lc, Lc, dtype=torch.bool, device=qc.device))
                M_intra = torch.exp(torch.clamp(log_M_intra, max=0.0)) * causal_bool.unsqueeze(0).unsqueeze(0)

                aff_intra = torch.matmul(qc, kc.transpose(-1, -2).contiguous()) * M_intra  # [B, H, Lc, Lc]
                num_intra = torch.matmul(aff_intra, vc)
                den_intra = aff_intra.sum(dim=-1, keepdim=True)

                # Inter-chunk contribution from preceding chunks
                decay_from_past = torch.exp(torch.clamp(log_gamma_c, max=0.0)).unsqueeze(-1)  # [B, H, Lc, 1]
                num_inter = torch.matmul(qc, S_chunk) * decay_from_past
                den_inter = torch.matmul(qc, Z_chunk) * decay_from_past

                out_c = (num_intra + num_inter) / (den_intra + den_inter + self.eps)
                outputs.append(out_c)

                # Update running state to end of this chunk
                total_chunk_decay = torch.exp(torch.clamp(log_gamma_c[:, :, -1:], max=0.0)).unsqueeze(-1)  # [B, H, 1, 1]
                log_w_to_end = log_gamma_c[:, :, -1:] - log_gamma_c  # [B, H, Lc]
                w_to_end = torch.exp(torch.clamp(log_w_to_end, max=0.0)).unsqueeze(-1)  # [B, H, Lc, 1]
                kc_w = kc * w_to_end

                S_chunk = total_chunk_decay * S_chunk + torch.matmul(kc_w.transpose(-1, -2).contiguous(), vc)
                Z_chunk = total_chunk_decay * Z_chunk + kc_w.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()

            context = torch.cat(outputs, dim=2).to(q.dtype)
            return context, (S_chunk, Z_chunk)

        # Sequential associative memory recurrence loop
        if past_state is not None:
            S, Z = past_state
            S = S.to(device=q.device, dtype=compute_dtype)
            Z = Z.to(device=q.device, dtype=compute_dtype)
        else:
            S = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=compute_dtype)
            Z = torch.zeros(B, H, D_k, 1, device=q.device, dtype=compute_dtype)

        outputs = []
        for t in range(N):
            qt = phi_q[:, :, t : t + 1, :]
            kt = phi_k[:, :, t : t + 1, :]
            vt = v_f[:, :, t : t + 1, :]
            lam_t = lambda_decay[:, :, t]

            kt_t = kt.transpose(-1, -2).contiguous()
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
        if mask is None:
            return triton_dispatch(
                self.triton_kernel,
                q, k, v,
                is_causal=self.is_decoder,
                **self.triton_kwargs,
            )

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

        # Causal mode for decoders (O(N) chunk-wise accumulation, chunk_size=64)
        if self.is_decoder and N > 1:
            chunk_size = 64
            num_chunks = (N + chunk_size - 1) // chunk_size
            S_chunk = torch.zeros(B, H, D_k, D_v, device=q.device, dtype=torch.float32)
            Z_chunk = torch.zeros(B, H, D_k, 1, device=q.device, dtype=torch.float32)
            outputs = []

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, N)
                qc = phi_q[:, :, start:end, :]  # [B, H, Lc, D_k]
                kc = phi_k[:, :, start:end, :]  # [B, H, Lc, D_k]
                vc = v_f[:, :, start:end, :]    # [B, H, Lc, D_v]
                Lc = end - start

                # Intra-chunk causal affinity: [B, H, Lc, Lc]
                aff_intra = torch.matmul(qc, kc.transpose(-1, -2).contiguous())
                causal_intra = torch.tril(torch.ones(Lc, Lc, device=q.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
                aff_intra = aff_intra * causal_intra
                num_intra = torch.matmul(aff_intra, vc)
                den_intra = aff_intra.sum(dim=-1, keepdim=True)

                # Inter-chunk contribution from preceding chunks
                num_inter = torch.matmul(qc, S_chunk)
                den_inter = torch.matmul(qc, Z_chunk)

                out_c = (num_intra + num_inter) / (den_intra + den_inter + self.eps)
                outputs.append(out_c)

                # Update running state
                S_chunk = S_chunk + torch.matmul(kc.transpose(-1, -2).contiguous(), vc)
                Z_chunk = Z_chunk + kc.sum(dim=-2, keepdim=True).transpose(-1, -2).contiguous()

            context = torch.cat(outputs, dim=2).to(q.dtype)
        else:
            # Associative accumulation for bidirectional encoders
            kv_state = torch.matmul(phi_k.transpose(-1, -2).contiguous(), v_f)  # [B, H, D_k, D_v]
            numerator = torch.matmul(phi_q, kv_state)
            k_sum = phi_k.sum(dim=-2, keepdim=True)
            denominator = torch.matmul(phi_q, k_sum.transpose(-1, -2).contiguous()) + self.eps
            context = (numerator / denominator).to(q.dtype)

        return context

    def _forward_exact_softmax(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        is_cross: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        is_causal_attn = self.is_decoder and not is_cross and q.shape[-2] > 1

        # Use PyTorch SDPA when output_attentions is False and mask format is directly compatible
        if not output_attentions:
            try:
                sdpa_mask = None
                can_use_sdpa = False
                if mask is None:
                    can_use_sdpa = True
                elif mask.dim() == 2 and not is_causal_attn:
                    sdpa_mask = mask.bool().view(mask.shape[0], 1, 1, mask.shape[1])
                    can_use_sdpa = True
                elif mask.dim() == 4 and not is_causal_attn:
                    sdpa_mask = mask.bool() if mask.dtype == torch.bool else mask
                    can_use_sdpa = True

                if can_use_sdpa:
                    drop_p = self.dropout.p if self.training else 0.0
                    context = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=sdpa_mask,
                        dropout_p=drop_p,
                        is_causal=is_causal_attn if sdpa_mask is None else False,
                    )
                    return context, None
            except Exception:
                pass  # Fall back to explicit softmax

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2).contiguous()) * scale

        if is_causal_attn:
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

        if return_state and self.mode not in ("decay_linear", "linear_decay", "decay", "gated_linear") and not self.use_decay_gating:
            raise ValueError(
                f"return_state=True is only supported in decay_linear/gated_linear modes, got mode='{self.mode}'"
            )

        if hidden_states is None:
            hidden_states = query if query is not None else key
        if attention_mask is None:
            attention_mask = mask

        B, N, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Project Q, K, V
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

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
            context, attn_probs = self._forward_exact_softmax(
                q, k, v, eff_mask, is_cross=is_cross, output_attentions=bool(output_attentions)
            )
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
                head_mask = head_mask.unsqueeze(-1)
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
        q = self.q_proj(x_t).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_proj(x_t).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_proj(x_t).view(B, 1, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        phi_q = F.elu(q.float()) + 1.0
        phi_k = F.elu(k.float()) + 1.0

        _, H, _, D_k = q.shape
        D_v = v.shape[-1]

        if self.use_decay_gating:
            decay_logits = self.decay_gate(x_t)
            if self.decay_channel_wise:
                decay_logits = decay_logits.view(B, 1, H, D_k).permute(0, 2, 1, 3)
                lam_t = compute_exponential_decay(decay_logits, activation=self.decay_activation).unsqueeze(-1).squeeze(2)
            else:
                decay_logits = decay_logits.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
                lam_t = compute_exponential_decay(decay_logits, activation=self.decay_activation).squeeze(2)
        else:
            lam_t = torch.ones(B, H, 1, 1, device=x_t.device, dtype=torch.float32)

        if past_state is not None:
            S_prev, Z_prev = past_state
        else:
            S_prev = torch.zeros(B, H, D_k, D_v, device=x_t.device, dtype=torch.float32)
            Z_prev = torch.zeros(B, H, D_k, 1, device=x_t.device, dtype=torch.float32)

        kt_t = phi_k.transpose(-1, -2).contiguous()
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
    rank: Optional[int] = 192,
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

    # Determine nested encoder injection point (e.g. RoBERTa target.attention.self)
    is_nested = hasattr(target, "attention") and hasattr(target.attention, "self")
    if is_nested and curr_attn is target.attention:
        curr_attn = target.attention.self

    try:
        new_attn = SVDLinearAttention(
            config=cfg,
            old_attn=curr_attn,
            rank=rank,
            mode=mode,
            lora_rank=lora_rank,
            **kwargs,
        )
        new_attn.is_nested_encoder = is_nested
    except Exception as e:
        logger.warning(f"Failed to instantiate SVDLinearAttention: {e}")
        return None

    if target is not None:
        if is_nested:
            target.attention.self = new_attn
        elif hasattr(target, "attention"):
            target.attention = new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn

    return new_attn
