"""
SVD Projection Attention.
Each projection (Q, K, V, Out) is factored into low-rank SVD components U and V,
with trainable LoRA adapters for recovery fine-tuning.

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.triton_kernels import triton_dispatch

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    b = getattr(base, "bias", None)
    if w is not None:
        w = w.data.clone()
        if (module.__class__.__name__ == "Conv1D") or (base.__class__.__name__ == "Conv1D"):
            w = w.t().contiguous()
    if b is not None:
        b = b.data.clone()
    return w, b


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    rotary_dim = x.shape[-1]
    rotary_dim = rotary_dim - (rotary_dim % 2)
    x_rot = x[..., :rotary_dim]
    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    if x.shape[-1] > rotary_dim:
        return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) with partial dimension support."""
    rotary_dim = cos.shape[-1]
    rotary_dim = rotary_dim - (rotary_dim % 2)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos = cos[..., :rotary_dim]
    sin = sin[..., :rotary_dim]

    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.shape[-1] > 0 else q_rot
    k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.shape[-1] > 0 else k_rot
    return q, k


class SVDProjectionLinear(nn.Module):
    is_custom_attention: bool = True
    r"""Low-Rank SVD Linear Layer with Trainable LoRA Residual Adapters.

    Forward computes:
    $$y = x (U V)^\top + b + \frac{\alpha}{r_{\text{lora}}} x A^\top B^\top$$
    where $U \in \mathbb{R}^{d_{\text{out}} \times r}$, $V \in \mathbb{R}^{r \times d_{\text{in}}}$
    are frozen SVD singular subspaces, and $A, B$ are trainable LoRA adapters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = rank
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha
        self.scaling: float = lora_alpha / max(1, lora_rank)

        self.U: nn.Parameter = nn.Parameter(torch.empty(out_features, rank))
        self.V: nn.Parameter = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.U.requires_grad = False
            self.V.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def reset_parameters(self) -> None:
        """Initialize SVD parameters with Kaiming uniform distribution."""
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective dense weight matrix."""
        return torch.matmul(self.U, self.V)

    triton_kernel: str = "low_rank"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base factorized SVD projection
        out = triton_dispatch(self.triton_kernel, x, self.U, self.V, bias=self.bias)

        # Trainable LoRA residual branch
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_out

        return out

    def set_svd_factors(self, U: torch.Tensor, V: torch.Tensor) -> None:
        """Set precomputed U and V factors."""
        with torch.no_grad():
            r = min(self.rank, U.size(1), V.size(0))
            self.U.data.zero_()
            self.V.data.zero_()
            self.U.data[:, :r].copy_(U[:, :r].to(device=self.U.device, dtype=self.U.dtype))
            self.V.data[:r, :].copy_(V[:r, :].to(device=self.V.device, dtype=self.V.dtype))


@ATTENTION_REGISTRY.register("svd_proj", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_proj_attn", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_projection_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("svd_projection", supports_causal=True, supports_gqa=True)
class SVDProjectionAttention(BaseAttention):
    r"""Multi-Head Attention using Factorized SVD Linear Projections.
    Hardened for production use across diverse transformer architectures.
    """

    supports_gqa: bool = True
    is_custom_attention: bool = True
    triton_kernel: str = "low_rank"

    def __init__(
        self,
        config: Any | None = None,
        old_attn: nn.Module | None = None,
        rank: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        dropout: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        # 1. Dimensions & Architecture
        self.hidden_size: int = kwargs.get(
            "hidden_size",
            _cfg_get(config, "hidden_size", _cfg_get(config, "dim", _cfg_get(config, "d_model", 768)))
        )
        self.num_heads: int = kwargs.get(
            "num_heads",
            kwargs.get(
                "num_attention_heads",
                _cfg_get(config, "num_attention_heads", _cfg_get(config, "num_heads", _cfg_get(config, "n_heads", 12)))
            )
        )
        self.head_dim: int = kwargs.get(
            "head_dim",
            _cfg_get(config, "head_dim", self.hidden_size // self.num_heads if self.num_heads > 0 else 64)
        )

        cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.q_head_size: int = self.num_heads * self.head_dim

        # 2. Causal decoder flag detection
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(_cfg_get(config, "model_type", "") or "").lower()
            _dec_pfx = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or any(_mt.startswith(p) for p in _dec_pfx)
                or (old_attn is not None and hasattr(old_attn, "c_attn"))
                or bool(kwargs.get("is_causal", False))
            )

        self.is_nested_encoder: bool = False
        comp_config = _cfg_get(config, "compression_config", kwargs.get("comp_config"))
        self.rank: int = (
            rank
            if rank is not None
            else _cfg_get(comp_config, "rank", _cfg_get(config, "svd_rank", _cfg_get(config, "rank", 32)))
        )
        self.lora_rank = _cfg_get(config, "lora_rank", _cfg_get(comp_config, "lora_rank", lora_rank if lora_rank is not None else 16))
        self.lora_alpha = _cfg_get(config, "lora_alpha", _cfg_get(comp_config, "lora_alpha", lora_alpha))

        # 3. Factorized SVD linear projections
        self.q_proj = SVDProjectionLinear(
            self.hidden_size,
            self.q_head_size,
            self.rank,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.k_proj = SVDProjectionLinear(
            self.hidden_size,
            self.kv_head_size,
            self.rank,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.v_proj = SVDProjectionLinear(
            self.hidden_size,
            self.kv_head_size,
            self.rank,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )
        self.out_proj = SVDProjectionLinear(
            self.q_head_size,
            self.hidden_size,
            self.rank,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=True,
        )

        dropout_p = _cfg_get(
            config,
            "dropout",
            _cfg_get(
                config,
                "hidden_dropout_prob",
                _cfg_get(config, "attention_probs_dropout_prob", dropout),
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)
        self.triton_kwargs = {}

        # 4. Copy weights if old_attn passed
        if old_attn is not None:
            self.copy_weights(old_attn)
            try:
                src_device = next(old_attn.parameters()).device
                self.to(src_device)
            except StopIteration:
                pass

    @property
    def query(self) -> SVDProjectionLinear:
        return self.q_proj

    @property
    def key(self) -> SVDProjectionLinear:
        return self.k_proj

    @property
    def value(self) -> SVDProjectionLinear:
        return self.v_proj

    @property
    def q_lin(self) -> SVDProjectionLinear:
        return self.q_proj

    @property
    def k_lin(self) -> SVDProjectionLinear:
        return self.k_proj

    @property
    def v_lin(self) -> SVDProjectionLinear:
        return self.v_proj

    @property
    def out_lin(self) -> SVDProjectionLinear:
        return self.out_proj

    def _decompose_and_set(
        self, target: SVDProjectionLinear, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> None:
        """Decomposes full weight matrix into SVD factors and copies bias."""
        with torch.no_grad():
            W = weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            r = min(target.rank, U.size(1), Vt.size(0))
            sqrt_s = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            target.U.data.zero_()
            target.V.data.zero_()
            target.U.data[:, :r] = (U[:, :r] * sqrt_s.unsqueeze(0)).to(
                device=target.U.device, dtype=target.U.dtype
            )
            target.V.data[:r, :] = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(
                device=target.V.device, dtype=target.V.dtype
            )
            if target.bias is not None and bias is not None:
                target.bias.data.copy_(
                    bias.to(device=target.bias.device, dtype=target.bias.dtype)
                )

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

            # Pattern 1: GPT-2 Fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn_w, c_attn_b = _unpack_weight_bias(src.c_attn)
                if c_attn_w is not None:
                    q_dim = self.q_head_size
                    k_dim = self.kv_head_size
                    v_dim = self.kv_head_size
                    if c_attn_w.size(0) >= q_dim + k_dim + v_dim:
                        w_q = c_attn_w[:q_dim, :]
                        w_k = c_attn_w[q_dim : q_dim + k_dim, :]
                        w_v = c_attn_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]
                        if c_attn_b is not None:
                            b_q = c_attn_b[:q_dim]
                            b_k = c_attn_b[q_dim : q_dim + k_dim]
                            b_v = c_attn_b[q_dim + k_dim : q_dim + k_dim + v_dim]
                    else:
                        split_size = c_attn_w.size(0) // 3
                        w_q = c_attn_w[:split_size, :]
                        w_k = c_attn_w[split_size : 2 * split_size, :]
                        w_v = c_attn_w[2 * split_size : 3 * split_size, :]
                        if c_attn_b is not None:
                            b_q = c_attn_b[:split_size]
                            b_k = c_attn_b[split_size : 2 * split_size]
                            b_v = c_attn_b[2 * split_size : 3 * split_size]

                w_o, b_o = _unpack_weight_bias(src.c_proj)

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
                    w_q, b_q = UniversalAdapter.get_weight_bias(old_attn, "q")
                    w_k, b_k = UniversalAdapter.get_weight_bias(old_attn, "k")
                    w_v, b_v = UniversalAdapter.get_weight_bias(old_attn, "v")
                    if w_o is None:
                        w_o, b_o = UniversalAdapter.get_weight_bias(old_attn, "o")
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
                logger.warning(f"Failed to extract required projections: {missing}")
                return

            # Frobenius norm checks
            for name, w in [("w_q", w_q), ("w_k", w_k), ("w_v", w_v)]:
                if torch.norm(w.float(), p="fro").item() < 1e-4:
                    logger.warning(f"Copied weight {name} has near-zero Frobenius norm (< 1e-4)")

            # Decompose and set Q
            self._decompose_and_set(self.q_proj, w_q, b_q)

            # Handle GQA pooling/repetition for K and V
            def _align_kv(w: torch.Tensor, b: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
                if w.shape[0] == self.kv_head_size:
                    return w, b
                elif w.shape[0] > self.kv_head_size and w.shape[0] % self.kv_head_size == 0:
                    rep = w.shape[0] // self.kv_head_size
                    w_pooled = (
                        w.view(self.num_kv_heads, rep, self.head_dim, -1)
                        .mean(dim=1)
                        .view(self.kv_head_size, -1)
                    )
                    b_pooled = (
                        b.view(self.num_kv_heads, rep, self.head_dim).mean(dim=1).view(-1)
                        if b is not None else None
                    )
                    return w_pooled, b_pooled
                elif w.shape[0] < self.kv_head_size and self.kv_head_size % w.shape[0] == 0:
                    rep = self.kv_head_size // w.shape[0]
                    w_rep = w.repeat(rep, 1)
                    b_rep = b.repeat(rep) if b is not None else None
                    return w_rep, b_rep
                else:
                    new_w = torch.zeros(self.kv_head_size, w.shape[1], device=w.device, dtype=w.dtype)
                    new_w[:min(self.kv_head_size, w.shape[0]), :min(self.hidden_size, w.shape[1])].copy_(
                        w[:min(self.kv_head_size, w.shape[0]), :min(self.hidden_size, w.shape[1])]
                    )
                    new_b = None
                    if b is not None:
                        new_b = torch.zeros(self.kv_head_size, device=b.device, dtype=b.dtype)
                        new_b[:min(self.kv_head_size, b.shape[0])].copy_(b[:min(self.kv_head_size, b.shape[0])])
                    return new_w, new_b

            w_k_aligned, b_k_aligned = _align_kv(w_k, b_k)
            self._decompose_and_set(self.k_proj, w_k_aligned, b_k_aligned)

            w_v_aligned, b_v_aligned = _align_kv(w_v, b_v)
            self._decompose_and_set(self.v_proj, w_v_aligned, b_v_aligned)

            # Decompose and set out_proj if not nested encoder
            if w_o is not None and not self.is_nested_encoder:
                if torch.norm(w_o.float(), p="fro").item() < 1e-4:
                    logger.warning("Copied out_proj weight has near-zero Frobenius norm (< 1e-4)")
                self._decompose_and_set(self.out_proj, w_o, b_o)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Project Q, K, V
        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # 2. RoPE support
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # 3. GQA expansion
        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # 4. Attention scores in FP32
        scale = 1.0 / math.sqrt(self.head_dim)
        # Memory-efficient: use SDPA to avoid OOM on large seqs; fallback to explicit if needed

        try:

            _qs = Q.half() if Q.dtype == torch.float32 else Q

            _ks = K.half() if K.dtype == torch.float32 else K

            _vs = V.half() if V.dtype == torch.float32 else V

            _is_causal = getattr(self, 'is_decoder', False) and not is_cross and N > 1 if 'is_cross' in dir() else getattr(self, 'is_decoder', False) and N > 1

            _sdpa_out = torch.nn.functional.scaled_dot_product_attention(_qs, _ks, _vs, is_causal=_is_causal, dropout_p=self.dropout.p if self.training else 0.0)

            _sdpa_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(B, N, self.all_head_size)

            _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)

            return (self.out_proj(_sdpa_out) if not getattr(self, 'is_nested_encoder', False) else _sdpa_out), None

        except Exception:

            pass

        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        # 5. Masking
        if self.is_decoder and not is_cross and N > 1:
            causal_mask = torch.triu(
                torch.full((N, kv_len), float("-inf"), device=scores.device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                scores = scores + eff_mask.float()

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_weights = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        attn_weights = self.dropout(attn_weights)

        # 6. Safe head_mask application
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn_weights = attn_weights * head_mask

        # 7. Context computation & Head Merge
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, N, -1)
        valid_mask = get_token_valid_mask(eff_mask, B, N, hidden_states.device)
        context = context * valid_mask.unsqueeze(-1).to(context.dtype)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        # 8. Output projection (bypassed if nested encoder)
        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        if output.shape[-1] != hidden_states.shape[-1]:
            if output.shape[-1] < hidden_states.shape[-1]:
                output = F.pad(output, (0, hidden_states.shape[-1] - output.shape[-1]))
            else:
                output = output[..., :hidden_states.shape[-1]]

        return output, (attn_weights if output_attentions else None)


@ATTENTION_REGISTRY.register("svd_projection_attention_inject")
@ATTENTION_REGISTRY.register("svd_proj_attn_inject")
@ATTENTION_REGISTRY.register("inject_svd_proj_attn")
@ATTENTION_REGISTRY.register("inject_svd_projection_attention")
@ATTENTION_REGISTRY.register("inject_svd_projection")
def inject_svd_proj_attn(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    lora_rank: int = 16,
    rank: int | None = None,
    **kwargs: Any,
) -> Any | None:
    """Safely constructs and injects SVDProjectionAttention into a model or transformer layer."""
    target = layer if layer is not None else model
    cfg = model_config or getattr(target, "config", None) or kwargs.get("config")

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
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                inject_svd_proj_attn(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    lora_rank=lora_rank,
                    rank=rank,
                    **kwargs,
                )
            return target

    src = old_attn
    if src is None and target is not None:
        for attr in ("attention", "attn", "self_attn", "self"):
            if hasattr(target, attr):
                src = getattr(target, attr)
                break

    try:
        new_attn = SVDProjectionAttention(
            config=cfg,
            old_attn=src,
            lora_rank=lora_rank,
            rank=rank,
            comp_config=comp_config,
            **kwargs,
        )
    except Exception as e:
        logger.warning(f"Failed to instantiate SVDProjectionAttention: {e}")
        return None

    if target is not None:
        if hasattr(target, "attention") and hasattr(target.attention, "self"):
            new_attn.is_nested_encoder = True
            target.attention.self = new_attn
            return new_attn
        elif hasattr(target, "self") and not hasattr(target, "dense"):
            new_attn.is_nested_encoder = True
            target.self = new_attn
            return new_attn
        elif hasattr(target, "attention"):
            target.attention = new_attn
            return new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
            return new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn
            return new_attn
        target.attention = new_attn

    return new_attn
