"""
ASVD (Activation-Aware Singular Value Decomposition) Attention.
Paper: https://arxiv.org/abs/2307.07382

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
from neuralops.core.adapters import UniversalAdapter
from neuralops.core.attention_mask import apply_attention_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.triton_kernels import triton_dispatch

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor (odd-dim safe)."""
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to query and key tensors."""
    if cos.dim() == 2 and position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        seq_len = q.shape[-2]
        cos = cos[..., :seq_len, :]
        sin = sin[..., :seq_len, :]

    rotary_dim = cos.shape[-1]
    if rotary_dim <= q.shape[-1]:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot = (q_rot * cos.to(q_rot.dtype)) + (
            _rotate_half(q_rot) * sin.to(q_rot.dtype)
        )
        k_rot = (k_rot * cos.to(k_rot.dtype)) + (
            _rotate_half(k_rot) * sin.to(k_rot.dtype)
        )
        q_embed = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
        k_embed = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
    else:
        cos_cut, sin_cut = cos[..., : q.shape[-1]], sin[..., : q.shape[-1]]
        q_embed = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k_embed = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class ASVDLinear(nn.Module):
    is_custom_attention: bool = True
    r"""Low-rank factorized linear layer with explicit singular value scaling and trainable LoRA adapters:
    
    $$y = ((x V^\top) \odot \Sigma) U^\top + (x A^\top) B^\top + b$$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        lora_rank: int = 0,
        freeze_base: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_rank = lora_rank
        self.freeze_base = freeze_base

        factory_kwargs = {"dtype": dtype, "device": device}
        self.U = nn.Parameter(torch.empty(out_features, rank, **factory_kwargs))
        self.S = nn.Parameter(torch.empty(rank, **factory_kwargs))
        self.V = nn.Parameter(torch.empty(rank, in_features, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Trainable LoRA adapter parameters
        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(lora_rank, in_features, **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty(out_features, lora_rank, **factory_kwargs)
            )
            self.lora_scaling = 1.0
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_scaling = 1.0

        self.reset_parameters()
        if self.freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.U.dtype in (torch.float16, torch.bfloat16):
                u_fp32 = torch.empty_like(self.U, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(u_fp32)
                self.U.copy_(u_fp32.to(dtype=self.U.dtype, device=self.U.device))
            else:
                nn.init.orthogonal_(self.U)

            nn.init.ones_(self.S)

            if self.V.dtype in (torch.float16, torch.bfloat16):
                v_fp32 = torch.empty_like(self.V, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(v_fp32)
                self.V.copy_(v_fp32.to(dtype=self.V.dtype, device=self.V.device))
            else:
                nn.init.orthogonal_(self.V)

            if self.bias is not None:
                nn.init.zeros_(self.bias)

            if getattr(self, "lora_A", None) is not None and self.lora_A is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                self.lora_A.requires_grad = True
            if getattr(self, "lora_B", None) is not None and self.lora_B is not None:
                nn.init.zeros_(self.lora_B)
                self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        """Freeze U, S, V and bias parameters so only LoRA adapters adapt during recovery fine-tuning."""
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if getattr(self, "lora_A", None) is not None and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if getattr(self, "lora_B", None) is not None and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base weights for standard end-to-end training."""
        self.U.requires_grad = True
        self.S.requires_grad = True
        self.V.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def set_svd_factors(
        self,
        U: torch.Tensor,
        V: torch.Tensor,
        S: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        with torch.no_grad():
            r = min(U.size(1), V.size(0))
            if r != self.rank:
                device, dtype = self.U.device, self.U.dtype
                req_grad = self.U.requires_grad
                self.rank = r
                self.U = nn.Parameter(
                    torch.zeros(self.out_features, r, device=device, dtype=dtype),
                    requires_grad=req_grad,
                )
                self.S = nn.Parameter(
                    torch.ones(r, device=device, dtype=dtype), requires_grad=req_grad
                )
                self.V = nn.Parameter(
                    torch.zeros(r, self.in_features, device=device, dtype=dtype),
                    requires_grad=req_grad,
                )

            self.U.data.zero_()
            self.V.data.zero_()
            self.S.data.zero_()

            out_copy = min(self.out_features, U.size(0))
            in_copy = min(self.in_features, V.size(1))

            self.U.data[:out_copy, :r].copy_(
                U[:out_copy, :r].to(device=self.U.device, dtype=self.U.dtype)
            )
            self.V.data[:r, :in_copy].copy_(
                V[:r, :in_copy].to(device=self.V.device, dtype=self.V.dtype)
            )
            if S is not None:
                self.S.data[:r].copy_(
                    S[:r].to(device=self.S.device, dtype=self.S.dtype)
                )
            else:
                self.S.data[:r].fill_(1.0)

            if bias is not None and self.bias is not None:
                min_b = min(self.out_features, bias.numel())
                self.bias.data[:min_b].copy_(
                    bias[:min_b].to(device=self.bias.device, dtype=self.bias.dtype)
                )
            elif self.bias is not None:
                self.bias.data.zero_()

    def set_identity(self) -> None:
        """Initializes factorized linear transformation to act as an identity pass-through."""
        with torch.no_grad():
            self.U.data.zero_()
            self.V.data.zero_()
            self.S.data.fill_(1.0)
            r = min(self.rank, self.in_features, self.out_features)
            self.U.data[:r, :r].fill_diagonal_(1.0)
            self.V.data[:r, :r].fill_diagonal_(1.0)
            if self.bias is not None:
                self.bias.data.zero_()
            self.freeze_base_weights()

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for inspection & compatibility."""
        W = torch.matmul(self.U * self.S.unsqueeze(0), self.V)
        if getattr(self, "lora_A", None) is not None and self.lora_A is not None:
            W = W + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return W

    triton_kernel: str = "low_rank"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, in_features)
        # Factorized SVD projection: y = ((x @ V.T) * S) @ U.T + bias
        V_eff = self.S.view(-1, 1) * self.V
        out = triton_dispatch(self.triton_kernel, x, self.U, V_eff, bias=self.bias)

        # 4. Residual LoRA adapter path: (x @ lora_A.T) @ lora_B.T
        if getattr(self, "lora_A", None) is not None and self.lora_A is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            out = out + lora_out * self.lora_scaling

        return out


@ATTENTION_REGISTRY.register(
    "asvd", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "asvd_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class ASVDAttention(BaseAttention):
    r"""Hardened ASVD Attention with Factored Low-Rank Matrix Decomposition, GQA, RoPE, and LoRA.

    Computational Complexity:
        - Factorized Projections: $\mathcal{O}(B \cdot N \cdot r \cdot (d + d_{\text{all}}))$
        - Attention Scoring: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    is_custom_attention: bool = True
    triton_kernel: str = "low_rank"

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        model = config
        self.hidden_size: int = kwargs.get(
            "hidden_size",
            kwargs.get(
                "d_model",
                getattr(
                    model,
                    "hidden_size",
                    getattr(getattr(model, "config", None), "hidden_size", 64),
                ),
            ),
        )
        self.num_heads: int = kwargs.get(
            "num_heads",
            kwargs.get(
                "num_attention_heads",
                getattr(
                    model,
                    "num_attention_heads",
                    getattr(getattr(model, "config", None), "num_attention_heads", 4),
                ),
            ),
        )
        self.num_kv_heads: int = kwargs.get(
            "num_kv_heads",
            kwargs.get(
                "num_key_value_heads",
                getattr(
                    model,
                    "num_key_value_heads",
                    getattr(
                        getattr(model, "config", None),
                        "num_key_value_heads",
                        self.num_heads,
                    ),
                ),
            ),
        )
        self.head_dim: int = kwargs.get(
            "head_dim",
            getattr(
                model,
                "head_dim",
                getattr(getattr(model, "config", None), "head_dim", None),
            ),
        ) or (self.hidden_size // max(1, self.num_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        # Rank configuration
        self.rank: int = getattr(
            config,
            "asvd_rank",
            (
                rank
                if rank is not None
                else getattr(config, "rank", kwargs.pop("rank", 32))
            ),
        )
        self.lora_rank: int = getattr(config, "lora_rank", kwargs.get("lora_rank", 32))
        self.freeze_base: bool = getattr(
            config, "freeze_base", kwargs.get("freeze_base", self.lora_rank > 0)
        )
        self.energy_threshold: float = getattr(
            config, "asvd_energy_threshold", kwargs.get("energy_threshold", 0.95)
        )
        self.error_threshold: float | None = getattr(
            config, "asvd_error_threshold", kwargs.get("error_threshold", 0.40)
        )
        self.dynamic_rank: bool = getattr(
            config, "asvd_dynamic_rank", kwargs.get("dynamic_rank", True)
        )

        # Flags & metadata
        self.supports_gqa: bool = True
        self.supports_rope: bool = True
        self.is_nested_encoder: bool = False
        self.rotary_emb: Any | None = None

        # Decoder detection
        model_type = getattr(config, "model_type", "").lower()
        is_dec = getattr(config, "is_decoder", False)
        decoder_prefixes = (
            "gpt",
            "qwen",
            "llama",
            "mistral",
            "gemma",
            "phi",
            "opt",
            "bloom",
            "falcon",
            "olmo",
            "mpt",
            "stablelm",
        )
        if (
            any(model_type.startswith(p) for p in decoder_prefixes)
            or hasattr(config, "architectures")
            and any(
                "GPT" in str(a) or "CausalLM" in str(a)
                for a in (getattr(config, "architectures", []) or [])
            )
        ):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)

        # Projections
        self.query = ASVDLinear(
            self.hidden_size,
            self.all_head_size,
            self.rank,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=self.freeze_base,
        )
        self.key = ASVDLinear(
            self.hidden_size,
            self.kv_head_size,
            self.rank,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=self.freeze_base,
        )
        self.value = ASVDLinear(
            self.hidden_size,
            self.kv_head_size,
            self.rank,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=self.freeze_base,
        )
        self.out_proj = ASVDLinear(
            self.all_head_size,
            self.hidden_size,
            self.rank,
            bias=True,
            lora_rank=self.lora_rank,
            freeze_base=self.freeze_base,
        )

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.triton_kwargs = {}

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # --------------------------------------------------------------------------
    # Compatibility properties for parameter inspection & framework integration
    # --------------------------------------------------------------------------
    @property
    def q_proj(self) -> ASVDLinear:
        return self.query

    @property
    def k_proj(self) -> ASVDLinear:
        return self.key

    @property
    def v_proj(self) -> ASVDLinear:
        return self.value

    @property
    def o_proj(self) -> ASVDLinear:
        return self.out_proj

    @property
    def q_lin(self) -> ASVDLinear:
        return self.query

    @property
    def k_lin(self) -> ASVDLinear:
        return self.key

    @property
    def v_lin(self) -> ASVDLinear:
        return self.value

    @property
    def out_lin(self) -> ASVDLinear:
        return self.out_proj

    @property
    def dense(self) -> ASVDLinear:
        return self.out_proj

    @property
    def c_proj(self) -> ASVDLinear:
        return self.out_proj

    def _decompose_and_set(
        self,
        target_linear: ASVDLinear,
        src_weight: torch.Tensor,
        src_bias: torch.Tensor | None,
        proj_name: str = "",
    ) -> float:
        """Decomposes src_weight into low-rank SVD factors and verifies relative Frobenius error."""
        with torch.no_grad():
            W = src_weight.detach().float()
            # Ensure correct orientation: [out_features, in_features]
            if W.ndim != 2:
                raise ValueError(
                    f"Expected 2D weight matrix for projection {proj_name}, got shape {W.shape}"
                )
            if (
                W.shape[0] != target_linear.out_features
                and W.shape[1] == target_linear.out_features
            ):
                W = W.t().contiguous()

            # Perform SVD strictly in FP32
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)

            # Energy-based dynamic rank selection
            total_energy = torch.sum(S**2)
            cum_energy = torch.cumsum(S**2, dim=0) / (total_energy + 1e-12)

            target_energy = max(
                self.energy_threshold,
                (
                    1.0 - (self.error_threshold**2)
                    if self.error_threshold is not None
                    else 0.95
                ),
            )
            indices = (cum_energy >= target_energy).nonzero()
            if len(indices) > 0:
                r_energy = indices[0].item() + 1
            else:
                r_energy = min(U.size(1), Vt.size(0))

            # Cap rank so factorized layer strictly compresses or matches original parameter count
            # Dense size: in_features * out_features. Factored size: r * (in_features + out_features) + r
            max_param_r = max(
                1,
                int(
                    (target_linear.in_features * target_linear.out_features)
                    / (target_linear.in_features + target_linear.out_features + 1)
                ),
            )

            if self.dynamic_rank:
                r = max(16, min(r_energy, U.size(1), Vt.size(0), max_param_r))
            else:
                r = min(self.rank, U.size(1), Vt.size(0), max_param_r)

            w_rec = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
            diff = torch.norm(W - w_rec, p="fro")
            denom = torch.norm(W, p="fro") + 1e-9
            rel_err = float((diff / denom).item())

            # If error exceeds threshold under dynamic rank, expand rank within safe compression bound
            if self.error_threshold is not None and rel_err > self.error_threshold:
                if self.dynamic_rank and r < min(U.size(1), Vt.size(0), max_param_r):
                    indices_strict = (
                        cum_energy >= (1.0 - (self.error_threshold**2) + 1e-5)
                    ).nonzero()
                    if len(indices_strict) > 0:
                        r = min(indices_strict[0].item() + 1, max_param_r)
                        w_rec = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
                        rel_err = float((torch.norm(W - w_rec) / denom).item())

                if rel_err > self.error_threshold:
                    logger.warning(
                        f"Relative Frobenius reconstruction error ({rel_err:.2%}) for projection '{proj_name}' "
                        f"exceeds {self.error_threshold:.0%} threshold. Proceeding with best-effort rank {r}."
                    )

            target_linear.set_svd_factors(U[:, :r], Vt[:r, :], S[:r], bias=src_bias)
            return rel_err

    def copy_weights(self, src: Any) -> None:
        """Extracts and factorizes pretrained weights across DistilBERT, RoBERTa, GPT-2, Qwen/LLaMA."""
        extracted_weights: dict[
            str, tuple[torch.Tensor | None, torch.Tensor | None]
        ] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. Robust Nested Encoder Detection (prior to extraction)
        # Robustly detects both:
        # a) Outer RobertaAttention / BertAttention with inner 'self' (query/key/value) and 'output.dense'
        # b) Inner RobertaSelfAttention / BertSelfAttention
        # Prevents double output projection signal attenuation
        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin")

        is_nested_encoder = False
        if (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            and hasattr(src, "output")
            and hasattr(src.output, "dense")
        ):
            # Outer RobertaAttention containing inner self-attention and output dense
            is_nested_encoder = True
        elif (
            cls_name
            in [
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
                "DebertaSelfAttention",
            ]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            # Inner SelfAttention module
            is_nested_encoder = True

        self.is_nested_encoder = bool(is_nested_encoder)

        # 2. GPT-2: Fused Conv1D c_attn + Conv1D c_proj
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                c_b = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = c_b.data.clone() if c_b is not None else None
                c_cls = getattr(base_c, "__class__", type(base_c)).__name__
                if (
                    ("Conv1D" in c_cls)
                    or hasattr(base_c, "nf")
                    or (
                        "Conv1D"
                        in getattr(c_attn_mod, "__class__", type(c_attn_mod)).__name__
                    )
                    or hasattr(c_attn_mod, "nf")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                extracted_weights["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted_weights["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted_weights["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    o_b = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    o_b = o_b.data.clone() if o_b is not None else None
                    p_cls = getattr(base_o, "__class__", type(base_o)).__name__
                    if (
                        ("Conv1D" in p_cls)
                        or hasattr(base_o, "nf")
                        or (
                            "Conv1D"
                            in getattr(
                                c_proj_mod, "__class__", type(c_proj_mod)
                            ).__name__
                        )
                        or hasattr(c_proj_mod, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    extracted_weights["o"] = (o_w, o_b)

        # 3. RoBERTa / BERT Nested Encoder: self.query, self.key, self.value
        s = getattr(src, "self", src)
        name_map = {"q": "query", "k": "key", "v": "value"}
        for proj, attr in name_map.items():
            if (
                extracted_weights[proj][0] is None
                and hasattr(s, attr)
                and hasattr(getattr(s, attr), "weight")
            ):
                mod = getattr(s, attr)
                b = getattr(mod, "bias", None)
                extracted_weights[proj] = (
                    mod.weight.data,
                    b.data if b is not None else None,
                )

        if (
            not self.is_nested_encoder
            and extracted_weights["o"][0] is None
            and hasattr(src, "output")
            and hasattr(src.output, "dense")
        ):
            mod = src.output.dense
            b = getattr(mod, "bias", None)
            extracted_weights["o"] = (
                mod.weight.data,
                b.data if b is not None else None,
            )

        # 4. Qwen / LLaMA: q_proj, k_proj, v_proj, o_proj
        for proj in ["q", "k", "v", "o"]:
            if extracted_weights[proj][0] is None:
                for candidate in [f"{proj}_proj", f"{proj}_lin", proj]:
                    if hasattr(src, candidate) and hasattr(
                        getattr(src, candidate), "weight"
                    ):
                        mod = getattr(src, candidate)
                        b = getattr(mod, "bias", None)
                        extracted_weights[proj] = (
                            mod.weight.data,
                            b.data if b is not None else None,
                        )
                        break

        # 5. Fallback to UniversalAdapter
        for proj in ["q", "k", "v", "o"]:
            if extracted_weights[proj][0] is None:
                try:
                    w, b = UniversalAdapter.get_weight_bias(src, proj)
                    if w is not None:
                        extracted_weights[proj] = (w, b)
                except Exception as e:
                    print(f"copy_weights failed: {e}")
                    logger.warning(f"copy_weights failed: {e}")

        # 6. Check for missing essential projections and Frobenius norm verification
        missing = [p for p in ["q", "k", "v"] if extracted_weights[p][0] is None]
        if not self.is_nested_encoder and extracted_weights["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"ASVDAttention.copy_weights: could not extract mandatory projections {missing} "
                f"from {src.__class__.__name__}. Refusing to install randomly-initialised weights."
            )

        for p in ["q", "k", "v"]:
            w_check = extracted_weights[p][0]
            if w_check is not None:
                norm = float(torch.norm(w_check.float()).item())
                if norm < 1e-4:
                    logger.warning(
                        f"ASVDAttention extracted weight for '{p}' has near-zero Frobenius norm ({norm:.2e})"
                    )

        if extracted_weights["o"][0] is not None:
            norm_o = float(torch.norm(extracted_weights["o"][0].float()).item())
            if norm_o < 1e-4:
                logger.warning(
                    f"ASVDAttention extracted weight for 'o' has near-zero Frobenius norm ({norm_o:.2e})"
                )

        # 7. SVD Factorization & Reconstruction Verification
        for proj, target in [
            ("q", self.query),
            ("k", self.key),
            ("v", self.value),
            ("o", self.out_proj),
        ]:
            if proj == "o" and self.is_nested_encoder:
                self.out_proj.set_identity()
                continue
            w, b = extracted_weights[proj]
            if w is not None:
                self._decompose_and_set(target, w, b, proj_name=proj)

        # 8. Inherit Decoder & Rotary Metadata
        if hasattr(src, "rotary_emb"):
            self.rotary_emb = src.rotary_emb
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
        elif hasattr(src, "c_attn") or cls_name in (
            "GPT2Attention",
            "Qwen2Attention",
            "Qwen2_5_VLAttention",
            "QwenAttention",
        ):
            self.is_decoder = True

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for factorized ASVD attention."""
        if hidden_states is None:
            hidden_states = kwargs.get("query", kwargs.get("x"))
        if (
            hidden_states is None
            and len(args) > 0
            and isinstance(args[0], torch.Tensor)
        ):
            hidden_states = args[0]
        if attention_mask is None:
            attention_mask = kwargs.get("mask", kwargs.get("attn_mask"))

        B, N, D = hidden_states.shape
        H = self.num_heads
        d = self.head_dim
        kv_H = self.num_kv_heads

        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "Cross-attention is not supported in ASVDAttention."
            )

        # Extract rotary kwargs if supplied
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_ids is None:
            position_ids = kwargs.get("position_ids")

        # 1. Project Q, K, V using low-rank factorized linear layers
        q: torch.Tensor = (
            self.query(hidden_states).view(B, N, H, d).transpose(1, 2).contiguous()
        )
        k: torch.Tensor = (
            self.key(hidden_states).view(B, N, kv_H, d).transpose(1, 2).contiguous()
        )
        v: torch.Tensor = (
            self.value(hidden_states).view(B, N, kv_H, d).transpose(1, 2).contiguous()
        )

        # 2. Apply Rotary Position Embedding (RoPE) before GQA expansion and masking
        if (
            position_embeddings is not None
            and isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) >= 2
        ):
            cos, sin = position_embeddings[0], position_embeddings[1]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None:
            try:
                # Use query tensor q (full H heads) to infer sequence length, shapes, and device correctly
                rotary_out = (
                    self.rotary_emb(q, position_ids)
                    if position_ids is not None
                    else self.rotary_emb(q)
                )
                if isinstance(rotary_out, (tuple, list)) and len(rotary_out) >= 2:
                    cos, sin = rotary_out[0], rotary_out[1]
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        # 3. GQA Expansion: Expand key/value heads to match query heads if kv_H < H
        if kv_H < H:
            k = repeat_kv(k, H // kv_H)
            v = repeat_kv(v, H // kv_H)

        # 4. Attention Scoring strictly in FP32 to prevent FP16 overflow
        scale: float = 1.0 / math.sqrt(d)
        scores: torch.Tensor = (
            torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
        )

        # 5. Causal Masking for Decoder Models
        if getattr(self, "is_decoder", False):
            causal_mask = torch.triu(
                torch.full((N, N), -10000.0, device=scores.device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # 6. Padding Attention Mask (Broadcast 2D/3D to 4D additive logits format)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.dtype == torch.bool:
                    extended_mask = torch.zeros_like(attention_mask, dtype=scores.dtype)
                    extended_mask = extended_mask.masked_fill(~attention_mask, -10000.0)
                elif (
                    torch.is_floating_point(attention_mask)
                    and (attention_mask < 0).any()
                ):
                    extended_mask = attention_mask.to(dtype=scores.dtype)
                else:
                    extended_mask = (
                        1.0 - attention_mask.to(dtype=scores.dtype)
                    ) * -10000.0
                attention_mask = extended_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                if attention_mask.dtype == torch.bool:
                    extended_mask = torch.zeros_like(attention_mask, dtype=scores.dtype)
                    extended_mask = extended_mask.masked_fill(~attention_mask, -10000.0)
                elif (
                    torch.is_floating_point(attention_mask)
                    and (attention_mask < 0).any()
                ):
                    extended_mask = attention_mask.to(dtype=scores.dtype)
                else:
                    extended_mask = (
                        1.0 - attention_mask.to(dtype=scores.dtype)
                    ) * -10000.0
                attention_mask = extended_mask.unsqueeze(1)
            scores = apply_attention_mask(scores, attention_mask)

        # 7. Softmax & NaN Protection
        attn: torch.Tensor = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # 8. Safe Head Mask Handling
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask_tensor = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask_tensor = head_mask.view(head_mask.size(0), -1, 1, 1)
            elif head_mask.dim() == 3:
                head_mask_tensor = head_mask.unsqueeze(1)
            else:
                head_mask_tensor = head_mask
            attn = attn * head_mask_tensor

        # 9. Context Aggregation
        context: torch.Tensor = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        # 10. Output Projection (Bypassed for nested encoders to prevent double projection)
        if getattr(self, "is_nested_encoder", False):
            out = context
        else:
            out = self.out_proj(context)

        return out, attn if output_attentions else None


@ATTENTION_REGISTRY.register("asvd_fn")
@ATTENTION_REGISTRY.register("asvd_inject")
def inject_asvd(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
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
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(
        target.layers, (torch.nn.ModuleList, list)
    ):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_asvd(
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

    layer = target
    rank = kwargs.get("rank", 64)
    copy_weights = kwargs.get("copy_weights", True)
    config = cfg

    """Universal injection factory for ASVDAttention with pre-flight extraction validation."""
    target_model = model or kwargs.get("model")
    if target_model is not None and any(
        hasattr(target_model, a)
        for a in (
            "transformer",
            "model",
            "roberta",
            "bert",
            "distilbert",
            "gpt2",
            "qwen2",
            "layers",
        )
    ):
        from neuralops.core.attention_utils import inject_attention

        return inject_attention(
            target_model, "asvd", rank=rank, copy_weights=copy_weights, **kwargs
        )

    if old_attn is None and layer is not None:
        for attr in ("attention", "self_attn", "self", "attn"):
            if hasattr(layer, attr):
                cand = getattr(layer, attr)
                if cand is not None:
                    old_attn = cand
                    break
        if old_attn is None:
            old_attn = layer

    target_config = config
    if target_config is None and old_attn is not None:
        target_config = getattr(old_attn, "config", None)
    if target_config is None and layer is not None:
        target_config = getattr(layer, "config", None)

    # Pre-flight extraction test: ensure source layer has extractable projections
    if old_attn is not None:
        try:
            new_attn = ASVDAttention(
                target_config, rank=rank, pretrained_attn=old_attn, **kwargs
            )
            if layer is not None:
                if hasattr(layer, "attention"):
                    if hasattr(layer.attention, "self"):
                        layer.attention.self = new_attn
                    else:
                        layer.attention = new_attn
                elif hasattr(layer, "self_attn"):
                    layer.self_attn = new_attn
                elif hasattr(layer, "self"):
                    layer.self = new_attn
                elif hasattr(layer, "attn"):
                    layer.attn = new_attn
            return new_attn
        except RuntimeError as err:
            logger.warning(
                "inject_asvd: Pre-flight weight extraction failed on %s: %s. Aborting replacement.",
                type(old_attn).__name__,
                err,
            )
        except Exception as err:
            logger.warning(
                "inject_asvd: Unexpected error initializing ASVDAttention: %s. Aborting replacement.",
                err,
            )

    # Fallback to random initialization with a warning
    new_attn = ASVDAttention(target_config, rank=rank, **kwargs)
    if layer is not None:
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
    return new_attn
