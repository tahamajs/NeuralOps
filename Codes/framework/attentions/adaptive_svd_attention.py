r"""
Adaptive SVD Attention with Residual Adapters and Dynamic Energy Thresholding.

This module provides low-rank factorized linear projections for Multi-Head Attention,
decomposing full-rank weight matrices via dynamic cumulative energy-thresholded
Singular Value Decomposition (SVD):
$$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
with symmetrical $\sqrt{\Sigma}$ singular value distribution:
$$U_r = U_{:, :r} \sqrt{\Sigma_r}, \quad V_r = \sqrt{\Sigma_r} V^\top_{:r, :}$$
augmented with trainable low-rank residual adapters:
$$W \approx U_r V_r + W_{\text{lora\_B}} W_{\text{lora\_A}}$$
"""

import logging
import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import apply_attention_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter
from framework.utils.compat import maybe_apply_rotary, repeat_kv

logger = logging.getLogger(__name__)


class AdaptiveSVDLinear(nn.Module):
    is_custom_attention: bool = True
    r"""
    Linear layer decomposed into a frozen SVD base and a trainable residual adapter.

    Uses `lora_A` and `lora_B` parameter names so fine-tuning optimizers and LoRA filters
    register them automatically as trainable parameters. The SVD base U and V matrices
    are frozen by default to preserve pretrained representation stability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int,
        residual_rank: int = 16,
        bias: bool = True,
        trainable_svd: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        max_possible_rank = min(in_features, out_features)
        self.svd_rank: int = min(svd_rank, max_possible_rank)
        self.residual_rank: int = min(residual_rank, max_possible_rank)

        factory_kwargs = {"dtype": dtype, "device": device}

        self.U: nn.Parameter = nn.Parameter(
            torch.empty((out_features, self.svd_rank), **factory_kwargs),
            requires_grad=trainable_svd,
        )
        self.V: nn.Parameter = nn.Parameter(
            torch.empty((self.svd_rank, in_features), **factory_kwargs),
            requires_grad=trainable_svd,
        )

        if self.residual_rank > 0:
            self.lora_B: Optional[nn.Parameter] = nn.Parameter(torch.empty((out_features, self.residual_rank), **factory_kwargs))
            self.lora_A: Optional[nn.Parameter] = nn.Parameter(torch.empty((self.residual_rank, in_features), **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Initializes parameters using orthogonal base and zeroed residual adapter (BA = 0 at init)."""
        with torch.no_grad():
            if self.U.dtype in (torch.float16, torch.bfloat16):
                u_fp32 = torch.empty_like(self.U, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(u_fp32)
                self.U.copy_(u_fp32.to(dtype=self.U.dtype, device=self.U.device))
            else:
                nn.init.orthogonal_(self.U)

            if self.V.dtype in (torch.float16, torch.bfloat16):
                v_fp32 = torch.empty_like(self.V, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(v_fp32)
                self.V.copy_(v_fp32.to(dtype=self.V.dtype, device=self.V.device))
            else:
                nn.init.orthogonal_(self.V)

            if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def set_svd_factors(self, U: torch.Tensor, V: torch.Tensor) -> None:
        r"""Sets the SVD factors $U$ and $V$ safely clamped to tensor dimensions."""
        with torch.no_grad():
            r = min(U.size(1), V.size(0), self.svd_rank)
            out_dim = min(self.U.size(0), U.size(0))
            in_dim = min(self.V.size(1), V.size(1))
            self.U.zero_()
            self.V.zero_()
            self.U[:out_dim, :r].copy_(U[:out_dim, :r])
            self.V[:r, :in_dim].copy_(V[:r, :in_dim])
            if getattr(self, "lora_B", None) is not None:
                nn.init.zeros_(self.lora_B)

    @property
    def A(self) -> Optional[nn.Parameter]:
        """Alias for lora_A."""
        return self.lora_A

    @property
    def B(self) -> Optional[nn.Parameter]:
        """Alias for lora_B."""
        return self.lora_B

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        W = torch.matmul(self.U, self.V)
        if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
            W = W + torch.matmul(self.lora_B, self.lora_A)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward computation for low-rank SVD + trainable residual adapter."""
        orig_dtype = x.dtype
        x_in = x.to(self.V.dtype)
        Vx = F.linear(x_in, self.V)
        out = F.linear(Vx, self.U.to(Vx.dtype))
        if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
            a = self.lora_A.to(x_in.dtype)
            b = self.lora_B.to(x_in.dtype)
            Ax = F.linear(x_in, a)
            out = out + F.linear(Ax, b)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.to(orig_dtype)


@ATTENTION_REGISTRY.register("adaptive_svd_attn", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("adaptive_svd", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("adaptive_svd_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class AdaptiveSVDAttention(BaseAttention):
    r"""Multi-Head Attention with Adaptive SVD + Residual factorized projections."""

    is_custom_attention: bool = True
    supports_causal = True
    supports_bidirectional = True
    supports_gqa = True
    supports_rope = True

    def __init__(
        self,
        config: Any,
        svd_rank: Optional[int] = None,
        residual_rank: int = 16,
        energy_threshold: float = 0.92,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        if config is not None:
            self.adapter = ModelAdapter(config)
            self.num_heads = kwargs.get("num_heads", getattr(self.adapter, "num_heads", None) or getattr(config, "num_attention_heads", 12))
            self.num_kv_heads = kwargs.get("num_kv_heads", getattr(self.adapter, "num_key_value_heads", None) or getattr(config, "num_key_value_heads", self.num_heads))
            self.hidden_size = kwargs.get("hidden_size", getattr(self.adapter, "hidden_size", None) or getattr(config, "hidden_size", 768))
            self.svd_rank = getattr(config, "svd_rank", svd_rank if svd_rank is not None else getattr(config, "rank", 32))
            self.residual_rank = getattr(config, "residual_rank", residual_rank)
            self.energy_threshold = float(getattr(config, "energy_threshold", kwargs.get("energy_threshold", energy_threshold)))
            dropout_p = getattr(config, "dropout", getattr(config, "hidden_dropout_prob", getattr(config, "attention_probs_dropout_prob", 0.1)))
            self.relative_attention_num_buckets = getattr(config, "relative_attention_num_buckets", 32)
            self.relative_attention_max_distance = getattr(config, "relative_attention_max_distance", 128)
        else:
            self.adapter = None
            self.hidden_size = kwargs.get("hidden_size", 768)
            self.num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads", 12))
            self.num_kv_heads = kwargs.get("num_kv_heads", kwargs.get("num_key_value_heads", self.num_heads))
            self.svd_rank = svd_rank if svd_rank is not None else kwargs.pop("rank", 32)
            self.residual_rank = residual_rank
            self.energy_threshold = float(kwargs.get("energy_threshold", energy_threshold))
            dropout_p = kwargs.get("dropout", 0.1)
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128

        self.num_heads = int(self.num_heads or 12)
        self.num_kv_heads = int(self.num_kv_heads or self.num_heads)
        self.hidden_size = int(self.hidden_size or 768)

        # Robust head_dim resolution: check source weight first
        head_dim = kwargs.get("head_dim", None)
        if not head_dim and pretrained_attn is not None:
            for attr in ("q_proj", "query", "q_lin"):
                mod = getattr(pretrained_attn, attr, None)
                if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
                    if mod.weight.ndim == 2:
                        head_dim = mod.weight.shape[0] // max(1, self.num_heads)
                        break
            if not head_dim:
                head_dim = getattr(pretrained_attn, "head_dim", None)

        if not head_dim and config is not None:
            head_dim = getattr(config, "head_dim", None)
            if not head_dim and self.adapter is not None:
                head_dim = getattr(self.adapter, "head_dim", None)

        if not head_dim or head_dim <= 0:
            head_dim = self.hidden_size // max(1, self.num_heads)

        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # Robust is_decoder detection with prefix matching
        _mt = str(getattr(config, "model_type", "") or "").lower() if config else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                             "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        is_dec = (
            getattr(config, "is_decoder", False)
            or getattr(config, "is_causal", False)
            or getattr(config, "causal", False)
            or any(_mt.startswith(p) for p in _decoder_prefixes)
            or (pretrained_attn is not None and hasattr(pretrained_attn, "c_attn"))
            or (pretrained_attn is not None and getattr(pretrained_attn, "is_causal", False))
            or (pretrained_attn is not None and getattr(pretrained_attn, "is_decoder", False))
            or bool(kwargs.get("is_decoder", False))
            or bool(kwargs.get("is_causal", False))
        )
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)

        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.supports_gqa: bool = True

        self.query = AdaptiveSVDLinear(
            self.hidden_size,
            self.all_head_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
        self.key = AdaptiveSVDLinear(
            self.hidden_size,
            self.kv_head_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
        self.value = AdaptiveSVDLinear(
            self.hidden_size,
            self.kv_head_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
        self.out_proj = AdaptiveSVDLinear(
            self.all_head_size,
            self.hidden_size,
            self.svd_rank,
            self.residual_rank,
            bias=True,
        )
        self.q_lin = self.query
        self.k_lin = self.key
        self.v_lin = self.value
        self.out_lin = self.out_proj
        self.q_proj = self.query
        self.k_proj = self.key
        self.v_proj = self.value
        self.dense = self.out_proj
        self.o_proj = self.out_proj
        self.c_proj = self.out_proj

        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        self.is_nested_encoder: bool = False
        self.has_relative_attention_bias: bool = False
        self.relative_attention_bias: Optional[nn.Embedding] = None

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _relative_position_bucket(
        self, relative_position: torch.Tensor, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.relative_attention_bias is not None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            relative_position = memory_position - context_position
            relative_position_bucket = self._relative_position_bucket(
                relative_position,
                bidirectional=(not self.is_decoder),
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            values = self.relative_attention_bias(relative_position_bucket)
            values = values.permute([2, 0, 1]).unsqueeze(0)
            return values
        return torch.zeros((1, self.num_heads, query_length, key_length), device=device)

    def _compute_energy_rank(self, S: torch.Tensor, max_rank: int, min_rank: int = 16) -> int:
        r"""
        Find smallest rank $r$ satisfying cumulative singular value energy threshold:
        $$\frac{\sum_{i=1}^r \sigma_i^2}{\sum_{j=1}^d \sigma_j^2} \ge \tau \quad (\tau = 0.92)$$
        """
        s_squared = S ** 2
        total_energy = s_squared.sum()
        if total_energy <= 1e-12:
            return max(1, min(min_rank, max_rank))

        cum_energy = torch.cumsum(s_squared, dim=0) / total_energy
        idx = torch.where(cum_energy >= self.energy_threshold)[0]
        r = int(idx[0].item()) + 1 if len(idx) > 0 else max_rank
        return max(1, min(max(r, min_rank), max_rank))

    def _init_svd_part(
        self,
        adap_linear: AdaptiveSVDLinear,
        W_data: torch.Tensor,
        bias_data: Optional[torch.Tensor],
    ) -> None:
        with torch.no_grad():
            W = W_data.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            max_r = min(adap_linear.svd_rank, U.size(1), Vt.size(0))
            r = self._compute_energy_rank(S, max_rank=max_r)

            sqrt_s = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_r = (U[:, :r] * sqrt_s.unsqueeze(0)).to(dtype=adap_linear.U.dtype, device=adap_linear.U.device)
            V_r = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(dtype=adap_linear.V.dtype, device=adap_linear.V.device)

            adap_linear.set_svd_factors(U_r, V_r)
            if adap_linear.bias is not None and bias_data is not None:
                adap_linear.bias.data.copy_(bias_data)


    def copy_weights(self, src: Any) -> None:
        """Decompose and copy weights from any HuggingFace or PyTorch MHA module with fallback."""
        from framework.core.adapters import UniversalAdapter
        
        # Test if we can extract weights
        extracted_all = True
        extracted_weights = {}

        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            w, b = None, None
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
            except Exception:
                pass

            if w is None:
                # Manual fallback for GPT-2 Fused c_attn Conv1D
                if hasattr(src, "c_attn") and proj in ["q", "k", "v"]:
                    mod_c = src.c_attn
                    base_c = getattr(mod_c, "base_layer", mod_c)
                    weight = getattr(base_c, "weight", getattr(mod_c, "weight", None))
                    if weight is not None:
                        weight = weight.data.clone()
                        bias = getattr(base_c, "bias", getattr(mod_c, "bias", None))
                        bias_data = bias.data.clone() if bias is not None else None
                        hidden_dim = getattr(self, "hidden_size", 768)
                        if (
                            mod_c.__class__.__name__ == "Conv1D"
                            or base_c.__class__.__name__ == "Conv1D"
                            or hasattr(mod_c, "nf")
                            or hasattr(base_c, "nf")
                            or (weight.dim() == 2 and weight.shape[1] == 3 * hidden_dim)
                            or (weight.dim() == 2 and weight.shape[0] < weight.shape[1])
                        ):
                            weight = weight.t().contiguous()
                        split_size = weight.shape[0] // 3
                        idx = {"q": 0, "k": 1, "v": 2}[proj]
                        w = weight[idx * split_size : (idx + 1) * split_size, :].contiguous()
                        b = bias_data[idx * split_size : (idx + 1) * split_size].contiguous() if bias_data is not None else None
                elif hasattr(src, "c_proj") and proj == "o":
                    mod_o = src.c_proj
                    base_o = getattr(mod_o, "base_layer", mod_o)
                    w_o = getattr(base_o, "weight", getattr(mod_o, "weight", None))
                    if w_o is not None:
                        w_o = w_o.data.clone()
                        hidden_dim = getattr(self, "hidden_size", 768)
                        if (
                            mod_o.__class__.__name__ == "Conv1D"
                            or base_o.__class__.__name__ == "Conv1D"
                            or hasattr(mod_o, "nf")
                            or hasattr(base_o, "nf")
                            or (w_o.dim() == 2 and w_o.shape[0] != hidden_dim and w_o.shape[1] == hidden_dim)
                            or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                        ):
                            w_o = w_o.t().contiguous()
                        b_o = getattr(base_o, "bias", getattr(mod_o, "bias", None))
                        w = w_o.contiguous()
                        b = b_o.data.clone().contiguous() if b_o is not None else None
                # Manual fallback for RoBERTa / BERT self-attention
                elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
                    s = getattr(src, "self")
                    name_map = {"q": "query", "k": "key", "v": "value"}
                    if proj in name_map:
                        mod = getattr(s, name_map[proj])
                        w = mod.weight
                        b = getattr(mod, "bias", None)
                elif hasattr(src, "output") and hasattr(getattr(src, "output"), "dense") and proj == "o":
                    mod = src.output.dense
                    w = mod.weight
                    b = getattr(mod, "bias", None)
                elif hasattr(src, "query") and proj == "q":
                    w = src.query.weight
                    b = getattr(src.query, "bias", None)
                elif hasattr(src, "key") and proj == "k":
                    w = src.key.weight
                    b = getattr(src.key, "bias", None)
                elif hasattr(src, "value") and proj == "v":
                    w = src.value.weight
                    b = getattr(src.value, "bias", None)
                # DistilBERT manual fallback
                elif hasattr(src, "q_lin") and proj == "q":
                    w = src.q_lin.weight
                    b = getattr(src.q_lin, "bias", None)
                elif hasattr(src, "k_lin") and proj == "k":
                    w = src.k_lin.weight
                    b = getattr(src.k_lin, "bias", None)
                elif hasattr(src, "v_lin") and proj == "v":
                    w = src.v_lin.weight
                    b = getattr(src.v_lin, "bias", None)
                elif hasattr(src, "out_lin") and proj == "o":
                    w = src.out_lin.weight
                    b = getattr(src.out_lin, "bias", None)
                elif hasattr(src, f"{proj}_proj"):
                    mod = getattr(src, f"{proj}_proj")
                    w = mod.weight
                    b = getattr(mod, "bias", None)

            extracted_weights[proj] = (w, b)
            if w is None:
                extracted_all = False

        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj")
        )
        is_nested_encoder = False
        if (
            hasattr(src, "self")
            and hasattr(getattr(src, "self"), "query")
            and hasattr(src, "output")
            and hasattr(getattr(src, "output"), "dense")
        ):
            is_nested_encoder = True
        elif (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj:
            is_nested_encoder = True

        if not extracted_all:
            # If it's a nested encoder, it's okay to miss 'o'
            if is_nested_encoder and extracted_weights["q"][0] is not None and extracted_weights["k"][0] is not None and extracted_weights["v"][0] is not None:
                extracted_all = True

        if not extracted_all:
            raise RuntimeError(
                f"AdaptiveSVDAttention.copy_weights: could not extract mandatory projections "
                f"{[p for p, (w, _) in extracted_weights.items() if w is None]} from {cls_name}. "
                f"Refusing to install randomly-initialised weights."
            )

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted_weights[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(f"AdaptiveSVDAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted_weights["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted_weights["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(f"AdaptiveSVDAttention extracted weight for o has near-zero Frobenius norm ({norm_o.item():.6f})")

        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            w, b = extracted_weights[proj]
            if w is not None:
                self._init_svd_part(target, w, b)
                target.U.requires_grad = False
                target.V.requires_grad = False
                if target.bias is not None:
                    target.bias.requires_grad = False
                if getattr(target, "lora_A", None) is not None:
                    target.lora_A.requires_grad = True
                if getattr(target, "lora_B", None) is not None:
                    target.lora_B.requires_grad = True

        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if hasattr(src, "relative_attention_bias") and src.relative_attention_bias is not None:
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

        self.is_nested_encoder = is_nested_encoder

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of Adaptive SVD Attention with universal T5, GQA, RoPE & masking support."""
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N_q, _ = hidden_states.shape
        N_k = kv_input.shape[1]
        H, D = self.num_heads, self.head_dim

        Q = self.query(hidden_states).view(B, N_q, H, D).transpose(1, 2).contiguous()
        K = self.key(kv_input).view(B, N_k, self.num_kv_heads, D).transpose(1, 2).contiguous()
        V = self.value(kv_input).view(B, N_k, self.num_kv_heads, D).transpose(1, 2).contiguous()

        # Apply RoPE (Direct kwargs support for Qwen3.5/Mistral/LLaMA)
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            def _rotate_half(x):
                _d = x.shape[-1]
                _h = _d // 2
                _rot = torch.cat((-x[..., _h : 2 * _h], x[..., :_h]), dim=-1)
                if _d % 2 == 1:
                    _rot = torch.cat([_rot, x[..., -1:]], dim=-1)
                return _rot
            if cos.dim() == 3:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            while cos.dim() < Q.dim():
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            if cos.shape[-2] != Q.shape[-2]:
                cos, sin = cos[..., :Q.shape[-2], :], sin[..., :Q.shape[-2], :]
            rotary_dim = cos.shape[-1]
            if rotary_dim <= Q.shape[-1]:
                q_rot, q_pass = Q[..., :rotary_dim], Q[..., rotary_dim:]
                k_rot, k_pass = K[..., :rotary_dim], K[..., rotary_dim:]
                q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
                k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
                Q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
                K = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
            else:
                cos_cut, sin_cut = cos[..., :Q.shape[-1]], sin[..., :Q.shape[-1]]
                Q = (Q * cos_cut.to(Q.dtype)) + (_rotate_half(Q) * sin_cut.to(Q.dtype))
                K = (K * cos_cut.to(K.dtype)) + (_rotate_half(K) * sin_cut.to(K.dtype))

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = D**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale  # [B, H, N_q, N_k]

        # Handle T5 position bias
        position_bias = kwargs.get("position_bias", None)
        if position_bias is None and self.has_relative_attention_bias:
            position_bias = self.compute_bias(N_q, N_k, device=hidden_states.device)

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if position_bias is not None:
            if mask is not None and mask.dim() == 4:
                position_bias = position_bias + mask.to(scores.dtype)
            scores = scores + position_bias.to(scores.dtype)
        elif mask is not None:
            scores = apply_attention_mask(scores, mask)

        # Explicit Causal Masking for autoregressive decoders
        _cfg = self.config
        _mt = str(getattr(_cfg, "model_type", "") or "").lower() if _cfg else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                             "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        is_causal = (
            self.is_decoder
            or (_cfg is not None and getattr(_cfg, "is_decoder", False))
            or (_cfg is not None and getattr(_cfg, "is_causal", False))
            or (_cfg is not None and getattr(_cfg, "causal", False))
            or any(_mt.startswith(p) for p in _decoder_prefixes)
        )
        if is_causal:
            causal_tril = torch.tril(torch.ones((N_q, N_k), device=hidden_states.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_tril, -10000.0)

        attn_weights = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(hidden_states.dtype)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V.to(attn_weights.dtype))
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        # In nested encoders (BERT/RoBERTa .self), context passes through directly because output.dense applies after
        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        output = output.to(hidden_states.dtype)
        if position_bias is not None:
            return output, position_bias
        return output, attn_weights if output_attentions else None



@ATTENTION_REGISTRY.register("adaptive_svd_attention_inject")
@ATTENTION_REGISTRY.register("adaptive_svd_inject")
def inject_adaptive_svd(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[AdaptiveSVDAttention, nn.Module]]:
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # Whole model traversal
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
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_adaptive_svd(
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

    # Single layer injection
    if old_attn is None and target is not None:
        for attr in ("attention", "self_attn", "self", "attn"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if cand is not None:
                    if hasattr(cand, "self"):
                        old_attn = cand.self
                    else:
                        old_attn = cand
                    break
    if old_attn is None and layer is not None:
        for attr in ("attention", "self_attn", "self", "attn"):
            if hasattr(layer, attr):
                cand = getattr(layer, attr)
                if cand is not None:
                    if hasattr(cand, "self"):
                        old_attn = cand.self
                    else:
                        old_attn = cand
                    break

    if model_config is None and target is not None:
        model_config = getattr(target, "config", None)
    if model_config is None and old_attn is not None:
        model_config = getattr(old_attn, "config", None)
    if model_config is None:
        hidden_size = getattr(target, "hidden_size", getattr(old_attn, "hidden_size", None))
        if not hidden_size and old_attn is not None:
            for attr in ("query", "q_proj", "q_lin", "c_attn"):
                mod = getattr(old_attn, attr, None)
                if mod is not None and hasattr(mod, "in_features"):
                    hidden_size = mod.in_features
                    break
        if not hidden_size and old_attn is not None:
            hidden_size = getattr(old_attn, "all_head_size", getattr(old_attn, "embed_dim", None))
        hidden_size = int(hidden_size or 768)

        num_heads = (
            getattr(target, "num_attention_heads", None)
            or getattr(target, "num_heads", None)
            or getattr(old_attn, "num_attention_heads", None)
            or getattr(old_attn, "num_heads", None)
            or getattr(old_attn, "n_heads", None)
            or 12
        )
        num_heads = int(num_heads or 12)
        head_dim = getattr(target, "head_dim", getattr(old_attn, "head_dim", hidden_size // max(1, num_heads)))
        model_config = type(
            "Config",
            (),
            {
                "hidden_size": hidden_size,
                "num_attention_heads": num_heads,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "num_key_value_heads": num_heads,
                "rank": 32,
                "svd_rank": 32,
                "residual_rank": 16,
                "energy_threshold": 0.92,
                "dropout": 0.1,
            },
        )()

    # Compute hidden_size from model_config (needed as fallback even when model_config is provided)
    _h_size_fallback = (
        getattr(model_config, "hidden_size", None)
        or getattr(model_config, "d_model", None)
        or getattr(model_config, "n_embd", None)
        or getattr(model_config, "dim", 768)
    ) if model_config is not None else 768

    try:
        svd_rank = getattr(
            comp_config,
            "svd_rank",
            getattr(
                comp_config,
                "rank",
                getattr(model_config, "svd_rank", getattr(model_config, "rank", kwargs.pop("rank", _h_size_fallback))),
            ),
        )
        residual_rank = getattr(comp_config, "residual_rank", getattr(model_config, "residual_rank", kwargs.get("residual_rank", 16)))
        energy_threshold = getattr(comp_config, "energy_threshold", getattr(model_config, "energy_threshold", kwargs.get("energy_threshold", 0.9999)))

        try:
            test_module = AdaptiveSVDAttention(
                config=model_config,
                svd_rank=svd_rank,
                residual_rank=residual_rank,
                energy_threshold=energy_threshold,
                pretrained_attn=old_attn,
                **kwargs,
            )
        except Exception:
            return None

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    test_module.is_nested_encoder = True
                    target.attention.self = test_module
                else:
                    test_module.is_nested_encoder = False
                    target.attention = test_module
            elif hasattr(target, "self_attn"):
                test_module.is_nested_encoder = False
                test_module.is_decoder = True
                test_module.is_causal = True
                target.self_attn = test_module
            elif hasattr(target, "attn"):
                test_module.is_nested_encoder = False
                test_module.is_decoder = True
                test_module.is_causal = True
                target.attn = test_module
            elif hasattr(target, "self"):
                test_module.is_nested_encoder = True
                target.self = test_module

        if old_attn is not None:
            test_module.train(old_attn.training)
        elif target is not None and hasattr(target, 'training'):
            test_module.train(target.training)

        return test_module
    except Exception as _inject_exc:
        import traceback as _tb
        logger.debug("inject_adaptive_svd failed: %s\n%s", _inject_exc, _tb.format_exc())
        return None




