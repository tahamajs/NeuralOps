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


class AdaptiveSVDLinear(nn.Module):
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
                if self.lora_B.dtype in (torch.float16, torch.bfloat16):
                    b_fp32 = torch.empty_like(self.lora_B, dtype=torch.float32, device="cpu")
                    nn.init.kaiming_uniform_(b_fp32, a=math.sqrt(5))
                    self.lora_B.copy_(b_fp32.to(dtype=self.lora_B.dtype, device=self.lora_B.device))
                else:
                    nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
                nn.init.zeros_(self.lora_A)

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
            if getattr(self, "lora_A", None) is not None:
                nn.init.zeros_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        W = torch.matmul(self.U, self.V)
        if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
            W = W + torch.matmul(self.lora_B, self.lora_A)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward computation for low-rank SVD + trainable residual adapter."""
        Vx = F.linear(x, self.V)
        out = F.linear(Vx, self.U)
        if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
            Ax = F.linear(x, self.lora_A)
            out = out + F.linear(Ax, self.lora_B)
        if self.bias is not None:
            out = out + self.bias
        return out


@ATTENTION_REGISTRY.register("adaptive_svd_attn", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("adaptive_svd", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("adaptive_svd_attention", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class AdaptiveSVDAttention(BaseAttention):
    r"""Multi-Head Attention with Adaptive SVD + Residual factorized projections."""

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
        super().__init__(config)
        self.adapter = ModelAdapter(config)
        self.num_heads: int = self.adapter.num_heads
        self.num_kv_heads: int = self.adapter.num_key_value_heads
        self.hidden_size: int = self.adapter.hidden_size
        self.head_dim: int = self.adapter.head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = self.adapter.kv_repeat
        self.supports_gqa: bool = True
        self.svd_rank: int = getattr(
            config,
            "svd_rank",
            svd_rank if svd_rank is not None else getattr(config, "rank", 32),
        )
        self.residual_rank: int = getattr(config, "residual_rank", residual_rank)
        self.energy_threshold: float = float(
            getattr(
                config,
                "energy_threshold",
                kwargs.get("energy_threshold", energy_threshold),
            )
        )

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
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        self.is_decoder: bool = getattr(config, "is_decoder", False)
        self.is_nested_encoder: bool = False
        self.has_relative_attention_bias: bool = False
        self.relative_attention_bias: Optional[nn.Embedding] = None
        self.relative_attention_num_buckets: int = getattr(config, "relative_attention_num_buckets", 32)
        self.relative_attention_max_distance: int = getattr(config, "relative_attention_max_distance", 128)

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
                    fused_w = src.c_attn.weight.T  # [3 * D, D]
                    fused_b = getattr(src.c_attn, "bias", None)
                    split_size = fused_w.size(0) // 3
                    idx = {"q": 0, "k": 1, "v": 2}[proj]
                    w = fused_w[idx * split_size : (idx + 1) * split_size, :]
                    b = fused_b[idx * split_size : (idx + 1) * split_size] if fused_b is not None else None
                elif hasattr(src, "c_proj") and proj == "o":
                    w = src.c_proj.weight.T
                    b = getattr(src.c_proj, "bias", None)
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
                elif hasattr(src, f"{proj}_proj"):
                    mod = getattr(src, f"{proj}_proj")
                    w = mod.weight
                    b = getattr(mod, "bias", None)

            extracted_weights[proj] = (w, b)
            if w is None:
                extracted_all = False

        if not extracted_all:
            # Check if this is a nested encoder which only has q, k, v (o is outside)
            cls_name = src.__class__.__name__
            has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
            is_nested_encoder = (cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (cls_name.endswith("SelfAttention") and not has_out_proj)) and not has_out_proj
            
            # If it's a nested encoder, it's okay to miss 'o'
            if is_nested_encoder and extracted_weights["q"][0] is not None and extracted_weights["k"][0] is not None and extracted_weights["v"][0] is not None:
                extracted_all = True
                
        if not extracted_all:
            raise RuntimeError(f"Could not extract all required weights for AdaptiveSVDAttention. Missing: {[p for p, (w, _) in extracted_weights.items() if w is None]}")

        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            w, b = extracted_weights[proj]
            if w is not None:
                self._init_svd_part(target, w, b)
                # Sanity check: ensure reconstructed weight isn't garbage
                reconstructed = (target.U @ target.V).detach()
                error = (torch.norm(w.to(reconstructed.device).float() - reconstructed.float()) / (torch.norm(w.float()) + 1e-8)).item()
                if error > 0.05:
                    print(f"CRITICAL WARNING: High SVD reconstruction error ({error:.2f}) for {proj} exceeds 5% threshold.")

        if hasattr(src, "has_relative_attention_bias"):
            self.has_relative_attention_bias = bool(src.has_relative_attention_bias)
        if hasattr(src, "relative_attention_bias") and src.relative_attention_bias is not None:
            self.relative_attention_bias = src.relative_attention_bias
            self.has_relative_attention_bias = True
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

        # Safely detect if this module is replacing an inner encoder self-attention (BERT/RoBERTa)
        # where the outer layer already applies output.dense (avoiding double output projection).
        # We verify the absence of own output projections so we never misflag DistilBERT (out_lin),
        # GPT-2 (c_proj), LLaMA/Qwen (o_proj), or T5 (o).
        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj")
        )
        if (cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (cls_name.endswith("SelfAttention") and not has_out_proj)) and not has_out_proj:
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

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

        Q = self.query(hidden_states).view(B, N_q, H, D).transpose(1, 2)
        K = self.key(kv_input).view(B, N_k, self.num_kv_heads, D).transpose(1, 2)
        V = self.value(kv_input).view(B, N_k, self.num_kv_heads, D).transpose(1, 2)

        # Apply RoPE (Direct kwargs support for Qwen3.5/Mistral/LLaMA)
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            def _rotate_half(x):
                return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
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
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N_q, N_k]

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
        if self.is_decoder or getattr(self.config, "is_decoder", False) or getattr(self.config, "model_type", "") == "gpt2":
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

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N_q, self.all_head_size)

        # In nested encoders (BERT/RoBERTa .self), context passes through directly because output.dense applies after
        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        if position_bias is not None:
            return output, position_bias
        return output, attn_weights if output_attentions else None



@ATTENTION_REGISTRY.register("adaptive_svd_attention_inject")
@ATTENTION_REGISTRY.register("adaptive_svd_inject")
def inject_adaptive_svd(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
    **kwargs: Any,
) -> Optional[AdaptiveSVDAttention]:
    try:
        svd_rank = getattr(comp_config, "svd_rank", getattr(comp_config, "rank", getattr(model_config, "rank", 32)))
        residual_rank = getattr(comp_config, "residual_rank", 16)
        energy_threshold = getattr(comp_config, "energy_threshold", 0.92)
        test_module = AdaptiveSVDAttention(
            config=model_config,
            svd_rank=svd_rank,
            residual_rank=residual_rank,
            energy_threshold=energy_threshold,
            pretrained_attn=old_attn,
        )
        layer.attention = test_module
        return test_module
    except RuntimeError:
        return None



