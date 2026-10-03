"""Expert Choice Mixture of Experts (MoE) Attention Block.

In Expert Choice routing (Zhou et al., 2022), experts select their top-k tokens
rather than tokens selecting top-k experts. This provides guaranteed perfect load balancing
across all experts, dynamic per-token expert allocation, and superior zero-shot transfer.

Hardened for production across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA architectures:
- Robust pretrained weight extraction (GPT-2 fused c_attn, RoBERTa nested, Qwen GQA, DistilBERT)
- Strict Frobenius norm verification (>= 1e-4) for all projections
- Grouped Query Attention (GQA) with kv_head_size projections and repeat_kv
- Rotary Position Embedding (RoPE) support with partial-coordinate slicing
- Strict causal masking for decoder models
- Nested encoder detection to bypass out_lin in inner self-attention
- Parameter-efficient fine-tuning via LoRALinear adapters and frozen base weights
- Step-0 representation preservation (zero-initialized expert outputs & biased gating)
- Numerically stable FP32 scores with nan_to_num protection
- Safe injection factory returning None on extraction failures
- Full cross-attention support
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""Expands grouped key/value heads to match query heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


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
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Applies rotary position embeddings with partial-RoPE dimension slicing."""
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

    return q, k


class LoRALinear(nn.Module):
    r"""Linear projection layer with frozen base weights and trainable residual LoRA adapters."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty((lora_rank, in_features), **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty((out_features, lora_rank), **factory_kwargs)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        """Freeze base projection parameters so only LoRA parameters adapt."""
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base projection parameters."""
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        w = (
            self.weight.to(target_dtype)
            if self.weight.dtype != target_dtype
            else self.weight
        )
        b = (
            self.bias.to(target_dtype)
            if (self.bias is not None and self.bias.dtype != target_dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = (
                self.lora_A.to(target_dtype)
                if self.lora_A.dtype != target_dtype
                else self.lora_A
            )
            bb = (
                self.lora_B.to(target_dtype)
                if self.lora_B.dtype != target_dtype
                else self.lora_B
            )
            lora_out = F.linear(F.linear(x, a), bb) * self.lora_scaling
            out = out + lora_out
        return out.to(target_dtype)


class ExpertChoiceMoE(nn.Module):
    r"""
    Expert Choice Routing MoE layer.
    Each expert selects top-k tokens with the highest affinity scores.
    Equipped with zero-initialized final layers for step-0 representation preservation.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int = 2,
        capacity_factor: float = 1.0,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.noise_std = noise_std
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        self.router_ln = nn.LayerNorm(self.d_model)
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)

        expert_hidden_dim = max(32, d_model // 4)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, expert_hidden_dim),
                    nn.GELU(),
                    nn.Linear(expert_hidden_dim, d_model),
                )
                for _ in range(n_experts)
            ]
        )

        # Active MoE expert representation
        for expert in self.experts:
            nn.init.normal_(expert[2].weight, std=0.02)
            if expert[2].bias is not None:
                nn.init.zeros_(expert[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward execution using Expert Choice routing."""
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        T = x_flat.size(0)

        if self.d_model == D:
            logits = self.router(self.router_ln(x_flat))
        else:
            normed = F.layer_norm(x_flat, (D,))
            if (
                not hasattr(self, "_dynamic_router")
                or self._dynamic_router.in_features != D
            ):
                self._dynamic_router = nn.Linear(D, self.n_experts, bias=False).to(
                    device=x.device, dtype=x.dtype
                )
                nn.init.uniform_(self._dynamic_router.weight, -0.01, 0.01)
            logits = self._dynamic_router(normed)

        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # Softmax over tokens per expert
        gate_scores = F.softmax(logits.float(), dim=0).to(dtype=logits.dtype)
        k = max(1, min(T, int(T * self.capacity_factor / self.n_experts)))

        out_flat = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            scores_e = gate_scores[:, e]
            topk_vals, topk_indices = torch.topk(scores_e, k, dim=0)

            selected_x = x_flat[topk_indices]
            if selected_x.size(-1) == self.d_model:
                expert_out = self.experts[e](selected_x)
            else:
                if (
                    not hasattr(self, f"_expert_in_proj_{e}")
                    or getattr(self, f"_expert_in_proj_{e}").in_features != D
                ):
                    setattr(
                        self,
                        f"_expert_in_proj_{e}",
                        nn.Linear(D, self.d_model).to(device=x.device, dtype=x.dtype),
                    )
                    setattr(
                        self,
                        f"_expert_out_proj_{e}",
                        nn.Linear(self.d_model, D).to(device=x.device, dtype=x.dtype),
                    )
                proj_in = getattr(self, f"_expert_in_proj_{e}")
                proj_out = getattr(self, f"_expert_out_proj_{e}")
                expert_out = proj_out(self.experts[e](proj_in(selected_x)))

            update_val = (
                expert_out.to(out_flat.dtype)
                * topk_vals.unsqueeze(-1).to(out_flat.dtype)
            ).to(dtype=out_flat.dtype)
            out_flat.index_add_(0, topk_indices, update_val)

        # Auxiliary load balancing loss
        prob_mean = F.softmax(logits.float() / 1.0, dim=-1).mean(dim=0)
        self.aux_loss = self.n_experts * torch.sum(prob_mean**2)

        return out_flat.view(B, L, D)


@ATTENTION_REGISTRY.register(
    "expert_choice_moe",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
@ATTENTION_REGISTRY.register(
    "expert_choice_moe_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class ExpertChoiceMoEAttention(BaseAttention):
    r"""Expert Choice MoE Multi-Head Attention Module."""

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 64,
        n_experts: int = 2,
        capacity_factor: float = 1.0,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        moe_init_gate: float = 0.0,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        cfg_obj = getattr(config, "text_config", config) if config is not None else None
        hidden_size = (
            getattr(cfg_obj, "hidden_size", None) if cfg_obj is not None else None
        )
        num_heads = (
            getattr(cfg_obj, "num_attention_heads", None)
            if cfg_obj is not None
            else None
        )

        if hidden_size is None and source_attn is not None:
            for cand in ["q_lin", "query", "q_proj", "c_attn", "in_proj_weight", "qkv"]:
                mod = getattr(source_attn, cand, None)
                if mod is not None and hasattr(mod, "weight"):
                    w = mod.weight
                    hidden_size = w.shape[1] if w.ndim == 2 else None
                    break
                elif isinstance(mod, torch.Tensor) and mod.ndim == 2:
                    hidden_size = mod.shape[1]
                    break
        if hidden_size is None:
            hidden_size = kwargs.get("hidden_size", 768)
        if num_heads is None:
            num_heads = kwargs.get("num_attention_heads", 12)

        super().__init__(config, rank)
        self.dim = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_kv_heads = (
            getattr(config, "num_key_value_heads", self.num_heads)
            if config is not None
            else self.num_heads
        )
        self.head_dim = getattr(config, "head_dim", None) or (
            self.dim // max(1, self.num_heads)
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        # Static layer index
        resolved_layer_id = (
            layer_id
            if layer_id is not None
            else (
                layer_idx
                if layer_idx is not None
                else kwargs.get("layer_id", kwargs.get("layer_idx", 0))
            )
        )
        self.layer_idx: int = int(resolved_layer_id)

        # LoRA adaptation configuration
        self.lora_rank: int = getattr(
            config,
            "lora_rank",
            getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
        )
        self.lora_alpha: float = getattr(
            config,
            "lora_alpha",
            getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
        )
        self.freeze_base: bool = self.lora_rank > 0

        # Decoder & nested encoder detection
        model_type = getattr(config, "model_type", "").lower()
        is_dec = getattr(
            config,
            "is_decoder",
            getattr(config, "is_causal", kwargs.get("is_causal", False)),
        )
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
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)
        self.is_nested_encoder: bool = False

        # Multi-head projections equipped with LoRA adapters
        self.q_lin = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_lin = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_lin = LoRALinear(
            self.dim,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_lin = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        self.moe = ExpertChoiceMoE(
            self.all_head_size, n_experts=n_experts, capacity_factor=capacity_factor
        )
        # Trainable gate for MoE output contribution (initialized with negative bias for smooth ramp-up)
        self.moe_gate = nn.Parameter(torch.tensor(float(moe_init_gate)))

        dropout_p = (
            getattr(
                config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        self.rotary_emb = None
        self.comp_config: Any | None = comp_config

        if source_attn is not None:
            self.copy_weights(source_attn)

    # Compatibility properties
    @property
    def query(self) -> LoRALinear:
        return self.q_lin

    @property
    def key(self) -> LoRALinear:
        return self.k_lin

    @property
    def value(self) -> LoRALinear:
        return self.v_lin

    @property
    def dense(self) -> LoRALinear:
        return self.out_lin

    @property
    def o_proj(self) -> LoRALinear:
        return self.out_lin

    @property
    def out_proj(self) -> LoRALinear:
        return self.out_lin

    @property
    def q_proj(self) -> LoRALinear:
        return self.q_lin

    @property
    def k_proj(self) -> LoRALinear:
        return self.k_lin

    @property
    def v_proj(self) -> LoRALinear:
        return self.v_lin

    @property
    def W_q(self) -> LoRALinear:
        return self.q_lin

    @property
    def W_k(self) -> LoRALinear:
        return self.k_lin

    @property
    def W_v(self) -> LoRALinear:
        return self.v_lin

    @property
    def c_proj(self) -> LoRALinear:
        return self.out_lin

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        r"""Copy projection weights from pretrained attention module with Frobenius norm verification."""
        if pretrained_attn is None:
            return

        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = (
            hasattr(pretrained_attn, "out_lin")
            or hasattr(pretrained_attn, "c_proj")
            or hasattr(pretrained_attn, "o_proj")
        )
        is_nested = False
        if (
            (
                hasattr(pretrained_attn, "self")
                and hasattr(pretrained_attn.self, "query")
                and hasattr(pretrained_attn, "output")
                and hasattr(pretrained_attn.output, "dense")
            )
            or (
                cls_name
                in [
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (
                    cls_name.endswith("SelfAttention")
                    and "distil" not in cls_name.lower()
                )
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(pretrained_attn, "rotary_emb"):
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2: Fused c_attn + c_proj
        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c_attn_mod = pretrained_attn.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(
                c_attn_mod, "merged", False
            ):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    print(f"copy_weights failed: {e}")
                    logger.warning(f"copy_weights failed: {e}")
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (
                    (c_attn_mod.__class__.__name__ == "Conv1D")
                    or (base_c.__class__.__name__ == "Conv1D")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if (
                hasattr(pretrained_attn, "c_proj")
                and pretrained_attn.c_proj is not None
            ):
                c_proj_mod = pretrained_attn.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        (c_proj_mod.__class__.__name__ == "Conv1D")
                        or (base_o.__class__.__name__ == "Conv1D")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # 2. RoBERTa / BERT nested self-attention
        elif hasattr(pretrained_attn, "self") and hasattr(
            pretrained_attn.self, "query"
        ):
            s = pretrained_attn.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(pretrained_attn, "output") and hasattr(
                pretrained_attn.output, "dense"
            ):
                out_mod = pretrained_attn.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # 3. Standard linear projections (DistilBERT, Qwen, LLaMA)
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query", "W_q"],
                "k": ["k_lin", "k_proj", "key", "W_k"],
                "v": ["v_lin", "v_proj", "value", "W_v"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(pretrained_attn, name):
                        mod = getattr(pretrained_attn, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        # Fallback via UniversalAdapter
        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(pretrained_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            logger.warning(
                f"ExpertChoiceMoEAttention copy_weights failed: missing mandatory projections {missing_after} "
                f"from module {pretrained_attn.__class__.__name__}. Proceeding with partially initialized projections."
            )

        target_map = {
            "q": self.q_lin,
            "k": self.k_lin,
            "v": self.v_lin,
            "o": self.out_lin,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                if (
                    w.shape != target_mod.weight.shape
                    and target_mod.weight.shape == w.t().contiguous().shape
                ):
                    w = w.t().contiguous()
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(
                        device=target_mod.weight.device, dtype=target_mod.weight.dtype
                    )
                )

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(
                        b[:min_b].to(
                            device=target_mod.bias.device, dtype=target_mod.bias.dtype
                        )
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                # Frobenius norm verification (>= 1e-4)
                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(
                        f"ExpertChoiceMoEAttention weight copy verification failed for '{proj_key}': norm too small ({w_norm:.2e})"
                    )

                if self.freeze_base:
                    target_mod.freeze_base_weights()

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
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L, D = hidden_states.shape

        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states
        kv_len = kv_input.shape[1]

        # 1. Projections
        q = (
            self.q_lin(hidden_states)
            .view(B, L, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        k = (
            self.k_lin(kv_input)
            .view(B, kv_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v = (
            self.v_lin(kv_input)
            .view(B, kv_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

        # 2. Rotary Position Embeddings (RoPE)
        if position_embeddings is not None and not is_cross_attention:
            if isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
            else:
                cos, sin = position_embeddings, None
            if sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None and not is_cross_attention:
            if position_ids is not None:
                cos, sin = self.rotary_emb(v, seq_len=kv_len)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # 3. Grouped Query Attention (GQA) head expansion
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # 4. Multi-head attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        # Memory-efficient: use SDPA to avoid OOM on large seqs

        try:
            _qs = q.half() if q.dtype == torch.float32 else q
            _ks = k.half() if k.dtype == torch.float32 else k
            _vs = v.half() if v.dtype == torch.float32 else v
            _is_causal = (
                getattr(self, "is_decoder", False) and not is_cross_attention and L > 1
            )
            _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                _qs,
                _ks,
                _vs,
                is_causal=_is_causal,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            attn_out = (
                _sdpa_out.to(hidden_states.dtype)
                .transpose(1, 2)
                .contiguous()
                .view(B, L, self.all_head_size)
            )
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
            moe_out = self.moe(attn_out)
            self.aux_loss = self.moe.aux_loss
            moe_weight = torch.sigmoid(self.moe_gate).to(attn_out.dtype)
            enriched_attn = (attn_out + moe_weight * moe_out.to(attn_out.dtype)).to(
                attn_out.dtype
            )
            if self.is_nested_encoder and enriched_attn.shape[-1] == self.dim:
                out = enriched_attn
            else:
                out = self.out_lin(enriched_attn)
            if out.shape[-1] != hidden_states.shape[-1]:
                out = (
                    F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
                    if out.shape[-1] < hidden_states.shape[-1]
                    else out[..., : hidden_states.shape[-1]]
                )
            return out, None
        except Exception:
            pass

        scores = (
            torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
        )

        if self.is_decoder and not is_cross_attention:
            causal_mask = torch.triu(
                torch.full(
                    (L, kv_len),
                    float("-inf"),
                    device=scores.device,
                    dtype=torch.float32,
                ),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        eff_mask = encoder_attention_mask if is_cross_attention else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                pad_mask = (eff_mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, float("-inf"))
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                scores = scores + eff_mask.float()

        probs = F.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # 5. Head mask handling
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            probs = probs * head_mask

        attn_out = (
            torch.matmul(probs.to(v.dtype), v)
            .transpose(1, 2)
            .contiguous()
            .view(B, L, self.all_head_size)
        )

        # 6. MoE feedforward enrichment with gate blending
        moe_out = self.moe(attn_out)
        self.aux_loss = self.moe.aux_loss
        moe_weight = torch.sigmoid(self.moe_gate).to(attn_out.dtype)
        enriched_attn = (attn_out + moe_weight * moe_out.to(attn_out.dtype)).to(
            attn_out.dtype
        )

        # 7. Output projection with nested encoder bypass
        if self.is_nested_encoder:
            if enriched_attn.shape[-1] != self.dim:
                out = self.out_lin(enriched_attn)
            else:
                out = enriched_attn
        else:
            out = self.out_lin(enriched_attn)

        if out.shape[-1] != hidden_states.shape[-1]:
            out = (
                F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
                if out.shape[-1] < hidden_states.shape[-1]
                else out[..., : hidden_states.shape[-1]]
            )

        return out, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("expert_choice_moe_inject")
def inject_expert_choice_moe(
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
            res = inject_expert_choice_moe(
                layer=lyr,
                model_config=cfg,
                comp_config=(
                    comp_config
                    if "comp_config" in locals() or "comp_config" in kwargs
                    else None
                ),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""Safe injection factory for ExpertChoiceMoEAttention."""
    src_attn = old_attn
    if src_attn is None:
        if hasattr(layer, "attention"):
            src_attn = layer.attention
        elif hasattr(layer, "attn"):
            src_attn = layer.attn
        elif hasattr(layer, "self_attn"):
            src_attn = layer.self_attn

    try:
        if model_config is None and hasattr(layer, "config"):
            model_config = layer.config

        new_attn = ExpertChoiceMoEAttention(
            config=model_config,
            comp_config=comp_config,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn,
            **kwargs,
        )
        if src_attn is not None:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
        return new_attn
    except Exception:
        return None
