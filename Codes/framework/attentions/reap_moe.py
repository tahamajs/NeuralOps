"""REAP-MoE: Robust Expert Allocation and Pruning Mixture of Experts Attention Module.

Vectorized Mixture-of-Experts block with load-balancing regularization, SVD low-rank
expert factorization, and multi-head attention weight transfer.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from framework.core.base import BaseAttention
from framework.core.gqa_utils import repeat_kv
from framework.core.registry import ATTENTION_REGISTRY


# ===========================================================================
# RoPE
# ===========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
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
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        cos_q = cos[..., :head_dim].to(q.dtype)
        sin_q = sin[..., :head_dim].to(q.dtype)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        q = (q * cos_q) + (_rotate_half(q) * sin_q)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# Helpers
# ===========================================================================
def _detect_decoder(
    config: Any,
    is_decoder: bool | None,
    old_attn: nn.Module | None,
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(
            (
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
        ):
            return True
    if old_attn is not None:
        if getattr(old_attn, "is_decoder", False):
            return True
        if hasattr(old_attn, "c_attn"):
            return True
        if "gpt" in old_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
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
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

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


# ===========================================================================
# SVDLinear
# ===========================================================================
class SVDLinear(nn.Module):
    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        freeze: bool = False,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank or max(1, min(in_features, out_features) // 4)

        self.U = nn.Parameter(
            torch.randn((out_features, self.rank), **factory_kwargs) * 0.02
        )
        self.S = nn.Parameter(torch.ones(self.rank, **factory_kwargs))
        self.V = nn.Parameter(
            torch.randn((in_features, self.rank), **factory_kwargs) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))

        if freeze:
            self.U.requires_grad = False
            self.S.requires_grad = False
            self.V.requires_grad = False
            self.bias.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        return (self.U * self.S.unsqueeze(0)) @ self.V.t().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = x.dtype
        V = self.V.to(target_dtype) if self.V.dtype != target_dtype else self.V
        S = self.S.to(target_dtype) if self.S.dtype != target_dtype else self.S
        U = self.U.to(target_dtype) if self.U.dtype != target_dtype else self.U
        b = (
            self.bias.to(target_dtype)
            if (self.bias is not None and self.bias.dtype != target_dtype)
            else self.bias
        )
        out = (x @ V) * S @ U.t().contiguous()
        if b is not None:
            out = out + b
        return out.to(target_dtype)


# ===========================================================================
# REAP-MoE block
# ===========================================================================
class REAPMoE(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_experts: int = 2,
        top_k: int = 1,
        noise_std: float = 0.05,
        use_svd: bool = True,
        config: Any | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if config is not None:
            d_model = getattr(
                config, "hidden_size", getattr(config, "d_model", d_model)
            )
            n_experts = getattr(
                config, "num_experts", getattr(config, "n_experts", n_experts)
            )
            top_k = getattr(config, "top_k", top_k)

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.noise_std = noise_std

        self.register_buffer("_aux_loss", torch.tensor(0.0), persistent=False)

        self.router_ln = nn.LayerNorm(self.d_model, **factory_kwargs)
        self.router = nn.Linear(d_model, n_experts, bias=False, **factory_kwargs)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)

        if use_svd:

            def expert_layer():
                return nn.Sequential(
                    SVDLinear(
                        d_model, d_model, rank=max(1, d_model // 8), **factory_kwargs
                    ),
                    nn.GELU(),
                    SVDLinear(
                        d_model, d_model, rank=max(1, d_model // 8), **factory_kwargs
                    ),
                )

        else:

            def expert_layer():
                return nn.Sequential(
                    nn.Linear(d_model, d_model, **factory_kwargs),
                    nn.GELU(),
                    nn.Linear(d_model, d_model, **factory_kwargs),
                )

        self.experts = nn.ModuleList([expert_layer() for _ in range(n_experts)])

        for exp in self.experts:
            final_layer = exp[-1]
            if isinstance(final_layer, SVDLinear):
                nn.init.zeros_(final_layer.U)
                nn.init.zeros_(final_layer.S)
                nn.init.zeros_(final_layer.bias)
            elif isinstance(final_layer, nn.Linear):
                nn.init.zeros_(final_layer.weight)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        target_dtype = x.dtype

        r_dtype = (
            self.router_ln.weight.dtype
            if hasattr(self.router_ln, "weight") and self.router_ln.weight is not None
            else target_dtype
        )
        normed_x = self.router_ln(x.to(r_dtype)).to(target_dtype)
        rw_dtype = (
            self.router.weight.dtype
            if hasattr(self.router, "weight") and self.router.weight is not None
            else target_dtype
        )
        logits = self.router(normed_x.to(rw_dtype)).to(target_dtype)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            scores = logits + noise
        else:
            scores = logits

        full_probs = F.softmax(scores.float(), dim=-1).to(target_dtype)
        topk_weights, topk_indices = torch.topk(full_probs, self.top_k, dim=-1)
        topk_weights = (
            topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        ).to(target_dtype)

        router_probs = F.softmax(logits.float(), dim=-1).view(-1, self.n_experts)
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_experts).float()
        expert_assignment = expert_mask.sum(dim=2).clamp(max=1.0)
        density = expert_assignment.view(-1, self.n_experts).mean(dim=0)
        prob_mean = router_probs.mean(dim=0)
        aux_loss = self.n_experts * torch.sum(density * prob_mean)
        self._aux_loss = aux_loss.detach()

        experts_stack = torch.stack(
            [exp(x).to(target_dtype) for exp in self.experts], dim=2
        )
        topk_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        gathered = torch.gather(experts_stack, 2, topk_expanded)
        out = torch.sum(gathered * topk_weights.unsqueeze(-1), dim=2).to(target_dtype)

        return out, aux_loss


# ===========================================================================
# REAP-MoE Attention
# ===========================================================================
@ATTENTION_REGISTRY.register("reap_moe")
@ATTENTION_REGISTRY.register("reap_moe_attn")
@ATTENTION_REGISTRY.register("reap_moe_attention")
class REAPMoEAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 64,
        n_experts: int = 2,
        top_k: int = 1,
        pretrained_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        pre_ln: bool | None = None,
        comp_config: Any | None = None,  # RP-FIX-21
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank or 64)
        self.config = config
        # If comp_config passed, prefer it; else fall back to config attribute.
        if comp_config is None:
            comp_config = getattr(config, "compression_config", None)
        self.comp_config = comp_config

        # RP-FIX-10: strict hidden_size
        h_size = (
            kwargs.get("hidden_size")
            or getattr(config, "hidden_size", None)
            or getattr(config, "d_model", None)
        )
        if h_size is None and pretrained_attn is not None:
            for cand in ("q_lin", "query", "q_proj", "c_attn"):
                mod = getattr(pretrained_attn, cand, None)
                if mod is not None and hasattr(mod, "weight"):
                    w = mod.weight
                    if w.ndim == 2:
                        if mod.__class__.__name__ == "Conv1D" or hasattr(mod, "nf"):
                            h_size = w.shape[0]
                        else:
                            h_size = w.shape[1]
                        break
        if h_size is None:
            raise ValueError("REAPMoEAttention: could not infer hidden_size.")

        self.dim = int(h_size)
        self.num_heads = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(config, "num_attention_heads", None)
            or getattr(config, "num_heads", None)
            or 12
        )
        self.num_kv_heads = (
            kwargs.get("num_key_value_heads")
            or kwargs.get("num_kv_heads")
            or getattr(config, "num_key_value_heads", None)
            or getattr(config, "num_kv_heads", None)
            or self.num_heads
        )
        # RP-FIX-23
        self.num_kv_heads = min(self.num_kv_heads, self.num_heads)
        self.head_dim = (
            kwargs.get("head_dim")
            or getattr(config, "head_dim", None)
            or getattr(config, "d_kv", None)
            or (self.dim // max(1, self.num_heads))
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        # RP-FIX-11
        self.is_decoder = _detect_decoder(config, is_decoder, pretrained_attn)
        self.is_nested_encoder = False  # RP-FIX-22

        # RP-FIX-12
        if pre_ln is None:
            pre_ln = getattr(config, "pre_ln", True) if config is not None else True
        self.pre_ln = bool(pre_ln)

        self.lora_rank = (
            getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
            if config
            else 16
        )
        self.lora_alpha = (
            getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
            if config
            else 32.0
        )
        self.freeze_base = self.lora_rank > 0
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        source_attn = (
            pretrained_attn
            if pretrained_attn is not None
            else kwargs.get("source_attn")
        )
        target_dtype = kwargs.get("dtype")
        if target_dtype is None and source_attn is not None:
            p = next(source_attn.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        self.q_lin = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.k_lin = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.v_lin = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.out_lin = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )

        # Aliases
        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin
        self.query = self.q_lin
        self.key = self.k_lin
        self.value = self.v_lin
        self.dense = self.out_lin
        self.o_proj = self.out_lin
        self.c_proj = self.out_lin

        self.moe = REAPMoE(
            self.dim,
            n_experts=n_experts,
            top_k=top_k,
            config=config,
            dtype=target_dtype,
        )
        self.norm1 = nn.LayerNorm(self.dim, dtype=target_dtype)
        self.norm2 = nn.LayerNorm(self.dim, dtype=target_dtype)

        dropout_p = (
            getattr(
                config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        self.register_buffer("_attn_aux_loss", torch.tensor(0.0), persistent=False)

        if source_attn is not None:
            self.copy_weights(source_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.moe.aux_loss

    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = (
            hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        )
        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
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

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (
                    c_attn_mod.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
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

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        c_proj_mod.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # RoBERTa / BERT nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # Flat modules
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        # UniversalAdapter fallback
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
                    w, b = adapter.get_weight_bias(src, k)
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
            raise RuntimeError(
                f"REAPMoEAttention.copy_weights: missing mandatory projections "
                f"{missing_after} from {src.__class__.__name__}."
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
                        # For nested, out_lin will be Identity (set below).
                        continue
                    else:
                        raise RuntimeError(
                            f"REAPMoEAttention: missing weight for {proj_key}."
                        )

                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif (
                    w.shape[0] < target_mod.weight.shape[0]
                    and target_mod.weight.shape[0] % w.shape[0] == 0
                    and w.shape[1] == target_mod.weight.shape[1]
                ):
                    repeats = target_mod.weight.shape[0] // w.shape[0]
                    target_mod.weight.copy_(w.repeat(repeats, 1))
                else:
                    min_0 = min(target_mod.weight.shape[0], w.shape[0])
                    min_1 = min(target_mod.weight.shape[1], w.shape[1])
                    target_mod.weight.zero_()
                    target_mod.weight[:min_0, :min_1].copy_(w[:min_0, :min_1])

                if b is not None and target_mod.bias is not None:
                    if b.shape == target_mod.bias.shape:
                        target_mod.bias.copy_(b)
                    elif (
                        b.shape[0] < target_mod.bias.shape[0]
                        and target_mod.bias.shape[0] % b.shape[0] == 0
                    ):
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                norm = float(torch.linalg.norm(target_mod.weight.float()).item())
                if norm < 1e-4:
                    logger.warning(
                        "REAPMoEAttention: weight for '%s' has near-zero "
                        "Frobenius norm (%.2e).",
                        proj_key,
                        norm,
                    )

        if self.freeze_base:
            for mod in [self.q_lin, self.k_lin, self.v_lin, self.out_lin]:
                mod.freeze_base_weights()

        # RP-FIX-16: nested encoder should NOT apply its own out_lin.
        if self.is_nested_encoder:
            self.out_lin = nn.Identity()

    # ------------------------------------------------------------------
    # Mask helper (RP-FIX-20)
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: torch.Tensor | None,
        B: int,
        N_kv: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 3:
            # (B, N_q, N_kv) or (B, 1, N_kv)
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            # Reduce over query dimension if present, else keep as is.
            if v.shape[1] > 1:
                v = v.any(dim=1)  # (B, N_kv)
            else:
                v = v.squeeze(1)  # (B, N_kv)
            return v
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # RP-FIX-24
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "REAPMoEAttention does not support cross-attention."
            )

        B, L, _ = hidden_states.shape
        device = hidden_states.device

        target_dtype = hidden_states.dtype

        # Pre-LN: apply norm before attention (RP-FIX-17)
        if self.pre_ln:
            n1_dtype = (
                self.norm1.weight.dtype
                if hasattr(self.norm1, "weight") and self.norm1.weight is not None
                else target_dtype
            )
            h_norm = self.norm1(hidden_states.to(n1_dtype)).to(target_dtype)
            attn_input = h_norm
            residual = hidden_states
        else:
            attn_input = hidden_states
            residual = hidden_states

        # Projections on attn_input
        q = (
            self.q_lin(attn_input)
            .view(B, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k_new = (
            self.k_lin(attn_input)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v_new = (
            self.v_lin(attn_input)
            .view(B, L, self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # RoPE with past_len
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            past_len_rope = 0
            if past_key_value is not None:
                if (
                    isinstance(past_key_value, (tuple, list))
                    and len(past_key_value) >= 1
                ):
                    past_len_rope = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    # RP-FIX-19: use layer_idx
                    idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                    past_len_rope = past_key_value.key_cache[idx].shape[-2]
            cos = cos[..., past_len_rope : past_len_rope + L, :]
            sin = sin[..., past_len_rope : past_len_rope + L, :]
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

        # KV-cache
        past_k = past_v = None
        if past_key_value is not None:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]

        if past_k is not None:
            K_cat = torch.cat([past_k, k_new], dim=-2)
            V_cat = torch.cat([past_v, v_new], dim=-2)
        else:
            K_cat = k_new
            V_cat = v_new

        if past_key_value is not None and hasattr(past_key_value, "update"):
            past_key_value.update(k_new, v_new, layer_idx)

        N_kv = K_cat.shape[2]
        past_len = N_kv - L if self.is_decoder else 0

        # GQA expansion
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            K_full = repeat_kv(K_cat, repeats)
            V_full = repeat_kv(V_cat, repeats)
        else:
            K_full = K_cat
            V_full = V_cat

        # RP-FIX-18: disable SDPA if causal and past_len > 0
        use_sdpa = (
            not output_attentions
            and head_mask is None
            and not (self.is_decoder and past_len > 0)
        )

        if use_sdpa:
            q_sdp = q.to(torch.float16) if q.dtype == torch.float32 else q
            k_sdp = (
                K_full.to(torch.float16) if K_full.dtype == torch.float32 else K_full
            )
            v_sdp = (
                V_full.to(torch.float16) if V_full.dtype == torch.float32 else V_full
            )

            attn_bias_sdp = None
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    attn_bias_sdp = attention_mask.to(q_sdp.dtype)
                else:
                    token_valid = self._prepare_valid_mask(
                        attention_mask, B, N_kv, device
                    )
                    if token_valid is not None:
                        attn_bias_sdp = (
                            (token_valid[:, None, None, :].float() - 1.0) * 1e4
                        ).to(q_sdp.dtype)

            is_causal_sdp = self.is_decoder and L > 1 and past_len == 0

            attn_out_sdp = F.scaled_dot_product_attention(
                q_sdp,
                k_sdp,
                v_sdp,
                attn_mask=attn_bias_sdp,
                is_causal=is_causal_sdp,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            attn_out = (
                attn_out_sdp.to(hidden_states.dtype)
                .transpose(1, 2)
                .contiguous()
                .view(B, L, self.all_head_size)
            )
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
            attn_ret = None
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (
                torch.matmul(q.float(), K_full.float().transpose(-2, -1).contiguous())
                * scale
            )

            token_valid = self._prepare_valid_mask(attention_mask, B, N_kv, device)

            if self.is_decoder:
                if past_len > 0:
                    abs_q = (past_len + torch.arange(L, device=device)).unsqueeze(1)
                    abs_k = torch.arange(N_kv, device=device).unsqueeze(0)
                    causal = (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)
                else:
                    causal = (
                        torch.tril(torch.ones(L, N_kv, device=device, dtype=torch.bool))
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                scores = scores.masked_fill(~causal, -1e4)

            if token_valid is not None:
                scores = scores.masked_fill(~token_valid[:, None, None, :].bool(), -1e4)

            scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
            probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
            probs = torch.nan_to_num(probs, nan=0.0)
            probs = self.dropout(probs)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask[:, :, :1].unsqueeze(-1)
                probs = probs * head_mask.to(probs.dtype)

            attn_out = (
                torch.matmul(probs, V_full)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(B, L, self.all_head_size)
            )
            attn_ret = probs if output_attentions else None

        # RP-FIX-16: skip out_lin for nested encoder
        if self.is_nested_encoder:
            proj = attn_out
        else:
            proj = self.out_lin(attn_out)

        proj = torch.nan_to_num(proj, nan=0.0, posinf=1e4, neginf=-1e4)

        # Residual + MoE (RP-FIX-17)
        n2_dtype = (
            self.norm2.weight.dtype
            if hasattr(self.norm2, "weight") and self.norm2.weight is not None
            else target_dtype
        )
        if self.pre_ln:
            h1 = (residual + self.dropout(proj.to(target_dtype))).to(target_dtype)
            h1_norm = self.norm2(h1.to(n2_dtype)).to(target_dtype)
            moe_out, _ = self.moe(h1_norm)
            out = (h1 + self.dropout(moe_out.to(target_dtype))).to(target_dtype)
        else:
            # Post-LN: norm after residual
            n1_dtype = (
                self.norm1.weight.dtype
                if hasattr(self.norm1, "weight") and self.norm1.weight is not None
                else target_dtype
            )
            h1 = (residual + self.dropout(proj.to(target_dtype))).to(target_dtype)
            h1 = self.norm1(h1.to(n1_dtype)).to(target_dtype)
            moe_out, _ = self.moe(h1)
            out = (h1 + self.dropout(moe_out.to(target_dtype))).to(target_dtype)
            out = self.norm2(out.to(n2_dtype)).to(target_dtype)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out.to(target_dtype), attn_ret


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("reap_moe_inject")
@ATTENTION_REGISTRY.register("inject_reap_moe")
def inject_reap_moe(
    model: nn.Module | None = None,
    layer: nn.Module | None = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
) -> nn.Module | None:
    if _depth > 8:
        logger.warning("inject_reap_moe: max recursion depth exceeded, aborting.")
        return None

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
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
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
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_reap_moe(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    try:
        src_attn = old_attn
        if src_attn is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if cand is not None:
                        src_attn = getattr(cand, "self", cand)
                        break

        def _already_injected(module):
            if module is None:
                return False
            if isinstance(module, REAPMoEAttention):
                return True
            if "REAPMoE" in module.__class__.__name__:
                return True
            inner = getattr(module, "inner", None)
            if isinstance(inner, REAPMoEAttention):
                return True
            wrapped = getattr(module, "_wrapped_attention", None)
            if isinstance(wrapped, REAPMoEAttention):
                return True
            return False

        if _already_injected(src_attn):
            return None

        if src_attn is None:
            return None

        cls_name = src_attn.__class__.__name__
        if "DeltaNet" in cls_name or "LinearAttention" in cls_name:
            return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        # RP-FIX-25: propagate n_experts, top_k from comp_config
        extra_kwargs = {}
        if comp_config is not None:
            for key in ("n_experts", "top_k"):
                val = getattr(comp_config, key, None)
                if val is not None:
                    extra_kwargs[key] = val

        new_attn = REAPMoEAttention(
            config=cfg,
            comp_config=comp_config,
            pretrained_attn=src_attn,
            **extra_kwargs,
            **kwargs,
        )

        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

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
    except Exception as e:
        logger.exception("inject_reap_moe failed: %s", e)
        return None


inject_reap_moe_attention = inject_reap_moe
inject_reap_moe_model = inject_reap_moe
