"""REAP-MoE: Robust Expert Allocation and Pruning Mixture of Experts Attention Module – Production-Hardened.

Implements a vectorized Mixture-of-Experts block with load-balancing regularization,
SVD low-rank expert factorization, and multi-head attention weight transfer.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal masking for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from framework.core.attention_mask import apply_attention_mask, get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
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
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


class SVDLinear(nn.Module):
    is_custom_attention: bool = True
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank or max(1, min(in_features, out_features) // 4)
        self.U = nn.Parameter(torch.randn((out_features, self.rank), **factory_kwargs) * 0.02)
        self.S = nn.Parameter(torch.ones(self.rank, **factory_kwargs))
        self.V = nn.Parameter(torch.randn((in_features, self.rank), **factory_kwargs) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))

    @property
    def weight(self) -> torch.Tensor:
        return (self.U * self.S.unsqueeze(0)) @ self.V.t().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.V) * self.S @ self.U.t().contiguous() + self.bias


class REAPMoE(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        n_experts: int = 2,
        top_k: int = 1,
        noise_std: float = 0.05,
        use_svd: bool = True,
        config: Optional[Any] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            top_k = getattr(config, "top_k", top_k)

        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.noise_std = noise_std
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        self.router_ln = nn.LayerNorm(self.d_model, **factory_kwargs)
        self.router = nn.Linear(d_model, n_experts, bias=False, **factory_kwargs)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)

        if use_svd:
            expert_layer = lambda: nn.Sequential(
                SVDLinear(d_model, d_model, rank=max(1, d_model // 8), **factory_kwargs),
                nn.GELU(),
                SVDLinear(d_model, d_model, rank=max(1, d_model // 8), **factory_kwargs),
            )
        else:
            expert_layer = lambda: nn.Sequential(
                nn.Linear(d_model, d_model, **factory_kwargs),
                nn.GELU(),
                nn.Linear(d_model, d_model, **factory_kwargs),
            )

        self.experts = nn.ModuleList([expert_layer() for _ in range(n_experts)])
        # Zero-initialize final layer parameters so MoE output is exactly 0 at step 0
        for exp in self.experts:
            final_layer = exp[-1]
            if isinstance(final_layer, SVDLinear):
                nn.init.zeros_(final_layer.U)
                nn.init.zeros_(final_layer.S)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)
            elif isinstance(final_layer, nn.Linear):
                nn.init.zeros_(final_layer.weight)
                if final_layer.bias is not None:
                    nn.init.zeros_(final_layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = x.shape

        logits = self.router(self.router_ln(x))
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            scores = logits + noise
        else:
            scores = logits

        full_probs = F.softmax(scores, dim=-1)
        topk_weights, topk_indices = torch.topk(full_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        router_probs = F.softmax(logits / 1.0, dim=-1).view(-1, self.n_experts)
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_experts).float()
        density = expert_mask.sum(dim=2).view(-1, self.n_experts).mean(dim=0)
        prob_mean = router_probs.mean(dim=0)
        self.aux_loss = self.n_experts * torch.sum(density * prob_mean)

        experts_stack = torch.stack([exp(x) for exp in self.experts], dim=2)  # [B, L, E, D]
        topk_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        gathered = torch.gather(experts_stack, 2, topk_expanded)  # [B, L, k, D]

        out = torch.sum(gathered * topk_weights.unsqueeze(-1), dim=2)
        return out, None


@ATTENTION_REGISTRY.register("reap_moe")
class REAPMoEAttention(BaseAttention):
    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = 64,
        n_experts: int = 2,
        top_k: int = 1,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank or 64)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        self.dim = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        if (config is None or not hasattr(config, "hidden_size")) and "hidden_size" not in kwargs and pretrained_attn is not None:
            for cand in ["q_lin", "query", "q_proj", "c_attn"]:
                mod = getattr(pretrained_attn, cand, None)
                if mod is not None and hasattr(mod, "weight"):
                    w = mod.weight
                    self.dim = w.shape[1] if w.ndim == 2 else self.dim
                    break
        self.num_heads = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.num_kv_heads = getattr(
            config,
            "num_key_value_heads",
            kwargs.get("num_key_value_heads", self.num_heads),
        )
        self.head_dim = getattr(
            config,
            "head_dim",
            kwargs.get("head_dim", self.dim // self.num_heads),
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        is_dec = getattr(config, "is_decoder", kwargs.get("is_decoder", False))
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        self.lora_rank = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16))) if config else 16
        self.lora_alpha = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))) if config else 32.0
        self.freeze_base = (self.lora_rank > 0)
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn", None)
        target_dtype = kwargs.get("dtype", None)
        if target_dtype is None and source_attn is not None:
            target_dtype = next(source_attn.parameters(), torch.tensor([])).dtype
            if not isinstance(target_dtype, torch.dtype):
                target_dtype = torch.float32

        # Projections
        self.q_lin = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base, dtype=target_dtype)
        self.k_lin = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base, dtype=target_dtype)
        self.v_lin = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base, dtype=target_dtype)
        self.out_lin = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base, dtype=target_dtype)

        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin
        self.query = self.q_lin
        self.key = self.k_lin
        self.value = self.v_lin

        self.moe = REAPMoE(self.dim, n_experts=n_experts, top_k=top_k, config=config, dtype=target_dtype)
        self.norm1 = nn.LayerNorm(self.dim, dtype=target_dtype)
        self.norm2 = nn.LayerNorm(self.dim, dtype=target_dtype)
        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)) if config else 0.1
        self.dropout = nn.Dropout(dropout_p)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        if source_attn is not None:
            self.copy_weights(source_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(getattr(src, "self"), "query")
            and hasattr(src, "output")
            and hasattr(getattr(src, "output"), "dense")
        ):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size : 2 * split_size, :], c_b[split_size : 2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size :, :], c_b[2 * split_size :] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 2. RoBERTa nested
        elif hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            s = getattr(src, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (out_mod.weight.data.clone(), out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 3. Flat modules
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
                            extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
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

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                f"REAPMoEAttention copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )
            return

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
                src_norm = torch.linalg.norm(w.float()).item()
                if src_norm < 1e-4:
                    logger.warning(f"Source weight for {proj_key} has near-zero norm {src_norm:.6f} < 1e-4")

                if w.shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w)
                elif w.t().contiguous().shape == target_mod.weight.shape:
                    target_mod.weight.copy_(w.t().contiguous())
                elif w.shape[0] < target_mod.weight.shape[0] and target_mod.weight.shape[0] % w.shape[0] == 0 and w.shape[1] == target_mod.weight.shape[1]:
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
                    elif b.shape[0] < target_mod.bias.shape[0] and target_mod.bias.shape[0] % b.shape[0] == 0:
                        repeats = target_mod.bias.shape[0] // b.shape[0]
                        target_mod.bias.copy_(b.repeat(repeats))
                    else:
                        min_b = min(target_mod.bias.shape[0], b.shape[0])
                        target_mod.bias.zero_()
                        target_mod.bias[:min_b].copy_(b[:min_b])
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                fnorm = torch.linalg.norm(target_mod.weight.float()).item()
                if fnorm < 1e-4:
                    logger.warning(f"Weight copy notice for {proj_key}: norm {fnorm:.6f} < 1e-4")

        if self.freeze_base:
            for mod in [self.q_lin, self.k_lin, self.v_lin, self.out_lin]:
                mod.freeze_base_weights()

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape

        q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_lin(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply RoPE if provided
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Repeat KV for GQA
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = repeat_kv(k, repeats)
            v = repeat_kv(v, repeats)

        scale = 1.0 / math.sqrt(self.head_dim)
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

        # Causal masking
        is_causal = self.is_decoder or kwargs.get("is_causal", False)
        if is_causal:
            causal_mask = torch.triu(torch.full((L, L), -1e4, device=scores.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Attention mask
        if attention_mask is not None:
            scores = apply_attention_mask(scores, attention_mask)

        probs = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        probs = torch.nan_to_num(probs, nan=0.0)
        probs = self.dropout(probs)

        # head_mask handling
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                probs = probs * head_mask.view(1, self.num_heads, 1, 1).to(probs.dtype)
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                probs = probs * head_mask.view(head_mask.size(0), self.num_heads, 1, 1).to(probs.dtype)
            elif head_mask.dim() == 4:
                probs = probs * head_mask.to(probs.dtype)

        attn_out = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            return attn_out, (probs if output_attentions else None)

        proj = self.out_lin(attn_out)
        h1 = self.norm1(hidden_states + self.dropout(proj))
        moe_out, _ = self.moe(h1)
        out = self.norm2(h1 + self.dropout(moe_out))
        self.aux_loss = self.moe.aux_loss

        return out, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("reap_moe_inject")
@ATTENTION_REGISTRY.register("inject_reap_moe")
def inject_reap_moe(
    model: Optional[nn.Module] = None,
    layer: Optional[nn.Module] = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[REAPMoEAttention, nn.Module]]:
    """Safe injection hook for REAPMoEAttention across diverse transformer architectures."""
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
            res = inject_reap_moe(
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
            if isinstance(src_attn, REAPMoEAttention):
                return None
            if "REAPMoE" in src_attn.__class__.__name__ or "reap_moe" in src_attn.__class__.__name__.lower():
                return None
            if hasattr(src_attn, "inner") and isinstance(getattr(src_attn, "inner"), REAPMoEAttention):
                return None
            if hasattr(src_attn, "_wrapped_attention") and isinstance(getattr(src_attn, "_wrapped_attention"), REAPMoEAttention):
                return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, REAPMoEAttention):
                    return None
                if "REAPMoE" in cand.__class__.__name__ or "reap_moe" in cand.__class__.__name__.lower():
                    return None
                if hasattr(cand, "inner") and isinstance(getattr(cand, "inner"), REAPMoEAttention):
                    return None
                if hasattr(cand, "_wrapped_attention") and isinstance(getattr(cand, "_wrapped_attention"), REAPMoEAttention):
                    return None

        # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
        if src_attn is not None:
            c_name = src_attn.__class__.__name__
            if "DeltaNet" in c_name or "LinearAttention" in c_name:
                return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        hidden_size = (
            kwargs.get("hidden_size") or
            getattr(src_attn, 'hidden_size', None) or
            getattr(src_attn, 'embed_dim', None) or
            (src_attn.q_proj.in_features if hasattr(src_attn, 'q_proj') and hasattr(src_attn.q_proj, 'in_features') else None) or
            (src_attn.query.in_features if hasattr(src_attn, 'query') and hasattr(src_attn.query, 'in_features') else None) or
            (src_attn.q_lin.in_features if hasattr(src_attn, 'q_lin') and hasattr(src_attn.q_lin, 'in_features') else None) or
            (target.in_features if hasattr(target, 'in_features') else None) or
            getattr(cfg, 'hidden_size', None) or
            getattr(cfg, 'd_model', 64)
        )
        num_heads = (
            kwargs.get("num_heads") or
            kwargs.get("num_attention_heads") or
            getattr(src_attn, 'num_heads', None) or
            getattr(src_attn, 'num_attention_heads', None) or
            getattr(cfg, 'num_attention_heads', None) or
            getattr(cfg, 'num_heads', 4)
        )

        new_attn = REAPMoEAttention(
            config=cfg,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
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
        return None


# Aliases
inject_reap_moe_attention = inject_reap_moe
inject_reap_moe_model = inject_reap_moe

