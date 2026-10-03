"""Native Sparse Attention (NSA) Module (Production-Hardened).

Combines sliding-window local context capture with dynamic, input-dependent top-$k$
global routing across key-value representations in a fully vectorized manner.

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
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY


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
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


@ATTENTION_REGISTRY.register(
    "nsa", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "nsa_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class NativeSparseAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        window_size: int = 32,
        top_k: int = 16,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        hidden_size = kwargs.get("hidden_size")
        num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
        num_kv_heads = kwargs.get("num_key_value_heads") or kwargs.get("num_kv_heads")

        if hidden_size is None:
            if config is not None:
                hidden_size = getattr(config, "hidden_size", getattr(config, "dim", 768))
            elif source_attn is not None:
                out_layer = getattr(
                    source_attn,
                    "out_lin",
                    getattr(
                        source_attn,
                        "dense",
                        getattr(source_attn, "c_proj", getattr(source_attn, "o_proj", None)),
                    ),
                )
                hidden_size = (
                    getattr(out_layer, "out_features", 768) if out_layer is not None else 768
                )
            else:
                hidden_size = 768

        if num_heads is None:
            if config is not None:
                num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
            elif source_attn is not None:
                num_heads = getattr(
                    source_attn,
                    "num_attention_heads",
                    getattr(source_attn, "n_heads", 12),
                )
            else:
                num_heads = 12

        if num_kv_heads is None:
            if config is not None:
                num_kv_heads = getattr(
                    config,
                    "num_key_value_heads",
                    getattr(config, "num_kv_heads", num_heads),
                )
            elif source_attn is not None:
                num_kv_heads = getattr(source_attn, "num_key_value_heads", num_heads)
            else:
                num_kv_heads = num_heads

        self.dim = int(hidden_size)
        self.hidden_size = self.dim
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.dim // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True
        self.window_size = (
            getattr(config, "nsa_window_size", window_size) if config else window_size
        )
        self.top_k = getattr(config, "nsa_top_k", top_k) if config else top_k

        # Decoder detection
        model_type = getattr(config, "model_type", "").lower() if config else ""
        is_dec = (
            getattr(
                config,
                "is_decoder",
                getattr(config, "is_causal", kwargs.get("is_causal", False)),
            )
            if config
            else False
        )
        _mt = (model_type or "").lower()
        _decoder_prefixes = (
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
            any(_mt.startswith(p) for p in _decoder_prefixes)
            or hasattr(config, "architectures")
            and any(
                "GPT" in str(a) or "CausalLM" in str(a)
                for a in (getattr(config, "architectures", []) or [])
            )
        ):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        # LoRA config
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

        # Projections
        self.q_proj = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj

        dropout_p = (
            getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if source_attn is not None:
            self.copy_weights(source_attn)

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = getattr(src, "__class__", type(src)).__name__
        has_out = (
            hasattr(src, "dense")
            or hasattr(src, "out_proj")
            or hasattr(src, "out_lin")
            or hasattr(src, "c_proj")
            or hasattr(src, "o_proj")
        )
        if cls_name in ["BertSelfAttention", "RobertaSelfAttention"] or (
            hasattr(src, "query") and not has_out
        ):
            self.is_nested_encoder = True
        if hasattr(src, "self") and hasattr(src, "output"):
            self.is_nested_encoder = True

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
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
                if (
                    (c_attn_mod.__class__.__name__ == "Conv1D")
                    or (base_c.__class__.__name__ == "Conv1D")
                    or hasattr(c_attn_mod, "nf")
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
                        (c_proj_mod.__class__.__name__ == "Conv1D")
                        or (base_o.__class__.__name__ == "Conv1D")
                        or hasattr(c_proj_mod, "nf")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # 2. RoBERTa nested
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
                            w_data = mod.weight.data.clone()
                            if (
                                (mod.__class__.__name__ == "Conv1D")
                                or hasattr(mod, "nf")
                                or (
                                    w_data.ndim == 2
                                    and proj_key == "o"
                                    and w_data.shape[0] < w_data.shape[1]
                                )
                            ):
                                w_data = w_data.t().contiguous()
                            extracted[proj_key] = (
                                w_data,
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug(f"UniversalAdapter fallback: {e}")

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"NativeSparseAttention copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
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
                        b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype)
                    )
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(
                        f"NativeSparseAttention weight verification failed for '{proj_key}': norm too small ({w_norm:.2e})"
                    )

                if self.freeze_base:
                    target_mod.freeze_base_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, d).permute(0, 2, 1, 3)
        is_cross = bool(kwargs.get("is_cross_attention", False))

        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = 1.0 / math.sqrt(d)
        if (
            (2 * self.window_size + 1) >= L
            and attention_mask is None
            and not output_attentions
            and head_mask is None
        ):
            try:
                _qs = Q.half() if Q.dtype == torch.float32 else Q
                _ks = K.half() if K.dtype == torch.float32 else K
                _vs = V.half() if V.dtype == torch.float32 else V
                _is_causal = getattr(self, "is_decoder", False) and not is_cross and L > 1
                _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                _sdpa_out = (
                    _sdpa_out.to(hidden_states.dtype)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, L, self.all_head_size)
                )
                _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                return (
                    self.out_proj(_sdpa_out)
                    if not getattr(self, "is_nested_encoder", False)
                    else _sdpa_out
                ), None
            except Exception:
                pass

        scores = (
            torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
        )  # [B, H, L, L]

        # Valid mask handling
        valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)
        key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)

        # Local sliding window boolean mask
        idx = torch.arange(L, device=device)
        if self.is_decoder:
            local_mask = (idx.unsqueeze(1) - idx.unsqueeze(0) <= self.window_size) & (
                idx.unsqueeze(1) >= idx.unsqueeze(0)
            )
        else:
            local_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        is_dec = bool(
            self.is_decoder
            or getattr(self, "is_causal", False)
            or getattr(self.config, "is_decoder", False)
            or getattr(self.config, "is_causal", False)
        )
        if is_dec:
            local_mask = (idx.unsqueeze(1) - idx.unsqueeze(0) <= self.window_size) & (
                idx.unsqueeze(1) >= idx.unsqueeze(0)
            )
        else:
            local_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        local_mask_4d = local_mask.unsqueeze(0).unsqueeze(0)

        if (2 * self.window_size + 1) >= L:
            total_mask = local_mask_4d & key_valid_4d
        else:
            non_local_scores = scores.masked_fill(local_mask_4d | (~key_valid_4d), -1e4)
            if is_dec:
                causal_tril = (
                    torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                non_local_scores = non_local_scores.masked_fill(~causal_tril, -1e4)
            k_val = min(self.top_k, max(1, L - (2 * self.window_size + 1)))
            _, topk_ind = torch.topk(non_local_scores, k=k_val, dim=-1)
            topk_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_ind, True)
            total_mask = (local_mask_4d | topk_mask) & key_valid_4d

        if is_dec:
            causal_tril = (
                torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            total_mask = total_mask & causal_tril

        scores = scores.masked_fill(~total_mask, -1e4)
        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.view(head_mask.size(0), -1, 1, 1)
            else:
                hm = head_mask
            attn = attn * hm

        context = torch.matmul(attn, V)
        context = context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(dtype)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("nsa_inject")
@ATTENTION_REGISTRY.register("inject_nsa")
@ATTENTION_REGISTRY.register("nsa_attention_inject")
@ATTENTION_REGISTRY.register("inject_nsa_attention")
def inject_nsa(
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
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_nsa(
                layer=lyr,
                model_config=cfg,
                comp_config=(
                    comp_config if "comp_config" in locals() or "comp_config" in kwargs else None
                ),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    src_attn = (
        old_attn
        if old_attn is not None
        else (
            getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None)))
            if layer is not None
            else None
        )
    )

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (layer.in_features if hasattr(layer, "in_features") else None)
        or getattr(model_config, "hidden_size", 64)
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(model_config, "num_attention_heads", 4)
    )

    try:
        window_size = getattr(comp_config, "window_size", 32) if comp_config else 32
        top_k = getattr(comp_config, "top_k", 16) if comp_config else 16
        new_attn = NativeSparseAttention(
            config=model_config,
            window_size=window_size,
            top_k=top_k,
            comp_config=comp_config,
            old_attn=src_attn,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            **kwargs,
        )
        if src_attn is not None:
            new_attn.copy_weights(src_attn)
        if layer is not None:
            if hasattr(layer, "attention"):
                if hasattr(layer.attention, "self"):
                    layer.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                else:
                    layer.attention = new_attn
            elif hasattr(layer, "self"):
                layer.self = new_attn
            elif hasattr(layer, "attn"):
                layer.attn = new_attn
        return new_attn
    except Exception as e:
        logger.error(f"NativeSparseAttention injection failed: {e}")
        return None


# Aliases
NSAAttention = NativeSparseAttention
inject_nsa_attention = inject_nsa
