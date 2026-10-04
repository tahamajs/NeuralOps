"""
Quality and Capacity-Aware Grouped Query Attention (QCQA).

Dynamically synthesizes head-specific key-value representations for each
query head via continuous learned routing distributions over a compressed
set of KV heads.

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY


# ===========================================================================
# Helpers
# ===========================================================================
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


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


# ===========================================================================
# QCQA Attention
# ===========================================================================
@ATTENTION_REGISTRY.register("qcqa")
@ATTENTION_REGISTRY.register("qcqa_attention")
@ATTENTION_REGISTRY.register("qcqa_attn")
class QCQAAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        num_kv_heads: int | None = None,
        router_temp: float = 1.0,
        pretrained_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        rank = kwargs.pop("rank", 64)
        super().__init__(config, rank)
        self.config = config
        comp_config = kwargs.get("comp_config") or getattr(config, "compression_config", None)

        # hidden_size detection
        h_size = (
            kwargs.get("hidden_size")
            or getattr(config, "hidden_size", None)
            or getattr(config, "d_model", None)
        )
        if h_size is None and pretrained_attn is not None:
            for cand in ("q_proj", "q_lin", "query", "c_attn"):
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
            raise ValueError("QCQAAttention: could not infer hidden_size.")
        self.dim = int(h_size)

        self.num_heads = int(
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(config, "num_attention_heads", None)
            or getattr(config, "num_heads", None)
            or 12
        )

        cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if num_kv_heads is not None:
            self.num_kv_heads = int(num_kv_heads)
        elif cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        else:
            self.num_kv_heads = int(kwargs.get("num_kv_heads", 4))
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.head_dim = int(
            kwargs.get("head_dim")
            or getattr(config, "head_dim", None)
            or getattr(config, "d_kv", None)
            or (self.dim // max(1, self.num_heads))
        )
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        self.router_temp = float(
            _cfg_get(comp_config, "router_temp", kwargs.get("router_temp", router_temp))
        )
        if self.router_temp <= 0:
            self.router_temp = 1.0

        self.is_decoder = _detect_decoder(config, is_decoder, pretrained_attn)
        self.is_nested_encoder = False

        self.lora_rank = int(
            _cfg_get(
                config,
                "lora_rank",
                _cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
        )
        self.lora_alpha = float(
            _cfg_get(
                config,
                "lora_alpha",
                _cfg_get(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
        )
        self.freeze_base = self.lora_rank > 0
        self.supports_gqa = True

        target_dtype = kwargs.get("dtype")
        if target_dtype is None and pretrained_attn is not None:
            p = next(pretrained_attn.parameters(), None)
            if p is not None:
                target_dtype = p.dtype

        self.q_proj = LoRALinear(
            self.dim,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.k_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.v_proj = LoRALinear(
            self.dim,
            self.all_kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.dim,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )

        # Aliases
        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj
        self.q_lin = self.q_proj
        self.k_lin = self.k_proj
        self.v_lin = self.v_proj
        self.out_lin = self.out_proj
        self.dense = self.out_proj
        self.o_proj = self.out_proj
        self.c_proj = self.out_proj

        # QC-FIX-20: router maps q's last dim → num_kv_heads
        self.router = nn.Linear(self.head_dim, self.num_kv_heads, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

        dropout_p = _cfg_get(
            config, "dropout", _cfg_get(config, "attention_probs_dropout_prob", 0.1)
        )
        self.dropout = nn.Dropout(dropout_p)

        source_attn = pretrained_attn if pretrained_attn is not None else kwargs.get("source_attn")
        if source_attn is not None:
            self.copy_weights(source_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")

        is_nested = False
        if (
            (
                hasattr(src, "self")
                and hasattr(src.self, "query")
                and hasattr(src, "output")
                and hasattr(src.output, "dense")
            )
            or cls_name
            in (
                "BertSelfAttention",
                "RobertaSelfAttention",
                "XLMRobertaSelfAttention",
                "ElectraSelfAttention",
            )
            or cls_name.endswith("SelfAttention")
            and "distil" not in cls_name.lower()
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
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            raise RuntimeError(
                f"QCQAAttention.copy_weights: missing mandatory projections "
                f"{missing_after} from {src.__class__.__name__}."
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
                    else:
                        raise RuntimeError(f"QCQAAttention: missing weight for {proj_key}.")

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
                        "QCQAAttention: weight for '%s' has near-zero Frobenius norm (%.2e).",
                        proj_key,
                        norm,
                    )

        if self.freeze_base:
            for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
                mod.freeze_base_weights()

    # ------------------------------------------------------------------
    # QC-FIX-16: robust valid-mask helper (2D / 3D / 4D)
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
            # If query axis has size > 1, aggregate (any valid).
            if v.shape[1] > 1:
                v = v.any(dim=1)
            else:
                v = v.squeeze(1)
            return v  # (B, N_kv)
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)  # (B, N_kv)
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
        B, L, _ = hidden_states.shape
        device = hidden_states.device

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        # --- Projections ---
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k_new = self.k_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, self.head_dim)
        v_new = self.v_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, self.head_dim)

        # --- RoPE (self-attention only) ---
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len_rope = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len_rope = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                    past_len_rope = past_key_value.key_cache[idx].shape[-2]
            # QC-FIX-19: cos/sin must cover both q and k new tokens.
            cos_slice = cos[..., past_len_rope : past_len_rope + N_kv_new, :]
            sin_slice = sin[..., past_len_rope : past_len_rope + N_kv_new, :]

            # q uses the first L of these positions; k uses the last N_kv_new (== same if no cross)
            q_t = q.permute(0, 2, 1, 3).contiguous()  # (B, H_q, L, d)
            k_t = k_new.permute(0, 2, 1, 3).contiguous()  # (B, H_kv, N_kv_new, d)

            # Use slices matching L for q and N_kv_new for k.
            cos_q = cos_slice[..., :L, :]
            sin_q = sin_slice[..., :L, :]
            cos_k = cos_slice[..., :N_kv_new, :]
            sin_k = sin_slice[..., :N_kv_new, :]
            q_rot, _ = apply_rotary_pos_emb(q_t, q_t, cos_q, sin_q)
            _, k_rot = apply_rotary_pos_emb(k_t, k_t, cos_k, sin_k)
            q = q_rot.permute(0, 2, 1, 3).contiguous()
            k_new = k_rot.permute(0, 2, 1, 3).contiguous()

        # --- KV cache ---
        past_k = past_v = None
        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]

        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k, k_new], dim=1)  # (B, N_kv, H_kv, d)
            V_cat = torch.cat([past_v, v_new], dim=1)
        else:
            K_cat = k_new
            V_cat = v_new

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            past_key_value.update(k_new, v_new, layer_idx)

        N_kv = K_cat.shape[1]
        past_len = N_kv - N_kv_new if (self.is_decoder and not is_cross) else 0

        # ============================================================
        # QC-FIX-13: Dynamic routing → head-level KV synthesis → MHA
        # ============================================================
        # Router: per-token distribution over KV heads
        # q: (B, L, H_q, d) → router → (B, L, H_q, H_kv)
        route_logits = self.router(q) / max(self.router_temp, 1e-4)
        route_weights = F.softmax(route_logits.float(), dim=-1).to(q.dtype)  # (B, L, H_q, H_kv)

        # Aggregate over query positions to get head-level routing.
        # This keeps the routing input-dependent (from q) while remaining
        # computationally tractable and semantically meaningful.
        if self.is_decoder and not is_cross:
            # To prevent future information leaking backwards through the sequence mean,
            # use the routing weights from the first token only for the entire sequence.
            # This maintains O(1) KV mixing while preserving strict causality.
            w_head = route_weights[:, 0, :, :]  # (B, H_q, H_kv)
        else:
            w_head = route_weights.mean(dim=1)  # (B, H_q, H_kv)
        
        w_head = w_head / w_head.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Mix KV heads: k_cat[b, h_q, j, d] = Σ_{h_kv} w_head[b, h_q, h_kv] · k[b, j, h_kv, d]
        K_perm = K_cat.permute(0, 2, 1, 3)  # (B, H_kv, N_kv, d)
        V_perm = V_cat.permute(0, 2, 1, 3)
        # einsum: w_head[b,h_q,h_kv] · K_perm[b,h_kv,j,d] → (B, H_q, N_kv, d)
        K_mixed = torch.einsum("bqk,bkjd->bqjd", w_head, K_perm)  # (B, H_q, N_kv, d)
        V_mixed = torch.einsum("bqk,bkjd->bqjd", w_head, V_perm)

        # Prepare q for attention
        q_heads = q.permute(0, 2, 1, 3).contiguous()  # (B, H_q, L, d)

        scale = 1.0 / math.sqrt(self.head_dim)

        # Causal mask (decoder only)
        causal_mask = None
        if self.is_decoder and not is_cross:
            if past_len > 0:
                abs_q = (past_len + torch.arange(L, device=device)).unsqueeze(1)
                abs_k = torch.arange(N_kv, device=device).unsqueeze(0)
                causal = (abs_k <= abs_q).unsqueeze(0).unsqueeze(0)  # (1, 1, L, N_kv)
            else:
                causal = (
                    torch.tril(torch.ones(L, N_kv, device=device, dtype=torch.bool))
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
            causal_mask = causal

        # Padding mask
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)

        # --- Fast SDPA path (QC-FIX-18) ---
        # Build additive bias for SDPA
        attn_bias_sdpa = None
        if not output_attentions and head_mask is None:
            bias_parts = []
            if causal_mask is not None:
                causal_bias = torch.zeros(1, 1, L, N_kv, device=device, dtype=torch.float32)
                causal_bias.masked_fill_(~causal_mask, -1e4)
                bias_parts.append(causal_bias)
            if token_valid is not None:
                pad_bias = torch.zeros(1, 1, 1, N_kv, device=device, dtype=torch.float32)
                pad_bias.masked_fill_(~token_valid[:, None, None, :].bool(), -1e4)
                bias_parts.append(pad_bias)
            if bias_parts:
                combined = sum(bias_parts)
                attn_bias_sdpa = combined.to(K_mixed.dtype)

        if attn_bias_sdpa is not None or (causal_mask is None and token_valid is None):
            try:
                q_sdp = q_heads.to(torch.float16) if q_heads.dtype == torch.float32 else q_heads
                k_sdp = K_mixed.to(torch.float16) if K_mixed.dtype == torch.float32 else K_mixed
                v_sdp = V_mixed.to(torch.float16) if V_mixed.dtype == torch.float32 else V_mixed

                out_sdp = F.scaled_dot_product_attention(
                    q_sdp,
                    k_sdp,
                    v_sdp,
                    attn_mask=attn_bias_sdpa,
                    is_causal=False,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                context = (
                    out_sdp.to(q.dtype)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .view(B, L, self.all_head_size)
                )
                context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

                if self.is_nested_encoder:
                    return context, None
                out = self.out_proj(context)
                out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
                return out, None
            except Exception as e:
                logger.debug("QCQA SDPA path failed: %s; falling back to manual.", e)

        # --- Manual path (needed when output_attentions or head_mask) ---
        scores = (
            torch.matmul(
                q_heads.float(),
                K_mixed.float().transpose(-2, -1).contiguous(),
            )
            * scale
        )  # (B, H_q, L, N_kv)

        if causal_mask is not None:
            scores = scores.masked_fill(~causal_mask, -1e4)

        if token_valid is not None:
            scores = scores.masked_fill(~token_valid[:, None, None, :].bool(), -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_probs = F.softmax(scores, dim=-1).to(q.dtype)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.dropout(attn_probs)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            attn_probs = attn_probs * head_mask.to(attn_probs.dtype)

        # Context: (B, H_q, L, N_kv) @ (B, H_q, N_kv, d) → (B, H_q, L, d)
        context = torch.matmul(attn_probs, V_mixed)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            return context, (attn_probs if output_attentions else None)

        out = self.out_proj(context)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out, (attn_probs if output_attentions else None)


# ===========================================================================
# Injector
# ===========================================================================
@ATTENTION_REGISTRY.register("qcqa_inject")
@ATTENTION_REGISTRY.register("inject_qcqa")
def inject_qcqa(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    target_layer_idx: int | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_qcqa: max recursion depth exceeded, aborting.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
            res = inject_qcqa(
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

        def _already_injected(m):
            if m is None:
                return False
            if isinstance(m, QCQAAttention):
                return True
            if "QCQA" in m.__class__.__name__:
                return True
            inner = getattr(m, "inner", None)
            if isinstance(inner, QCQAAttention):
                return True
            wrapped = getattr(m, "_wrapped_attention", None)
            if isinstance(wrapped, QCQAAttention):
                return True
            return False

        if _already_injected(old_attn):
            return None

        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            return None

        if _already_injected(src):
            return None

        new_attn = QCQAAttention(
            config=cfg,
            comp_config=comp_config,
            pretrained_attn=src,
            **kwargs,
        )

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
        logger.exception("inject_qcqa failed: %s", e)
        return None


inject_qcqa_attention = inject_qcqa


__all__ = [
    "QCQAAttention",
    "LoRALinear",
    "inject_qcqa",
    "inject_qcqa_attention",
]
