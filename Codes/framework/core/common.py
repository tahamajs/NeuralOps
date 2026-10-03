"""
Unified shared primitives and utility layers for attention and compression modules.
Centralizes repeatedly copied helpers (_cfg_get, repeat_kv, _detect_decoder, LoRALinear, FactorizedLinear).
"""

from __future__ import annotations

import math
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Configuration Helper
# ==============================================================================

def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """
    Safely retrieves a configuration attribute from a dict, object, or None.
    Handles nested fallbacks without throwing AttributeErrors.
    """
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    val = getattr(config, key, None)
    return val if val is not None else default


# ==============================================================================
# 2. Attention Tensor Helpers
# ==============================================================================

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key/Value heads along the head dimension for Grouped-Query Attention (GQA).
    hidden_states: (batch, num_key_value_heads, seqlen, head_dim)
    Returns: (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep <= 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(b, h, n_rep, s, d)
        .reshape(b, h * n_rep, s, d)
    )


def _detect_decoder(
    config: Any = None,
    is_decoder: bool | None = None,
    old_attn: nn.Module | None = None,
) -> bool:
    """
    Detects whether an attention layer should operate in causal decoder mode.
    """
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                          "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
            return True
        arch = getattr(config, "architectures", None)
        if arch and any(("GPT" in str(a) or "CausalLM" in str(a)) for a in arch):
            return True
    if old_attn is not None:
        if getattr(old_attn, "is_decoder", False):
            return True
        if hasattr(old_attn, "c_attn"):
            return True
        if "gpt" in old_attn.__class__.__name__.lower():
            return True
    return False


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding (RoPE) to Query and Key tensors.
    """
    if cos.dim() < q.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)


# ==============================================================================
# 3. Parameter-Efficient / Decomposed Layers
# ==============================================================================

class LoRALinear(nn.Module):
    r"""
    Universal LoRA linear layer:
    Supports both wrapper mode: LoRALinear(base_layer, r=16, ...)
    and standalone mode: LoRALinear(in_features, out_features, bias=True, lora_rank=16, ...)
    """
    def __init__(
        self,
        base_or_in_features: nn.Module | int,
        out_features: int | None = None,
        bias: bool = True,
        lora_rank: int = 16,
        r: int | None = None,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        freeze_base: bool = False,
        merge_weights: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        effective_r = r if r is not None else lora_rank
        self.lora_rank = effective_r
        self.r = effective_r
        self.lora_alpha = float(lora_alpha)
        self.lora_scaling = (lora_alpha / effective_r) if effective_r > 0 else 1.0
        self.scaling = self.lora_scaling
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.merged = False
        self.merge_weights = merge_weights

        factory_kwargs = {"device": device, "dtype": dtype}

        if isinstance(base_or_in_features, nn.Module):
            # Wrapper mode
            self.base_layer = base_or_in_features
            in_feat = getattr(self.base_layer, "in_features", None)
            out_feat = getattr(self.base_layer, "out_features", None)
            if in_feat is None and hasattr(self.base_layer, "weight"):
                if hasattr(self.base_layer, "nf"): # Conv1D
                    in_feat, out_feat = self.base_layer.weight.shape[0], self.base_layer.weight.shape[1]
                else:
                    out_feat, in_feat = self.base_layer.weight.shape[0], self.base_layer.weight.shape[1]
            self.in_features = in_feat
            self.out_features = out_feat
            self.is_wrapper = True
        else:
            # Standalone mode
            self.is_wrapper = False
            self.in_features = int(base_or_in_features)
            self.out_features = int(out_features if out_features is not None else base_or_in_features)
            self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)

        if self.lora_rank > 0 and self.in_features and self.out_features:
            self.lora_A = nn.Parameter(torch.empty((self.lora_rank, self.in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((self.out_features, self.lora_rank), **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        if not self.is_wrapper:
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
        if self.is_wrapper:
            for p in self.base_layer.parameters():
                p.requires_grad = False
        else:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        if self.is_wrapper:
            for p in self.base_layer.parameters():
                p.requires_grad = True
        else:
            self.weight.requires_grad = True
            if self.bias is not None:
                self.bias.requires_grad = True

    def merge(self) -> None:
        if self.merged or self.lora_rank <= 0 or self.lora_A is None:
            return
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.lora_scaling
            if self.is_wrapper:
                if hasattr(self.base_layer, "nf"):
                    delta = delta.t()
                self.base_layer.weight.data.add_(delta.to(self.base_layer.weight.dtype))
            else:
                self.weight.data.add_(delta.to(self.weight.dtype))
            self.merged = True

    def unmerge(self) -> None:
        if not self.merged or self.lora_rank <= 0 or self.lora_A is None:
            return
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.lora_scaling
            if self.is_wrapper:
                if hasattr(self.base_layer, "nf"):
                    delta = delta.t()
                self.base_layer.weight.data.sub_(delta.to(self.base_layer.weight.dtype))
            else:
                self.weight.data.sub_(delta.to(self.weight.dtype))
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_wrapper:
            res = self.base_layer(x)
        else:
            res = F.linear(x, self.weight, self.bias)

        if not self.merged and self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.lora_scaling
            res = res + lora_out
        return res


class FactorizedLinear(nn.Module):
    """
    Decomposed linear layer: W ≈ W2 @ W1 with optional LoRA residual branch.
    """
    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        lora_rank: int = 32,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.svd_rank = self.rank
        self.lora_rank = min(int(lora_rank), min(self.in_features, self.out_features)) if lora_rank > 0 else 0

        factory_kwargs = {"device": device, "dtype": dtype}
        self.lin1 = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
        self.lin2 = nn.Linear(self.rank, self.out_features, bias=bias, **factory_kwargs)

        self.lin1.weight.requires_grad = False
        self.lin2.weight.requires_grad = False
        if self.lin2.bias is not None:
            self.lin2.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features, **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.lora_rank, **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lin1.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lin2.weight, a=math.sqrt(5))
            if self.lin2.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lin2.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.lin2.bias, -bound, bound)
            if self.lora_A is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    def freeze_base_weights(self) -> None:
        self.lin1.weight.requires_grad = False
        self.lin2.weight.requires_grad = False
        if self.lin2.bias is not None:
            self.lin2.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.lin2.weight, self.lin1.weight)
        if self.lora_A is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A)
        return w

    @property
    def bias(self) -> torch.Tensor | None:
        return self.lin2.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin2(self.lin1(x))
        if self.lora_A is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out


__all__ = [
    "_cfg_get",
    "repeat_kv",
    "_detect_decoder",
    "_apply_rope",
    "LoRALinear",
    "FactorizedLinear",
]
