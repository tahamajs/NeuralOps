"""
Unified shared primitives and utility layers for attention and compression modules.
Centralizes repeatedly copied helpers (_cfg_get, repeat_kv, _detect_decoder, LoRALinear, FactorizedLinear).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from neuralops.core.triton_kernels import triton_dispatch

    HAS_TRITON_COMMON = True
except Exception:
    triton_dispatch = None
    HAS_TRITON_COMMON = False

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




def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats Key/Value heads along the head dimension for Grouped-Query Attention (GQA).
    hidden_states: (batch, num_key_value_heads, seqlen, head_dim)
    Returns: (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep <= 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


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
    if cos.dim() == 2 and q.dim() == 4:
        # (N, D) → (1, 1, N, D) to broadcast against (B, H, N, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() < q.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)


    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.type_as(q), k_embed.type_as(k)



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
                if hasattr(self.base_layer, "nf"):  # Conv1D
                    in_feat, out_feat = (
                        self.base_layer.weight.shape[0],
                        self.base_layer.weight.shape[1],
                    )
                else:
                    out_feat, in_feat = (
                        self.base_layer.weight.shape[0],
                        self.base_layer.weight.shape[1],
                    )
            self.in_features = in_feat
            self.out_features = out_feat
            self.is_wrapper = True
        else:
            # Standalone mode
            self.is_wrapper = False
            self.in_features = int(base_or_in_features)
            self.out_features = int(
                out_features if out_features is not None else base_or_in_features
            )
            self.weight = nn.Parameter(
                torch.empty((self.out_features, self.in_features), **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
            else:
                self.register_parameter("bias", None)

        if self.lora_rank > 0 and self.in_features and self.out_features:
            self.lora_A = nn.Parameter(
                torch.empty((self.lora_rank, self.in_features), **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty((self.out_features, self.lora_rank), **factory_kwargs)
            )
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

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> "LoRALinear":
        """Wrap an existing nn.Linear layer with a LoRA adapter.

        The base weight is copied and (optionally) frozen; only the LoRA
        A/B matrices will be trained.
        """
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"from_linear expects nn.Linear, got {type(layer).__name__}")

        mod = cls(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=(layer.bias is not None),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_base=freeze_base,
            dtype=layer.weight.dtype,
            device=layer.weight.device,
        )
        with torch.no_grad():
            mod.weight.data.copy_(layer.weight.data)
            if layer.bias is not None and mod.bias is not None:
                mod.bias.data.copy_(layer.bias.data)
        return mod

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_wrapper:
            res = self.base_layer(x)
        else:
            res = F.linear(x, self.weight, self.bias)

        if (
            not self.merged
            and self.lora_rank > 0
            and self.lora_A is not None
            and self.lora_B is not None
        ):
            drop_x = self.lora_dropout(x)
            if HAS_TRITON_COMMON and triton_dispatch is not None and drop_x.is_cuda:
                try:
                    lora_out = (
                        triton_dispatch("low_rank", drop_x, self.lora_B, self.lora_A, bias=None)
                        * self.lora_scaling
                    )
                except Exception:
                    lora_out = (
                        F.linear(F.linear(drop_x, self.lora_A), self.lora_B) * self.lora_scaling
                    )
            else:
                lora_out = F.linear(F.linear(drop_x, self.lora_A), self.lora_B) * self.lora_scaling
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
        self.lora_rank = (
            min(int(lora_rank), min(self.in_features, self.out_features)) if lora_rank > 0 else 0
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.lin1 = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
        self.lin2 = nn.Linear(self.rank, self.out_features, bias=bias, **factory_kwargs)

        self.lin1.weight.requires_grad = False
        self.lin2.weight.requires_grad = False
        if self.lin2.bias is not None:
            self.lin2.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, self.in_features, **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty(self.out_features, self.lora_rank, **factory_kwargs)
            )
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

    def effective_weight(self) -> torch.Tensor:
        """Reconstruct the effective weight W = W2 @ W1 + B @ A.

        NOTE: This is O(out × in × rank) and rebuilds the tensor on every
        call. Cache it if you need it in a hot loop.
        """
        w = torch.matmul(self.lin2.weight, self.lin1.weight)
        if self.lora_A is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A)
        return w

    def effective_bias(self) -> torch.Tensor | None:
        """Return the bias vector (if any)."""
        return self.lin2.bias
    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        rank: int = 128,
        lora_rank: int = 32,
        freeze_base: bool = True,
    ) -> "FactorizedLinear":
        """Factorise an existing nn.Linear layer as W ≈ W2 @ W1 via truncated SVD,
        plus an optional LoRA residual branch.
        """
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"from_linear expects nn.Linear, got {type(layer).__name__}")

        w = layer.weight.data
        out_f, in_f = w.shape
        mod = cls(
            in_features=in_f,
            out_features=out_f,
            rank=rank,
            lora_rank=lora_rank,
            bias=(layer.bias is not None),
            dtype=w.dtype,
            device=w.device,
        )

        with torch.no_grad():
            # Truncated SVD: W ≈ U_r S_r Vh_r, then split S into two halves
            # so lin1 = S_r^{1/2} Vh_r and lin2 = U_r S_r^{1/2}.
            U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
            r = min(mod.rank, len(S))
            sqrt_s = torch.diag(torch.sqrt(S[:r]))
            w1 = (sqrt_s @ Vh[:r, :]).to(dtype=w.dtype)   # (r, in_f)
            w2 = (U[:, :r] @ sqrt_s).to(dtype=w.dtype)    # (out_f, r)

            mod.lin1.weight.data.copy_(w1)
            mod.lin2.weight.data.copy_(w2)
            if layer.bias is not None and mod.lin2.bias is not None:
                mod.lin2.bias.data.copy_(layer.bias.data)

            if not freeze_base:
                mod.lin1.weight.requires_grad = True
                mod.lin2.weight.requires_grad = True
                if mod.lin2.bias is not None:
                    mod.lin2.bias.requires_grad = True

        return mod

_FactorizedLinear = FactorizedLinear


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """Rectified non-negative sparse activation function with numerical stabilization."""
    m = x.max(dim=dim, keepdim=True).values
    scaled = (x - m) / max(temperature, 1e-6)
    scaled_clamped = torch.clamp(scaled, min=-50.0, max=50.0)
    e = torch.exp(scaled_clamped)
    m_clamped = torch.clamp(-m / max(temperature, 1e-6), min=-50.0, max=50.0)
    denom = e - torch.exp(m_clamped)
    p = torch.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)


class ResSVDLinear(nn.Module):
    """
    Spectral Residual LoRA (Res-SVD) Linear Layer.

    Executes forward pass as:
    y = x @ W_base^T + scaling * (x @ A^T) @ B^T + bias
    where W_base is factorized or full, and (lora_A, lora_B) are trainable residual adapters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_rank: int = 32,
        residual_rank: int = 8,
        scaling: float = 1.0,
        factorize_base: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        max_possible_rank = min(in_features, out_features)

        self.base_rank = max(1, min(base_rank, max_possible_rank))
        remaining_rank = max(0, max_possible_rank - self.base_rank)
        self.residual_rank = max(0, min(residual_rank, remaining_rank))
        self.scaling = float(scaling)
        self.factorize_base = factorize_base

        if self.factorize_base:
            self.base_V = nn.Parameter(
                torch.empty((self.base_rank, in_features), dtype=dtype, device=device)
            )
            self.base_U = nn.Parameter(
                torch.empty((out_features, self.base_rank), dtype=dtype, device=device)
            )
            self.base_V.requires_grad = False
            self.base_U.requires_grad = False
        else:
            self.base_weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=dtype, device=device)
            )
            self.base_weight.requires_grad = False

        if self.residual_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty((self.residual_rank, in_features), dtype=dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.empty((out_features, self.residual_rank), dtype=dtype, device=device)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,), dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming uniform parameter initialization for base and zero for adapters."""
        if self.factorize_base:
            nn.init.kaiming_uniform_(self.base_V, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.base_U, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        if self.residual_rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        if self.factorize_base:
            base = torch.matmul(self.base_U, self.base_V)
        else:
            base = self.base_weight
        if self.residual_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            res = self.scaling * torch.matmul(self.lora_B, self.lora_A)
            return base + res
        return base

    @classmethod
    def from_linear(
        cls,
        layer: nn.Linear,
        base_rank: int = 32,
        residual_rank: int = 8,
        scaling: float = 1.0,
        factorize_base: bool = True,
        freeze_base: bool = True,
    ) -> "ResSVDLinear":
        """Instantiate ResSVDLinear decomposing an existing nn.Linear layer."""
        w = layer.weight.data
        b = layer.bias.data if layer.bias is not None else None
        out_f, in_f = w.shape
        mod = cls(
            in_features=in_f,
            out_features=out_f,
            base_rank=base_rank,
            residual_rank=residual_rank,
            scaling=scaling,
            factorize_base=factorize_base,
            bias=(b is not None),
            dtype=w.dtype,
            device=w.device,
        )
        with torch.no_grad():
            if factorize_base:
                # Truncated SVD decomposition of base
                U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
                r = min(mod.base_rank, len(S))
                sqrt_s = torch.diag(torch.sqrt(S[:r]))
                u_mat = (U[:, :r] @ sqrt_s).to(dtype=w.dtype)
                v_mat = (sqrt_s @ Vh[:r, :]).to(dtype=w.dtype)
                mod.base_U.data[:out_f, :r].copy_(u_mat)
                mod.base_V.data[:r, :in_f].copy_(v_mat)
                mod.base_U.requires_grad = not freeze_base
                mod.base_V.requires_grad = not freeze_base
            else:
                mod.base_weight.data.copy_(w)
                mod.base_weight.requires_grad = not freeze_base
            if b is not None and mod.bias is not None:
                mod.bias.data.copy_(b)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward execution: y = Base(x) + scaling * (x @ A^T) @ B^T + bias."""
        if self.factorize_base:
            base_intermediate = F.linear(x, self.base_V)
            base_out = F.linear(base_intermediate, self.base_U)
        else:
            base_out = F.linear(x, self.base_weight)

        if self.residual_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            res_intermediate = F.linear(x, self.lora_A)
            res_out = self.scaling * F.linear(res_intermediate, self.lora_B)
            out = base_out + res_out
        else:
            out = base_out

        if self.bias is not None:
            out = out + self.bias
        return out


__all__ = [
    "_cfg_get",
    "repeat_kv",
    "_detect_decoder",
    "_apply_rope",
    "LoRALinear",
    "FactorizedLinear",
    "_FactorizedLinear",
    "ResSVDLinear",
    "softpick",
]


