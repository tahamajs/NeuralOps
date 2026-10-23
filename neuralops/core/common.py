#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Shared Core Constants and Utilities

Common typing aliases, numerical epsilon constants, and shared tensor validation primitives.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable
import os

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


_get_cfg: Callable[..., Any] = _cfg_get


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


from neuralops.core.peft.lora import LoRALinear
from neuralops.core.low_rank.factorized import FactorizedLinear, FactorisedLinear

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
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)

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
        if self.factorize_base and x.is_cuda:
            try:
                from neuralops.core.triton_kernels.low_rank import fused_svd_lora_sparse_attn_triton
                return fused_svd_lora_sparse_attn_triton(
                    x,
                    w_down=self.base_V,
                    w_up=self.base_U,
                    lora_A=self.lora_A if self.residual_rank > 0 else None,
                    lora_B=self.lora_B if self.residual_rank > 0 else None,
                    lora_scale=self.scaling,
                    bias=self.bias,
                )
            except Exception:
                pass

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


def _get_device(model: nn.Module) -> torch.device:
    """Safe model device resolution."""
    p = next(iter(model.parameters()), None)
    return p.device if p is not None else torch.device("cpu")


def _is_conv1d(module: nn.Module) -> bool:
    """Check if module is a Conv1D or equivalent layer."""
    cls = module.__class__.__name__.lower()
    return "conv1d" in cls or hasattr(module, "nf")


def _orthogonal_init(tensor: torch.Tensor) -> None:
    """Robust orthogonal initialization for square and rectangular 2D matrices."""
    rows, cols = tensor.shape
    if rows == cols:
        nn.init.orthogonal_(tensor)
        return
    a = torch.randn(max(rows, cols), min(rows, cols), device=tensor.device, dtype=tensor.dtype)
    q, _ = torch.linalg.qr(a)
    if rows >= cols:
        tensor.copy_(q[:rows, :])
    else:
        tensor.copy_(q[:, :cols].t())


def _env_float(name: str, default: float) -> float:
    """Read float configuration from environment variable safely."""
    v = os.environ.get(name, "")
    try:
        return float(v) if v else default
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    """Read boolean flag from environment variable."""
    v = os.environ.get(name, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _unpack_weight_bias(module: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract weight and bias tensors from a module or wrapper."""
    if module is None:
        return None, None
    base = getattr(module, "base_layer", module)
    w = getattr(base, "weight", None)
    if w is None and hasattr(base, "U") and hasattr(base, "V") and base.U is not None and base.V is not None:
        w = torch.matmul(base.U, base.V)
    b = getattr(base, "bias", None)
    return w, b


def _read_past_kv(past_key_value: Any, layer_idx: int = 0) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Read past Key, Value, and length from DynamicCache, list, or tuple cache formats."""
    if past_key_value is None or past_key_value is False:
        return None, None, 0

    k, v = None, None
    if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
        kc, vc = past_key_value.key_cache, past_key_value.value_cache
        if len(kc) > 0 and layer_idx < len(kc):
            k = kc[layer_idx]
            v = vc[layer_idx] if layer_idx < len(vc) else None
    elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) > 0:
        if isinstance(past_key_value[0], (tuple, list)):
            if layer_idx < len(past_key_value):
                entry = past_key_value[layer_idx]
                if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                    k, v = entry[0], entry[1]
        elif len(past_key_value) >= 2:
            k, v = past_key_value[0], past_key_value[1]

    past_len = k.shape[-2] if (k is not None and hasattr(k, "shape") and k.dim() >= 2) else 0
    return k, v, past_len


class FactorizedLoRALinear(nn.Module):
    """Linear layer decomposed into low-rank SVD projections with additive LoRA adapter."""

    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(max(1, min(int(rank), self.in_features, self.out_features)))
        self.svd_rank = self.rank
        self.lora_rank = int(lora_rank) if lora_rank > 0 else 0
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        factory = {"device": device, "dtype": dtype}
        self.down = nn.Linear(self.in_features, self.rank, bias=False, **factory)
        self.up = nn.Linear(self.rank, self.out_features, bias=bias, **factory)

        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features, **factory))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.lora_rank, **factory))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight)

    @property
    def bias(self) -> torch.Tensor | None:
        return self.up.bias

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        self.down.weight.requires_grad = True
        self.up.weight.requires_grad = True
        if self.up.bias is not None:
            self.up.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if x.dtype != self.down.weight.dtype:
            x = x.to(self.down.weight.dtype)
        out = self.up(self.down(x))
        if self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out.to(orig_dtype)


def temperature_softmax(
    x: torch.Tensor,
    dim: int = -1,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute temperature-scaled softmax numerically stably."""
    x_f = x.float()
    temp = max(float(temperature), 1e-6)
    return F.softmax(x_f / temp, dim=dim).to(x.dtype)


def _detect_return_format(source_attn: nn.Module | None, is_nested: bool = False) -> str:
    """Choose output tuple layout for host model (bert, gpt2, llama, etc.)."""
    if is_nested:
        return "bert"
    if source_attn is None:
        return "llama"
    cls_name = source_attn.__class__.__name__.lower()
    if "bert" in cls_name or "distil" in cls_name:
        return "bert"
    if "gpt2" in cls_name or "gpt" in cls_name:
        return "gpt2"
    return "llama"


def _find_layer_list(model: nn.Module) -> tuple[Any, str, Any] | None:
    """Return (parent, attr_name, layer_list) for Transformer architectures or None."""
    paths = [
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
    ]
    for path in paths:
        obj = model
        ok = True
        for part in path:
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            parent = model
            for part in path[:-1]:
                parent = getattr(parent, part)
            return parent, path[-1], obj
    return None


def _infer_d_model(
    config: Any = None,
    source_attn: nn.Module | None = None,
    d_model: int | None = None,
    **kwargs: Any,
) -> int:
    """Infer hidden dimension d_model from config, source attention module, or kwargs."""
    if d_model is not None:
        return int(d_model)
    if kwargs.get("hidden_size") is not None:
        return int(kwargs["hidden_size"])
    if config is not None:
        for attr in ("hidden_size", "d_model", "dim", "n_embd"):
            val = getattr(config, attr, None)
            if val is not None:
                return int(val)
    if source_attn is not None:
        for proj_name in ("q_proj", "q_lin", "c_attn", "query", "in_proj", "dense"):
            proj = getattr(source_attn, proj_name, None)
            if proj is not None and hasattr(proj, "in_features"):
                return int(proj.in_features)
            if proj is not None and hasattr(proj, "weight") and proj.weight.ndim == 2:
                return int(proj.weight.shape[1])
    return 768


__all__ = [
    "_cfg_get",
    "_get_cfg",
    "repeat_kv",
    "_detect_decoder",
    "_apply_rope",
    "LoRALinear",
    "FactorizedLinear",
    "_FactorizedLinear",
    "ResSVDLinear",
    "softpick",
    "_get_device",
    "_is_conv1d",
    "_orthogonal_init",
    "_env_float",
    "_env_flag",
    "_unpack_weight_bias",
    "_read_past_kv",
    "FactorizedLoRALinear",
    "temperature_softmax",
    "_detect_return_format",
    "_find_layer_list",
    "_infer_d_model",
]
