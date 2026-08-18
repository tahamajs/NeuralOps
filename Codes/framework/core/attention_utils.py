"""
Unified Attention Injection & Weight Adaptation Engine for Foundation Models.

Provides architecture-agnostic layer detection, dynamic regex projection matching,
fused QKV tensor slicing, and mathematically symmetric Singular Value Decomposition (SVD)
low-rank factor copying.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import get_adapter
from .architecture import detect_architecture, get_attention_modules
from .model_utils import replace_module
from .registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)

# Canonical projection search patterns (ordered by specificity)
PROJ_REGEX_PATTERNS: Dict[str, List[re.Pattern]] = {
    "q": [
        re.compile(r"^q_proj$", re.I),
        re.compile(r"^query$", re.I),
        re.compile(r"^q_lin$", re.I),
        re.compile(r"^wq$", re.I),
        re.compile(r"^q$", re.I),
    ],
    "k": [
        re.compile(r"^k_proj$", re.I),
        re.compile(r"^key$", re.I),
        re.compile(r"^k_lin$", re.I),
        re.compile(r"^wk$", re.I),
        re.compile(r"^k$", re.I),
    ],
    "v": [
        re.compile(r"^v_proj$", re.I),
        re.compile(r"^value$", re.I),
        re.compile(r"^v_lin$", re.I),
        re.compile(r"^wv$", re.I),
        re.compile(r"^v$", re.I),
    ],
    "o": [
        re.compile(r"^out_proj$", re.I),
        re.compile(r"^o_proj$", re.I),
        re.compile(r"^dense$", re.I),
        re.compile(r"^out_lin$", re.I),
        re.compile(r"^c_proj$", re.I),
        re.compile(r"^o$", re.I),
    ],
}

FUSED_QKV_REGEX = re.compile(
    r"^(qkv_proj|query_key_value|c_attn|qkv|Wqkv|in_proj)$", re.I
)


def _find_submodule_by_regex(module: nn.Module, patterns: List[re.Pattern]) -> Optional[nn.Module]:
    """Find the first submodule matching any compiled regex pattern."""
    for name, sub in module.named_children():
        for pattern in patterns:
            if pattern.match(name):
                return sub
    return None


def _slice_fused_qkv_weights(
    fused_weight: torch.Tensor,
    fused_bias: Optional[torch.Tensor],
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Slices a fused QKV projection tensor (e.g. from Falcon, GPT-2, GPT-NeoX, StarCoder)
    into independent Q, K, and V weight and bias tensors.
    
    Handles standard MHA (3 * hidden_size), GQA (q_heads + 2 * kv_heads), and interleaved layouts.
    """
    total_out_dim = fused_weight.size(0)
    in_dim = fused_weight.size(1)

    if num_kv_heads is None or num_kv_heads == num_heads:
        # Standard MHA: 3 equal partitions
        split_size = total_out_dim // 3
        q_w = fused_weight[:split_size, :]
        k_w = fused_weight[split_size : 2 * split_size, :]
        v_w = fused_weight[2 * split_size :, :]

        q_b = fused_bias[:split_size] if fused_bias is not None else None
        k_b = fused_bias[split_size : 2 * split_size] if fused_bias is not None else None
        v_b = fused_bias[2 * split_size :] if fused_bias is not None else None
    else:
        # Grouped-Query Attention (GQA / MQA) fused layout
        h_dim = head_dim or (total_out_dim // (num_heads + 2 * num_kv_heads))
        q_dim = num_heads * h_dim
        kv_dim = num_kv_heads * h_dim

        q_w = fused_weight[:q_dim, :]
        k_w = fused_weight[q_dim : q_dim + kv_dim, :]
        v_w = fused_weight[q_dim + kv_dim : q_dim + 2 * kv_dim, :]

        q_b = fused_bias[:q_dim] if fused_bias is not None else None
        k_b = fused_bias[q_dim : q_dim + kv_dim] if fused_bias is not None else None
        v_b = fused_bias[q_dim + kv_dim : q_dim + 2 * kv_dim] if fused_bias is not None else None

    return {
        "q": (q_w.contiguous(), q_b.contiguous() if q_b is not None else None),
        "k": (k_w.contiguous(), k_b.contiguous() if k_b is not None else None),
        "v": (v_w.contiguous(), v_b.contiguous() if v_b is not None else None),
    }


def _copy_projection_weights(
    target_mod: nn.Module,
    src_weight: torch.Tensor,
    src_bias: Optional[torch.Tensor],
) -> None:
    """
    Copies weights and biases into target module with mathematically balanced
    Singular Value Decomposition energy distribution:
    $$U_r = U_{:, :r} \\cdot \\text{diag}(\\sqrt{S_{:r}}), \\quad V_r = \\text{diag}(\\sqrt{S_{:r}}) \\cdot V^T_{:r, :}$$
    
    Prevents gradient scale asymmetry during SGD/AdamW backpropagation.
    """
    # 1. Standard nn.Linear target
    if hasattr(target_mod, "weight") and isinstance(target_mod.weight, torch.Tensor):
        with torch.no_grad():
            if target_mod.weight.shape == src_weight.shape:
                target_mod.weight.data.copy_(src_weight)
            elif (
                target_mod.weight.shape[1] == src_weight.shape[1]
                and target_mod.weight.shape[0] < src_weight.shape[0]
            ):
                # Head compression or GQA slicing
                target_mod.weight.data.copy_(src_weight[: target_mod.weight.shape[0]])
            
            if hasattr(target_mod, "bias") and target_mod.bias is not None:
                if src_bias is not None and target_mod.bias.shape == src_bias.shape:
                    target_mod.bias.data.copy_(src_bias)
                elif src_bias is not None and target_mod.bias.shape[0] <= src_bias.shape[0]:
                    target_mod.bias.data.copy_(src_bias[: target_mod.bias.shape[0]])
                elif src_bias is None:
                    nn.init.zeros_(target_mod.bias)
        return

    # 2. Factorized SVD Linear Module (e.g., SVDLinear with U, V parameters)
    if hasattr(target_mod, "set_low_rank_factors") or hasattr(target_mod, "set_svd_factors"):
        setter_fn = getattr(target_mod, "set_low_rank_factors", getattr(target_mod, "set_svd_factors", None))
        with torch.no_grad():
            # Perform SVD in FP32 on CPU to guarantee numerical stability across all matrix shapes
            W_float = src_weight.detach().cpu().float()
            U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)

            target_rank = getattr(
                target_mod,
                "rank",
                getattr(target_mod, "svd_rank", getattr(getattr(target_mod, "U", None), "size", lambda i: 32)(1)),
            )
            r = min(target_rank, U.size(1), Vt.size(0))

            # Balanced Square-Root Singular Energy Allocation:
            # Prevents ||U|| >> ||V|| gradient scale explosions during optimization
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_balanced = (U[:, :r] * sqrt_S.unsqueeze(0)).to(
                device=src_weight.device, dtype=getattr(target_mod, "U", src_weight).dtype
            )
            V_balanced = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(
                device=src_weight.device, dtype=getattr(target_mod, "V", src_weight).dtype
            )

            setter_fn(U_balanced, V_balanced)

            if hasattr(target_mod, "bias") and target_mod.bias is not None:
                if src_bias is not None:
                    target_mod.bias.data.copy_(src_bias.to(target_mod.bias.device, target_mod.bias.dtype))
                else:
                    nn.init.zeros_(target_mod.bias)
        return


def inject_attention(
    model: nn.Module,
    attention_type: str,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Universal attention layer replacement engine for HuggingFace Transformers.
    
    Dynamically discovers attention blocks, handles both decoupled Q,K,V and fused QKV
    projection layers, distributes SVD energy symmetrically, and replaces modules in-place.
    """
    attn_class = ATTENTION_REGISTRY.resolve(attention_type)
    config = getattr(model, "config", None)

    if config is not None and not hasattr(config, "dropout"):
        dropout_val = getattr(
            config,
            "hidden_dropout_prob",
            getattr(config, "attention_probs_dropout_prob", 0.1),
        )
        setattr(config, "dropout", dropout_val)

    arch = detect_architecture(model)
    adapter = get_adapter(arch)

    num_heads = getattr(config, "num_attention_heads", 12) if config else 12
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads) if config else num_heads
    head_dim = getattr(config, "head_dim", getattr(config, "hidden_size", 768) // num_heads) if config else 64

    replaced = 0
    for name, old_attn in get_attention_modules(model):
        # 1. Instantiate new target attention mechanism
        new_attn = None
        signatures_to_try = [
            lambda: attn_class(config, rank=rank, pretrained_attn=old_attn, **kwargs),
            lambda: attn_class(config, rank=rank, **kwargs),
            lambda: attn_class(config, pretrained_attn=old_attn, **kwargs),
            lambda: attn_class(config, **kwargs),
            lambda: attn_class(config),
            lambda: attn_class(rank=rank, pretrained_attn=old_attn, **kwargs),
            lambda: attn_class(rank=rank, **kwargs),
            lambda: attn_class(**kwargs),
            lambda: attn_class(),
        ]

        for fn in signatures_to_try:
            try:
                new_attn = fn()
                break
            except (TypeError, ValueError):
                continue

        if new_attn is None:
            new_attn = attn_class(config) if config is not None else attn_class()

        # 2. Extract and transfer weights from old attention module
        if copy_weights:
            # Check for custom copy_weights hook on new module
            if hasattr(new_attn, "copy_weights") and callable(new_attn.copy_weights):
                try:
                    new_attn.copy_weights(old_attn)
                except Exception as e:
                    logger.debug("Custom copy_weights failed on %s: %s", name, e)

            # Check if old_attn has a fused QKV projection (e.g., qkv_proj, c_attn)
            fused_mod = None
            for child_name, child_mod in old_attn.named_children():
                if FUSED_QKV_REGEX.match(child_name) and hasattr(child_mod, "weight"):
                    fused_mod = child_mod
                    break

            if fused_mod is not None:
                # Sliced Fused QKV handling
                sliced = _slice_fused_qkv_weights(
                    fused_mod.weight.data,
                    fused_mod.bias.data if hasattr(fused_mod, "bias") and fused_mod.bias is not None else None,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                )
                for proj_key in ["q", "k", "v"]:
                    w, b = sliced[proj_key]
                    target_proj = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS[proj_key])
                    if target_proj is not None:
                        _copy_projection_weights(target_proj, w, b)

                # Output projection
                try:
                    src_w_o, src_b_o = adapter.get_weight_bias(old_attn, "o")
                    target_o = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS["o"])
                    if target_o is not None:
                        _copy_projection_weights(target_o, src_w_o, src_b_o)
                except ValueError:
                    pass
            else:
                # Standard decoupled Q, K, V, O projections
                for proj_key in ["q", "k", "v", "o"]:
                    try:
                        src_weight, src_bias = adapter.get_weight_bias(old_attn, proj_key)
                    except ValueError:
                        continue

                    target_proj = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS[proj_key])
                    if target_proj is not None:
                        _copy_projection_weights(target_proj, src_weight, src_bias)

        # 3. Replace module in model tree
        replace_module(model, name, new_attn)
        replaced += 1

    logger.info("Successfully injected '%s' into %d transformer blocks.", attention_type, replaced)
    return model


def available_attention_types() -> List[str]:
    """Return all registered attention mechanism identifiers."""
    return sorted(list(ATTENTION_REGISTRY.keys()))


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    r"""Rectified non-negative sparse activation function with numerical stabilization.

    Args:
        x: Input logits tensor of shape `[..., N]`.
        dim: Normalization dimension.
        temperature: Softmax temperature parameter $\tau$.

    Returns:
        Sparse normalized probability distribution `[..., N]`.
    """
    m = x.max(dim=dim, keepdim=True).values
    scaled = (x - m) / max(temperature, 1e-6)
    scaled_clamped = torch.clamp(scaled, min=-50.0, max=50.0)
    e = torch.exp(scaled_clamped)
    m_clamped = torch.clamp(-m / max(temperature, 1e-6), min=-50.0, max=50.0)
    denom = e - torch.exp(m_clamped)
    p = torch.relu(denom)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-9)
