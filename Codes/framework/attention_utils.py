"""
Unified Attention Injection & Weight Adaptation Engine for Foundation Models.

Provides architecture-agnostic layer detection, dynamic regex projection matching,
fused QKV tensor slicing, RoPE geometry preservation, and mathematically symmetric
Singular Value Decomposition (SVD) low-rank factor copying.
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

# Canonical projection search patterns
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

# FIX: Regex matcher for unified multi-head projection layers (Falcon, StarCoder, GPT-2, GPT-NeoX, LLaMA-fused)
FUSED_QKV_REGEX: re.Pattern = re.compile(
    r"^(qkv_proj|query_key_value|c_attn|qkv|Wqkv|W_qkv|in_proj|fused_qkv|qkv_linear)$", re.I
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
    FIX: Slices fused QKV projection tensors into separate Q, K, and V weight and bias tensors
    supporting Standard Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Multi-Query Attention (MQA).
    
    Args:
        fused_weight: Fused QKV weight tensor of shape [total_out_dim, in_dim]
        fused_bias: Optional fused bias tensor of shape [total_out_dim]
        num_heads: Number of query attention heads
        num_kv_heads: Number of key/value heads (for GQA/MQA)
        head_dim: Dimension per attention head
        
    Returns:
        Dictionary mapping 'q', 'k', 'v' to tuples of (weight, optional_bias).
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
        # Grouped-Query Attention (GQA) fused layout
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


# FIX: Public functional alias for fused QKV slicing
slice_fused_qkv = _slice_fused_qkv_weights


def _copy_projection_weights(
    target_mod: nn.Module,
    src_weight: torch.Tensor,
    src_bias: Optional[torch.Tensor],
) -> None:
    """
    FIX: Symmetrically distributes singular values across U and V using square-root scaling:
    U_r = U[:, :r] @ diag(sqrt(S[:r]))
    V_r = diag(sqrt(S[:r])) @ V^T[:r, :]
    This eliminates gradient scale mismatch during backpropagation.
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
                target_mod.weight.data.copy_(src_weight[: target_mod.weight.shape[0]])

            if hasattr(target_mod, "bias") and target_mod.bias is not None:
                if src_bias is not None and target_mod.bias.shape == src_bias.shape:
                    target_mod.bias.data.copy_(src_bias)
                elif src_bias is not None and target_mod.bias.shape[0] <= src_bias.shape[0]:
                    target_mod.bias.data.copy_(src_bias[: target_mod.bias.shape[0]])
                elif src_bias is None:
                    nn.init.zeros_(target_mod.bias)
        return

    # 2. Factorized SVD Linear Module (e.g. SVDLinear)
    if (
        hasattr(target_mod, "set_low_rank_factors")
        or hasattr(target_mod, "set_svd_factors")
        or (hasattr(target_mod, "U") and hasattr(target_mod, "V"))
    ):
        setter_fn = getattr(target_mod, "set_low_rank_factors", getattr(target_mod, "set_svd_factors", None))
        with torch.no_grad():
            # Perform SVD in FP32 on CPU for numerical stability across all matrix sizes
            W_float = src_weight.detach().cpu().float()
            U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)

            target_rank = getattr(
                target_mod,
                "rank",
                getattr(target_mod, "svd_rank", getattr(getattr(target_mod, "U", None), "size", lambda i: 32)(1)),
            )
            r = min(target_rank, U.size(1), Vt.size(0))

            # FIX: Symmetrically distribute singular energy: U_r = U @ sqrt(S), V_r = sqrt(S) @ V^T
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_balanced = (U[:, :r] * sqrt_S.unsqueeze(0)).to(
                device=src_weight.device, dtype=getattr(target_mod, "U", src_weight).dtype
            )
            V_balanced = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(
                device=src_weight.device, dtype=getattr(target_mod, "V", src_weight).dtype
            )

            if setter_fn is not None:
                setter_fn(U_balanced, V_balanced)
            else:
                target_mod.U.data.copy_(U_balanced)
                target_mod.V.data.copy_(V_balanced)

            if hasattr(target_mod, "bias") and target_mod.bias is not None:
                if src_bias is not None:
                    target_mod.bias.data.copy_(src_bias.to(target_mod.bias.device, target_mod.bias.dtype))
                else:
                    nn.init.zeros_(target_mod.bias)
        return


# FIX: Public functional alias for setting low-rank projection weights
split_and_set_qkv_weights = _copy_projection_weights


def apply_rope_after_up_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FIX: Applies Rotary Position Embeddings (RoPE) AFTER low-rank up-projection to full
    head dimensions, preserving spatial coordinate geometry and relative frequency angles.
    
    Args:
        q: Query tensor of shape [B, H, N, D_head]
        k: Key tensor of shape [B, H, N, D_head]
        cos: Cosine rotary embedding tensor
        sin: Sine rotary embedding tensor
        position_ids: Optional token position indices
        
    Returns:
        Tuple of (q_embedded, k_embedded) with RoPE relative rotation applied.
    """
    # q, k shape: [B, H, N, D_head]
    if position_ids is not None and cos.dim() == 2:
        cos = cos[position_ids].unsqueeze(1)  # [B, 1, N, D_head]
        sin = sin[position_ids].unsqueeze(1)  # [B, 1, N, D_head]
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)  # [B, 1, N, D_head]
        sin = sin.unsqueeze(1)  # [B, 1, N, D_head]
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, N, D_head]
        sin = sin.unsqueeze(0).unsqueeze(1)  # [1, 1, N, D_head]

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def inject_attention(
    model: nn.Module,
    attention_type: str,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Universal attention layer replacement engine for HuggingFace Transformers.
    """
    attn_class = ATTENTION_REGISTRY.resolve(attention_type)
    config = getattr(model, "config", None)
    if config is not None:
        hidden_size = getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "n_embd", getattr(config, "dim", None))))
        num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", getattr(config, "n_heads", getattr(config, "n_head", None))))
        head_dim = getattr(config, "head_dim", getattr(config, "d_kv", (hidden_size // max(1, num_heads) if (hidden_size and num_heads) else None)))
        num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
        if hidden_size is not None:
            setattr(config, "hidden_size", hidden_size)
        if num_heads is not None:
            setattr(config, "num_attention_heads", num_heads)
        if head_dim is not None:
            setattr(config, "head_dim", head_dim)
        if num_kv_heads is not None:
            setattr(config, "num_key_value_heads", num_kv_heads)
        if not hasattr(config, "dropout"):
            dropout_val = getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
            setattr(config, "dropout", dropout_val)
    else:
        hidden_size = getattr(model, "hidden_size", None)
        num_heads = getattr(model, "num_attention_heads", None)
        head_dim = getattr(model, "head_dim", None)
        num_kv_heads = getattr(model, "num_key_value_heads", num_heads)

    arch = detect_architecture(model)
    adapter = get_adapter(arch)

    replaced = 0
    modules_list = list(get_attention_modules(model))
    if not modules_list:
        logger.warning("No attention modules found to replace for architecture '%s'", arch)

    for name, old_attn in modules_list:
        # Dynamically infer dimensions from old_attn if not already determined
        if hidden_size is None or num_heads is None:
            for _, child in old_attn.named_children():
                if hasattr(child, "weight") and isinstance(child.weight, torch.Tensor) and child.weight.ndim == 2:
                    if hidden_size is None:
                        hidden_size = child.weight.shape[1]
                    break
            if num_heads is None:
                num_heads = getattr(old_attn, "num_heads", getattr(old_attn, "num_attention_heads", None))
            if head_dim is None and hidden_size is not None and num_heads is not None:
                head_dim = hidden_size // max(1, num_heads)

        # Ensure config exists or construct minimal namespace config
        if config is None:
            from types import SimpleNamespace
            config = SimpleNamespace(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                head_dim=head_dim,
                num_key_value_heads=num_kv_heads or num_heads,
                dropout=0.1,
            )

        effective_rank = rank if rank is not None else getattr(config, "rank", 32)

        # Standardized instantiation: (config, rank=32, **kwargs)
        try:
            new_attn = attn_class(config, rank=effective_rank, **kwargs)
        except TypeError:
            try:
                new_attn = attn_class(config, **kwargs)
            except TypeError:
                try:
                    new_attn = attn_class(rank=effective_rank, **kwargs)
                except TypeError:
                    new_attn = attn_class(**kwargs)

        # Match device and dtype of old attention layer
        old_param = next(old_attn.parameters(), None)
        if old_param is not None:
            new_attn.to(device=old_param.device, dtype=old_param.dtype)

        if copy_weights:
            if hasattr(new_attn, "copy_weights") and callable(new_attn.copy_weights):
                try:
                    new_attn.copy_weights(old_attn)
                except (AttributeError, RuntimeError, ValueError) as e:
                    logger.debug("Custom copy_weights failed on %s: %s", name, e)
                except Exception as e:
                    logger.warning("Unexpected error in custom copy_weights on %s: %s", name, e)

            # Detect fused QKV projection modules via regex and slice parameters
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
                    num_heads=num_heads if num_heads is not None else 12,
                    num_kv_heads=num_kv_heads or num_heads,
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
                except (ValueError, AttributeError) as e:
                    logger.debug("Could not copy output projection for %s: %s", name, e)
            else:
                # Standard decoupled Q, K, V, O projections
                for proj_key in ["q", "k", "v", "o"]:
                    try:
                        src_weight, src_bias = adapter.get_weight_bias(old_attn, proj_key)
                    except (ValueError, AttributeError) as e:
                        logger.debug("Could not get %s projection weights for %s: %s", proj_key, name, e)
                        continue

                    target_proj = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS[proj_key])
                    if target_proj is not None:
                        _copy_projection_weights(target_proj, src_weight, src_bias)

        from framework.utils.replace_attention import UniversalAttentionWrapper
        wrapped_attn = UniversalAttentionWrapper(new_attn)
        replace_module(model, name, wrapped_attn)
        replaced += 1

    logger.info("Successfully injected '%s' into %d transformer blocks.", attention_type, replaced)
    return model


def available_attention_types() -> List[str]:
    """Return all registered attention mechanism identifiers."""
    return sorted(list(ATTENTION_REGISTRY.keys()))


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
