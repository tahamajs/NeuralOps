"""
Unified attention injection for any HuggingFace model.
Uses architecture helpers and weight copying to swap attention modules.
"""

from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn

from .adapters import get_adapter
from .architecture import detect_architecture, get_attention_modules
from .model_utils import replace_module
from .registry import ATTENTION_REGISTRY


PROJ_TARGET_NAMES = {
    # Prefer canonical names (`query/key/value`) first; fall back to `_proj` variants.
    # Never use aliases that share memory with these – remove any `self.q_proj = self.query` patterns.
    "q": ["query", "q_proj", "q_lin", "Wq", "q"],
    "k": ["key", "k_proj", "k_lin", "Wk", "k"],
    "v": ["value", "v_proj", "v_lin", "Wv", "v"],
    "o": ["out_proj", "dense", "o_proj", "out_lin", "c_proj", "o"],
}


def _find_submodule(module: nn.Module, candidate_names: List[str]) -> Optional[nn.Module]:
    """Find the first matching submodule among candidate names."""
    for name in candidate_names:
        if hasattr(module, name):
            sub = getattr(module, name)
            if isinstance(sub, nn.Module):
                return sub
    return None


def _copy_projection_weights(target_mod: nn.Module, src_weight: torch.Tensor, src_bias: Optional[torch.Tensor]) -> None:
    """Copy weights and biases into target module, handling both standard Linear and SVD low-rank layers."""
    # Standard nn.Linear
    if hasattr(target_mod, "weight") and isinstance(target_mod.weight, torch.Tensor):
        with torch.no_grad():
            if target_mod.weight.shape == src_weight.shape:
                target_mod.weight.data.copy_(src_weight)
            elif target_mod.weight.shape[1] == src_weight.shape[1] and target_mod.weight.shape[0] < src_weight.shape[0]:
                # e.g., GQA head compression: average or slice query heads
                target_mod.weight.data.copy_(src_weight[:target_mod.weight.shape[0]])
            if src_bias is not None and hasattr(target_mod, "bias") and target_mod.bias is not None:
                if target_mod.bias.shape == src_bias.shape:
                    target_mod.bias.data.copy_(src_bias)
                elif target_mod.bias.shape[0] <= src_bias.shape[0]:
                    target_mod.bias.data.copy_(src_bias[:target_mod.bias.shape[0]])
            elif src_bias is None and hasattr(target_mod, "bias") and target_mod.bias is not None:
                nn.init.zeros_(target_mod.bias)
        return

    # SVD decomposed linear layer (e.g. SVDLinear with U, V or down, up)
    if hasattr(target_mod, "set_low_rank_factors"):
        with torch.no_grad():
            W = src_weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            target_rank = getattr(target_mod, "rank", getattr(target_mod, "svd_rank", getattr(target_mod.U, "size", lambda i: 32)(1)))
            r = min(target_rank, U.size(1), Vt.size(0), target_mod.U.size(1), target_mod.V.size(0))
            U_r = (U[:, :r] * S[:r]).to(target_mod.U.dtype)
            V_r = Vt[:r, :].to(target_mod.V.dtype)
            target_mod.set_low_rank_factors(U_r, V_r)
            if hasattr(target_mod, "bias") and target_mod.bias is not None and src_bias is not None:
                target_mod.bias.data.copy_(src_bias)
        return

    if hasattr(target_mod, "set_svd_factors"):
        with torch.no_grad():
            W = src_weight.float()
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            target_rank = getattr(target_mod, "svd_rank", getattr(target_mod, "rank", getattr(target_mod.U, "size", lambda i: 32)(1)))
            r = min(target_rank, U.size(1), Vt.size(0), target_mod.U.size(1), target_mod.V.size(0))
            U_r = (U[:, :r] * S[:r]).to(target_mod.U.dtype)
            V_r = Vt[:r, :].to(target_mod.V.dtype)
            target_mod.set_svd_factors(U_r, V_r)
            if hasattr(target_mod, "bias") and target_mod.bias is not None and src_bias is not None:
                target_mod.bias.data.copy_(src_bias)
        return


def inject_attention(
    model: nn.Module,
    attention_type: str,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Replace all attention modules in the model with the specified variant.
    Automatically copies pretrained weights using the appropriate adapter.
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

    replaced = 0
    for name, old_attn in get_attention_modules(model):
        # Create new attention instance
        new_attn = None
        signatures_to_try = []
        if config is not None:
            signatures_to_try.append(
                lambda: attn_class(
                    config, rank=rank, pretrained_attn=old_attn, **kwargs
                )
            )
            signatures_to_try.append(lambda: attn_class(config, rank=rank, **kwargs))
            signatures_to_try.append(lambda: attn_class(config, pretrained_attn=old_attn, **kwargs))
            signatures_to_try.append(lambda: attn_class(config, **kwargs))
            signatures_to_try.append(lambda: attn_class(config))
        else:
            signatures_to_try.append(lambda: attn_class(rank=rank, pretrained_attn=old_attn, **kwargs))
            signatures_to_try.append(lambda: attn_class(rank=rank, **kwargs))
            signatures_to_try.append(lambda: attn_class(**kwargs))
            signatures_to_try.append(lambda: attn_class())

        for fn in signatures_to_try:
            try:
                new_attn = fn()
                break
            except TypeError:
                continue

        if new_attn is None:
            if config is not None:
                new_attn = attn_class(config)
            else:
                new_attn = attn_class()

        # Copy pretrained weights
        if copy_weights:
            # If the module has its own copy_weights method, prefer it
            if hasattr(new_attn, "copy_weights") and callable(new_attn.copy_weights):
                try:
                    new_attn.copy_weights(old_attn)
                except Exception:
                    pass

            # Also perform universal attribute projection copying
            for proj in ["q", "k", "v", "o"]:
                try:
                    src_weight, src_bias = adapter.get_weight_bias(old_attn, proj)
                except ValueError:
                    continue  # skip if projection not found

                candidate_names = PROJ_TARGET_NAMES.get(proj, [])
                target_proj = _find_submodule(new_attn, candidate_names)
                if target_proj is not None:
                    _copy_projection_weights(target_proj, src_weight, src_bias)

        replace_module(model, name, new_attn)
        replaced += 1

    print(f"✅ Replaced {replaced} attention modules with '{attention_type}'.")
    return model


def available_attention_types() -> list:
    """Available attention types."""
    return ATTENTION_REGISTRY.keys()


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

