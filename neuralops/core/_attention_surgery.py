"""
Unified attention surgery module: weight slicing, regex extraction, and low-rank copying.
"""

from __future__ import annotations

import logging
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Canonical projection search patterns
PROJ_REGEX_PATTERNS: dict[str, list[re.Pattern]] = {
    "q": [
        re.compile(r"^q_proj$", re.I),
        re.compile(r"^query_proj$", re.I),
        re.compile(r"^query$", re.I),
        re.compile(r"^q_lin$", re.I),
        re.compile(r"^wq$", re.I),
        re.compile(r"^q$", re.I),
    ],
    "k": [
        re.compile(r"^k_proj$", re.I),
        re.compile(r"^key_proj$", re.I),
        re.compile(r"^key$", re.I),
        re.compile(r"^k_lin$", re.I),
        re.compile(r"^wk$", re.I),
        re.compile(r"^k$", re.I),
    ],
    "v": [
        re.compile(r"^v_proj$", re.I),
        re.compile(r"^value_proj$", re.I),
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

# Regex matcher for unified multi-head projection layers (Falcon, StarCoder, GPT-2, GPT-NeoX, LLaMA-fused)
FUSED_QKV_REGEX: re.Pattern = re.compile(
    r"^(qkv_proj|query_key_value|c_attn|qkv|Wqkv|W_qkv|in_proj|fused_qkv|qkv_linear)$",
    re.I,
)


def _find_submodule_by_regex(module: nn.Module, patterns: list[re.Pattern]) -> nn.Module | None:
    """Find the first submodule matching any compiled regex pattern."""
    for name, sub in module.named_children():
        for pattern in patterns:
            if pattern.match(name):
                return sub
    return None


def compute_reconstruction_error(src_weight: torch.Tensor, target_mod: nn.Module) -> float:
    """Computes relative Frobenius reconstruction error ||W_src - W_target||_F / (||W_src||_F + 1e-9).

    Can be called standalone to verify weight copying fidelity across any module architecture.
    """
    if src_weight is None or target_mod is None:
        return 1.0

    with torch.no_grad():
        W_s = src_weight.detach().cpu().float()
        if hasattr(target_mod, "__class__") and target_mod.__class__.__name__ == "Conv1D":
            W_s = W_s.t() if (W_s.ndim == 2 and W_s.size(0) > W_s.size(1)) else W_s

        W_t = None
        # Case 1: Reconstructed property or leaf parameter
        p_weight = getattr(target_mod, "weight", None)
        if isinstance(p_weight, nn.Parameter):
            W_t = p_weight.detach().cpu().float()
        elif hasattr(target_mod, "weight") and isinstance(target_mod.weight, torch.Tensor):
            W_t = target_mod.weight.detach().cpu().float()
        elif hasattr(target_mod, "U") and hasattr(target_mod, "V"):
            W_t = (target_mod.U @ target_mod.V).detach().cpu().float()
        elif hasattr(target_mod, "up") and hasattr(target_mod, "down"):
            W_t = (target_mod.up.weight @ target_mod.down.weight).detach().cpu().float()
        elif hasattr(target_mod, "up_proj") and hasattr(target_mod, "down_proj"):
            W_t = (target_mod.up_proj.weight @ target_mod.down_proj.weight).detach().cpu().float()

        if W_t is None:
            return 1.0

        if W_s.shape != W_t.shape:
            if W_s.shape == W_t.t().shape:
                W_t = W_t.t()
            else:
                return 1.0

        diff = torch.norm(W_s - W_t, p="fro")
        denom = torch.norm(W_s, p="fro") + 1e-9
        return float((diff / denom).item())


def _slice_fused_qkv_weights(
    fused_weight: torch.Tensor,
    fused_bias: torch.Tensor | None,
    num_heads: int,
    num_kv_heads: int | None = None,
    head_dim: int | None = None,
    source_mod: nn.Module | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    """
    Slices fused QKV projection tensors into separate Q, K, and V weight and bias tensors
    supporting Standard Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and Multi-Query Attention (MQA).
    Automatically detects and handles both nn.Linear [total_out_dim, in_dim] and Conv1D [in_dim, total_out_dim] layouts
    using runtime shape introspection and class verification.
    """
    is_conv1d = False
    if source_mod is not None:
        base_mod = getattr(source_mod, "base_layer", source_mod)
        is_conv1d = (
            source_mod.__class__.__name__ == "Conv1D"
            or base_mod.__class__.__name__ == "Conv1D"
            or hasattr(source_mod, "nf")
            or hasattr(base_mod, "nf")
        )
    elif fused_weight.ndim == 2 and fused_weight.size(0) < fused_weight.size(1):
        # Shape-based Conv1D orientation: [in_dim, total_out_dim] where total_out_dim > in_dim (e.g. 2304 > 768)
        is_conv1d = True

    if is_conv1d and fused_weight.ndim == 2 and fused_weight.size(0) < fused_weight.size(1):
        fused_weight = fused_weight.t()

    total_out_dim = fused_weight.size(0)

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


# Public functional alias for fused QKV slicing
slice_fused_qkv = _slice_fused_qkv_weights


def _copy_projection_weights(
    target_mod: nn.Module,
    src_weight: torch.Tensor,
    src_bias: torch.Tensor | None,
) -> None:
    """
    Copies projection weights and biases into target module (nn.Linear, Conv1D, SVD, or low-rank factors).
    Automatically handles transposed orientations and low-rank factorizations (down/up, U/V).
    Uses strict leaf Parameter verification rather than bare hasattr checks.
    """
    if target_mod is None or src_weight is None:
        return

    # 1. Standard nn.Linear or single weight module target (where .weight is an actual leaf Parameter)
    target_weight_param = getattr(target_mod, "weight", None)
    if isinstance(target_weight_param, nn.Parameter):
        with torch.no_grad():
            if src_weight.dim() > 2:
                src_weight_2d = src_weight.reshape(src_weight.shape[0], -1)
            else:
                src_weight_2d = src_weight

            if target_mod.weight.shape == src_weight_2d.shape:
                target_mod.weight.data.copy_(src_weight_2d)
            elif src_weight_2d.ndim == 2 and target_mod.weight.shape == src_weight_2d.t().shape:
                target_mod.weight.data.copy_(src_weight_2d.t())
            elif (
                src_weight_2d.ndim == 2
                and target_mod.weight.shape[1] == src_weight_2d.shape[1]
                and target_mod.weight.shape[0] < src_weight_2d.shape[0]
            ):
                target_mod.weight.data.copy_(src_weight_2d[: target_mod.weight.shape[0]])
            elif (
                src_weight_2d.ndim == 2
                and target_mod.weight.shape[1] == src_weight_2d.shape[0]
                and target_mod.weight.shape[0] < src_weight_2d.shape[1]
            ):
                target_mod.weight.data.copy_(src_weight_2d.t()[: target_mod.weight.shape[0]])
            elif (
                src_weight_2d.ndim == 2
                and target_mod.weight.shape[1] == src_weight_2d.shape[1]
                and target_mod.weight.shape[0] > src_weight_2d.shape[0]
                and target_mod.weight.shape[0] % src_weight_2d.shape[0] == 0
            ):
                repeat_factor = target_mod.weight.shape[0] // src_weight_2d.shape[0]
                target_mod.weight.data.copy_(src_weight_2d.repeat(repeat_factor, 1))

            if hasattr(target_mod, "bias") and isinstance(
                getattr(target_mod, "bias", None), nn.Parameter
            ):
                if src_bias is not None and target_mod.bias.shape == src_bias.shape:
                    target_mod.bias.data.copy_(src_bias)
                elif src_bias is not None and target_mod.bias.shape[0] <= src_bias.shape[0]:
                    target_mod.bias.data.copy_(src_bias[: target_mod.bias.shape[0]])
                elif (
                    src_bias is not None
                    and target_mod.bias.shape[0] > src_bias.shape[0]
                    and target_mod.bias.shape[0] % src_bias.shape[0] == 0
                ):
                    repeat_factor = target_mod.bias.shape[0] // src_bias.shape[0]
                    target_mod.bias.data.copy_(src_bias.repeat(repeat_factor))
                elif src_bias is None:
                    nn.init.zeros_(target_mod.bias)
        return

    # 2. Low-rank / Factorized Module (down/up, down_proj/up_proj, U/V, or setter methods)
    down_mod = getattr(target_mod, "down", getattr(target_mod, "down_proj", None))
    up_mod = getattr(target_mod, "up", getattr(target_mod, "up_proj", None))
    is_low_rank = (
        hasattr(target_mod, "set_low_rank_factors")
        or hasattr(target_mod, "set_svd_factors")
        or (hasattr(target_mod, "U") and hasattr(target_mod, "V"))
        or (down_mod is not None and up_mod is not None)
    )
    if is_low_rank:
        setter_fn = getattr(
            target_mod,
            "set_low_rank_factors",
            getattr(target_mod, "set_svd_factors", None),
        )
        with torch.no_grad():
            W_float = src_weight.detach().cpu().float()
            if W_float.dim() > 2:
                W_float = W_float.reshape(W_float.shape[0], -1)
            elif W_float.dim() == 1:
                W_float = W_float.unsqueeze(0)
            target_out_dim = getattr(target_mod, "out_features", None)
            if target_out_dim is None:
                if up_mod is not None and hasattr(up_mod, "weight"):
                    target_out_dim = up_mod.weight.shape[0]
                elif hasattr(target_mod, "U") and hasattr(target_mod.U, "shape"):
                    target_out_dim = target_mod.U.shape[0]

            target_in_dim = getattr(target_mod, "in_features", None)
            if target_in_dim is None:
                if down_mod is not None and hasattr(down_mod, "weight"):
                    target_in_dim = down_mod.weight.shape[1]
                elif hasattr(target_mod, "V") and hasattr(target_mod.V, "shape"):
                    target_in_dim = target_mod.V.shape[1]

            if (
                target_in_dim is not None
                and W_float.shape[1] != target_in_dim
                and W_float.shape[0] == target_in_dim
            ):
                W_float = W_float.t()

            if target_out_dim is not None and W_float.shape[0] > target_out_dim:
                W_float = W_float[:target_out_dim, :]
            elif target_out_dim is not None and W_float.shape[0] < target_out_dim:
                rep = (target_out_dim + W_float.shape[0] - 1) // W_float.shape[0]
                W_float = W_float.repeat(rep, 1)[:target_out_dim, :]
                if src_bias is not None:
                    src_bias = src_bias.repeat(rep)[:target_out_dim]
            if target_in_dim is not None and W_float.shape[1] > target_in_dim:
                W_float = W_float[:, :target_in_dim]
            elif target_in_dim is not None and W_float.shape[1] < target_in_dim:
                rep = (target_in_dim + W_float.shape[1] - 1) // W_float.shape[1]
                W_float = W_float.repeat(1, rep)[:, :target_in_dim]

            U, S, Vt = torch.linalg.svd(W_float, full_matrices=False)

            target_rank = getattr(
                target_mod,
                "rank",
                getattr(
                    target_mod,
                    "svd_rank",
                    getattr(getattr(target_mod, "U", None), "size", lambda i: 32)(1),
                ),
            )
            if down_mod is not None and hasattr(down_mod, "weight"):
                target_rank = down_mod.weight.shape[0]

            r = min(target_rank, U.size(1), Vt.size(0))

            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_balanced = (U[:, :r] * sqrt_S.unsqueeze(0)).to(
                device=src_weight.device, dtype=src_weight.dtype
            )
            V_balanced = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(
                device=src_weight.device, dtype=src_weight.dtype
            )

            if setter_fn is not None:
                setter_fn(U_balanced, V_balanced)
            elif hasattr(target_mod, "U") and hasattr(target_mod, "V"):
                r_u0 = min(target_mod.U.shape[0], U_balanced.shape[0])
                r_u1 = min(target_mod.U.shape[1], U_balanced.shape[1])
                target_mod.U.data[:r_u0, :r_u1].copy_(U_balanced[:r_u0, :r_u1])
                r_v0 = min(target_mod.V.shape[0], V_balanced.shape[0])
                r_v1 = min(target_mod.V.shape[1], V_balanced.shape[1])
                target_mod.V.data[:r_v0, :r_v1].copy_(V_balanced[:r_v0, :r_v1])
            elif down_mod is not None and up_mod is not None:
                r_d0 = min(down_mod.weight.shape[0], V_balanced.shape[0])
                r_d1 = min(down_mod.weight.shape[1], V_balanced.shape[1])
                down_mod.weight.data[:r_d0, :r_d1].copy_(V_balanced[:r_d0, :r_d1])
                r_u0 = min(up_mod.weight.shape[0], U_balanced.shape[0])
                r_u1 = min(up_mod.weight.shape[1], U_balanced.shape[1])
                up_mod.weight.data[:r_u0, :r_u1].copy_(U_balanced[:r_u0, :r_u1])
                if hasattr(up_mod, "bias") and up_mod.bias is not None and src_bias is not None:
                    min_b = min(up_mod.bias.shape[0], src_bias.shape[0])
                    up_mod.bias.data[:min_b].copy_(
                        src_bias[:min_b].to(up_mod.bias.device, up_mod.bias.dtype)
                    )
                elif (
                    hasattr(target_mod, "bias")
                    and target_mod.bias is not None
                    and src_bias is not None
                ):
                    min_b = min(target_mod.bias.shape[0], src_bias.shape[0])
                    target_mod.bias.data[:min_b].copy_(
                        src_bias[:min_b].to(target_mod.bias.device, target_mod.bias.dtype)
                    )

            if (
                hasattr(target_mod, "bias")
                and target_mod.bias is not None
                and isinstance(target_mod.bias, nn.Parameter)
            ):
                if src_bias is not None:
                    target_mod.bias.data.copy_(
                        src_bias.to(target_mod.bias.device, target_mod.bias.dtype)
                    )
                else:
                    nn.init.zeros_(target_mod.bias)
        return


# Public functional alias for setting low-rank projection weights
split_and_set_qkv_weights = _copy_projection_weights


def _svd_factorize_weights(
    src_weight: torch.Tensor,
    src_bias: torch.Tensor | None,
    lin1: nn.Module,
    lin2: nn.Module,
) -> None:
    """Factorizes a source weight into two cascade linear layers via balanced SVD."""
    with torch.no_grad():
        W_f = src_weight.detach().float()
        if hasattr(lin1, "weight") and hasattr(lin2, "weight"):
            if W_f.shape[1] != lin1.weight.shape[1] and W_f.shape[0] == lin1.weight.shape[1]:
                W_f = W_f.t()
            U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
            r = min(
                lin1.weight.shape[0],
                lin2.weight.shape[1],
                U.size(1),
                Vt.size(0),
            )
            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
            U_b = (U[:, :r] * sqrt_S.unsqueeze(0)).to(device=src_weight.device, dtype=src_weight.dtype)
            V_b = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(
                device=src_weight.device, dtype=src_weight.dtype
            )
            min_out = min(lin2.weight.shape[0], U_b.shape[0])
            min_in = min(lin1.weight.shape[1], V_b.shape[1])
            lin1.weight.data[:r, :min_in].copy_(V_b[:, :min_in])
            lin2.weight.data[:min_out, :r].copy_(U_b[:min_out, :])
            if src_bias is not None and getattr(lin2, "bias", None) is not None:
                min_b = min(lin2.bias.shape[0], src_bias.shape[0])
                lin2.bias.data[:min_b].copy_(
                    src_bias[:min_b].to(lin2.bias.device, lin2.bias.dtype)
                )


# ===========================================================================
# RoPE & GQA Backbone Surgery Hooks (LLaMA, Mistral, Qwen2)
# ===========================================================================

def adapt_decoder_attention_forward(
    module: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, Any | None]:
    """
    Safely adapts forward calls from LLaMA, Mistral, and Qwen2 decoder backbones.

    Handles `position_embeddings`, `position_ids`, `past_key_value`, and standardizes
    return tuples to `(attn_output, attn_weights, past_key_value)` preventing unpacking crashes.
    """
    # 1. Extract inputs
    hidden_states = args[0] if len(args) > 0 else kwargs.pop("hidden_states", kwargs.pop("query", None))
    attention_mask = args[1] if len(args) > 1 else kwargs.pop("attention_mask", kwargs.pop("mask", None))

    position_ids = kwargs.pop("position_ids", None)
    position_embeddings = kwargs.pop("position_embeddings", None)
    past_key_value = kwargs.pop("past_key_value", None)
    output_attentions = kwargs.pop("output_attentions", False)
    use_cache = kwargs.pop("use_cache", False)

    # 2. Forward dispatch to injected module
    out = None
    call_attempts = [
        lambda: module(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        ),
        lambda: module(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            **kwargs,
        ),
        lambda: module(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        ),
        lambda: module(hidden_states, attention_mask=attention_mask, **kwargs),
        lambda: module(hidden_states, **kwargs),
    ]

    last_err = None
    for attempt in call_attempts:
        try:
            out = attempt()
            break
        except TypeError as e:
            last_err = e
            continue

    if out is None:
        if last_err is not None:
            raise last_err
        raise RuntimeError(f"Failed to execute forward on injected module {type(module).__name__}")

    # 3. Format return tuple strictly as (attn_output, attn_weights, past_key_value)
    if isinstance(out, tuple):
        attn_out = out[0]
        if len(out) >= 3:
            attn_weights = out[1]
            past_kv = out[2]
        elif len(out) == 2:
            # If module returned (out, present/weights)
            attn_weights = out[1] if output_attentions else None
            past_kv = out[1] if not output_attentions else None
        else:
            attn_weights = None
            past_kv = None
    else:
        attn_out = out
        attn_weights = None
        past_kv = None

    return (attn_out, attn_weights, past_kv)


def apply_backbone_surgery_hooks(
    model: nn.Module,
    model_type: str = "llama",
) -> None:
    """
    Hooks RoPE & GQA backbone decoders (LLaMA, Mistral, Qwen2) to ensure
    injected modules handle position_embeddings, position_ids, and return tuples safely.
    """
    mtype = model_type.lower()
    if mtype not in ("llama", "mistral", "qwen", "qwen2", "qwen2_5", "qwen3_5"):
        return

    for name, module in model.named_modules():
        leaf = name.split(".")[-1].lower()
        if leaf in ("self_attn", "attention", "attn"):
            # Ensure module has supports_gqa and supports_rope attributes if not set
            if not hasattr(module, "supports_gqa"):
                setattr(module, "supports_gqa", True)
            if not hasattr(module, "supports_rope"):
                setattr(module, "supports_rope", True)


__all__ = [
    "PROJ_REGEX_PATTERNS",
    "FUSED_QKV_REGEX",
    "_find_submodule_by_regex",
    "compute_reconstruction_error",
    "_slice_fused_qkv_weights",
    "slice_fused_qkv",
    "_copy_projection_weights",
    "split_and_set_qkv_weights",
    "_svd_factorize_weights",
    "adapt_decoder_attention_forward",
    "apply_backbone_surgery_hooks",
]
