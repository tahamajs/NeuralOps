"""
Unified Attention Injection & Weight Adaptation Engine for Foundation Models.

Provides architecture-agnostic layer detection, dynamic regex projection matching,
fused QKV tensor slicing, RoPE geometry preservation, and mathematically symmetric
Singular Value Decomposition (SVD) low-rank factor copying.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
import torch.nn as nn

from .adapters import get_adapter
from .architecture import detect_architecture, get_attention_modules
from .gqa_utils import repeat_kv
from .model_utils import replace_module
from .registry import ATTENTION_REGISTRY
from framework.utils.model_family import ModelAdapter, detect_family
from framework.utils.qwen_compat import QwenIncompatibleError

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

# FIX: Regex matcher for unified multi-head projection layers (Falcon, StarCoder, GPT-2, GPT-NeoX, LLaMA-fused)
FUSED_QKV_REGEX: re.Pattern = re.compile(
    r"^(qkv_proj|query_key_value|c_attn|qkv|Wqkv|W_qkv|in_proj|fused_qkv|qkv_linear)$", re.I
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

            if hasattr(target_mod, "bias") and isinstance(getattr(target_mod, "bias", None), nn.Parameter):
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
        setter_fn = getattr(target_mod, "set_low_rank_factors", getattr(target_mod, "set_svd_factors", None))
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

            if target_in_dim is not None and W_float.shape[1] != target_in_dim and W_float.shape[0] == target_in_dim:
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
                getattr(target_mod, "svd_rank", getattr(getattr(target_mod, "U", None), "size", lambda i: 32)(1)),
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
                    up_mod.bias.data[:min_b].copy_(src_bias[:min_b].to(up_mod.bias.device, up_mod.bias.dtype))
                elif hasattr(target_mod, "bias") and target_mod.bias is not None and src_bias is not None:
                    min_b = min(target_mod.bias.shape[0], src_bias.shape[0])
                    target_mod.bias.data[:min_b].copy_(src_bias[:min_b].to(target_mod.bias.device, target_mod.bias.dtype))

            if hasattr(target_mod, "bias") and target_mod.bias is not None and isinstance(target_mod.bias, nn.Parameter):
                if src_bias is not None:
                    target_mod.bias.data.copy_(src_bias.to(target_mod.bias.device, target_mod.bias.dtype))
                else:
                    nn.init.zeros_(target_mod.bias)
        return


# Public functional alias for setting low-rank projection weights
split_and_set_qkv_weights = _copy_projection_weights


def apply_rope_after_up_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    r"""
    Expand grouped-query K/V heads along head dimension: [B, Hkv, L, D] -> [B, Hkv * n_rep, L, D].
    If n_rep <= 1, returns x unchanged without tensor allocation.
    """
    if n_rep <= 1:
        return x
    if x.dim() != 4:
        raise ValueError(f"repeat_kv expects [B, Hkv, L, D], got shape {tuple(x.shape)}")
    b, h, l, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, l, d).reshape(b, h * n_rep, l, d)


def inject_attention(
    model: nn.Module,
    attention_type: str,
    rank: int | None = None,
    copy_weights: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """Universal attention layer replacement engine for HuggingFace Transformers."""
    attn_class = ATTENTION_REGISTRY.resolve(attention_type)
    if callable(attn_class) and not isinstance(attn_class, type):
        try:
            res = attn_class(model=model, rank=rank, copy_weights=copy_weights, **kwargs)
            if res is not None:
                return res
        except Exception as inj_err:
            logger.debug("Dedicated injector %s failed, falling back to layer replacement: %s", attention_type, inj_err)

    config = getattr(model, "config", None)
    if config is not None:
        cfg_source = getattr(config, "text_config", config)
        hidden_size = getattr(cfg_source, "hidden_size", getattr(cfg_source, "d_model", getattr(cfg_source, "n_embd", getattr(cfg_source, "dim", None))))
        num_heads = getattr(cfg_source, "num_attention_heads", getattr(cfg_source, "num_heads", getattr(cfg_source, "n_heads", getattr(cfg_source, "n_head", None))))
        head_dim = getattr(cfg_source, "head_dim", getattr(cfg_source, "d_kv", (hidden_size // max(1, num_heads) if (hidden_size and num_heads) else None)))
        num_kv_heads = getattr(cfg_source, "num_key_value_heads", num_heads)
        if hidden_size is not None:
            config.hidden_size = hidden_size
        if num_heads is not None:
            config.num_attention_heads = num_heads
        if head_dim is not None:
            config.head_dim = head_dim
        if num_kv_heads is not None:
            config.num_key_value_heads = num_kv_heads
        if not hasattr(config, "dropout"):
            dropout_val = getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            )
            config.dropout = dropout_val
    else:
        hidden_size = getattr(model, "hidden_size", None)
        num_heads = getattr(model, "num_attention_heads", None)
        head_dim = getattr(model, "head_dim", None)
        num_kv_heads = getattr(model, "num_key_value_heads", num_heads)

    arch = detect_architecture(model)
    adapter = get_adapter(arch)
    family = detect_family(model)
    family_adapter = ModelAdapter(model, family)

    is_dec = family in ("gpt", "gpt2", "qwen", "qwen3_5", "llama", "mistral")
    if config is not None:
        config.is_decoder = bool(is_dec or getattr(config, "is_decoder", False))
        if not hasattr(config, "model_type") or not config.model_type:
            config.model_type = family

    replaced = 0
    modules_list = list(get_attention_modules(model))
    if not modules_list:
        logger.warning("No attention modules found to replace for architecture '%s'", arch)

    for name, old_attn in modules_list:
        # In hybrid architectures (like Qwen3.5), only replace linear_attn if candidate explicitly supports DeltaNet
        if "linear_attn" in name and not getattr(attn_class, "supports_delta_net", False):
            continue

        # Dynamically infer dimensions from family_adapter and old_attn
        if family_adapter is not None:
            if hidden_size is None:
                hidden_size = getattr(family_adapter, "hidden_size", None)
            if num_heads is None:
                num_heads = getattr(family_adapter, "num_heads", None)
            if num_kv_heads is None:
                num_kv_heads = getattr(family_adapter, "num_key_value_heads", None)
            if head_dim is None:
                head_dim = getattr(family_adapter, "head_dim", None)

        if hidden_size is None or num_heads is None:
            for _, child in old_attn.named_children():
                if hasattr(child, "weight") and isinstance(child.weight, torch.Tensor) and child.weight.ndim == 2:
                    if hidden_size is None:
                        hidden_size = child.weight.shape[1]
                    break
            if num_heads is None:
                num_heads = getattr(old_attn, "num_heads", getattr(old_attn, "num_attention_heads", None))
            if num_kv_heads is None:
                num_kv_heads = getattr(old_attn, "num_key_value_heads", num_heads)
            if head_dim is None and hidden_size is not None and num_heads is not None:
                head_dim = getattr(old_attn, "head_dim", hidden_size // max(1, num_heads))

        # Ensure config exists or construct minimal namespace config
        if config is None:
            from types import SimpleNamespace
            is_decoder_family_bool = bool(is_dec or family in ("gpt", "gpt2", "qwen", "qwen3_5", "llama", "mistral"))
            config = SimpleNamespace(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                head_dim=head_dim,
                num_key_value_heads=num_kv_heads or num_heads,
                dropout=0.1,
                is_decoder=is_decoder_family_bool,
                is_causal=is_decoder_family_bool,
                model_type=family,
            )
        else:
            if family in ("gpt", "gpt2", "qwen", "qwen3_5", "llama", "mistral") or is_dec:
                config.is_decoder = True
                config.is_causal = True

        # Ensure old_attn has canonical projection attributes for custom modules inspecting pretrained_attn
        for proj_key, alias_list in [
            ("q", ["query", "q_proj", "q_lin", "Wq", "wq", "q"]),
            ("k", ["key", "k_proj", "k_lin", "Wk", "wk", "k"]),
            ("v", ["value", "v_proj", "v_lin", "Wv", "wv", "v"]),
            ("o", ["dense", "out_proj", "o_proj", "out_lin", "c_proj", "wo", "o"]),
        ]:
            try:
                existing_mod = next((getattr(old_attn, a) for a in alias_list if hasattr(old_attn, a)), None)
                if existing_mod is None and hasattr(old_attn, "self"):
                    existing_mod = next((getattr(old_attn.self, a) for a in alias_list if hasattr(old_attn.self, a)), None)

                if existing_mod is not None:
                    for a in alias_list:
                        if not hasattr(old_attn, a):
                            try:
                                setattr(old_attn, a, existing_mod)
                            except Exception:
                                pass
                else:
                    w, b = adapter.get_weight_bias(old_attn, proj_key)
                    if w is not None:
                        proxy = nn.Linear(w.shape[1], w.shape[0], bias=(b is not None), device=w.device, dtype=w.dtype)
                        proxy.weight.data.copy_(w)
                        if b is not None:
                            proxy.bias.data.copy_(b)
                        for a in alias_list:
                            if not hasattr(old_attn, a):
                                try:
                                    setattr(old_attn, a, proxy)
                                except Exception:
                                    pass
            except Exception as ex:
                logger.debug("Failed to set projection aliases for %s on %s: %s", proj_key, name, ex)

        effective_rank = rank if rank is not None else getattr(config, "rank", 32)

        # Standardized instantiation: try passing pretrained_attn=old_attn first so modules
        # can inherit pretrained weights during initialization, then fallback without it
        attempts = [
            lambda: attn_class(config, rank=effective_rank, pretrained_attn=old_attn, **kwargs),
            lambda: attn_class(config, pretrained_attn=old_attn, **kwargs),
            lambda: attn_class(config=config, pretrained_attn=old_attn, **kwargs),
            lambda: attn_class(config, rank=effective_rank, **kwargs),
            lambda: attn_class(config, **kwargs),
            lambda: attn_class(rank=effective_rank, **kwargs),
            lambda: attn_class(**kwargs),
        ]
        new_attn = None
        for attempt in attempts:
            try:
                new_attn = attempt()
                break
            except TypeError:
                continue
            except Exception as e:
                logger.debug("Instantiation attempt failed: %s", e)
                continue

        if new_attn is None:
            raise RuntimeError(f"Could not instantiate attention class {attn_class.__name__}")


        # Do not let a legacy full-MHA replacement silently retain randomly
        # initialized K/V projections on GQA decoders.  Such a module must
        # explicitly advertise the geometry it implements.
        is_gqa = (family in ("qwen", "qwen3_5") and family_adapter.num_key_value_heads != family_adapter.num_heads)
        if is_gqa and not getattr(new_attn, "supports_gqa", False):
            expected = {
                "q": family_adapter.q_proj_out,
                "k": family_adapter.kv_proj_out,
                "v": family_adapter.kv_proj_out,
            }
            for key, patterns in (("q", PROJ_REGEX_PATTERNS["q"]),
                                  ("k", PROJ_REGEX_PATTERNS["k"]),
                                  ("v", PROJ_REGEX_PATTERNS["v"])):
                target = _find_submodule_by_regex(new_attn, patterns)
                out_features = getattr(target, "out_features", None)
                if target is not None and out_features not in {expected[key], None}:
                    if expected[key] is not None and out_features is not None:
                        if out_features % expected[key] == 0 or expected[key] % out_features == 0 or _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS["o"]) is not None:
                            pass
                        else:
                            raise QwenIncompatibleError(
                                f"{attention_type}: {key}_projection_out={out_features}, "
                                f"requires={expected[key]} (GQA)"
                            )

        # Match device and dtype of old attention layer
        old_param = next(old_attn.parameters(), None)
        if old_param is not None:
            new_attn.to(device=old_param.device, dtype=old_param.dtype)

        if copy_weights:
            custom_copied = False
            if is_dec or hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
                old_attn.is_decoder = True
                if hasattr(new_attn, "is_decoder"):
                    new_attn.is_decoder = True
            if hasattr(new_attn, "copy_weights") and callable(new_attn.copy_weights):
                try:
                    new_attn.copy_weights(old_attn)
                    custom_copied = True
                    if is_dec and hasattr(new_attn, "is_decoder"):
                        new_attn.is_decoder = True
                    if old_param is not None:
                        new_attn.to(device=old_param.device, dtype=old_param.dtype)
                except (AttributeError, RuntimeError, ValueError) as e:
                    logger.debug("Custom copy_weights failed on %s: %s", name, e)
                except Exception as e:
                        logger.warning("Unexpected error in custom copy_weights on %s: %s", name, e)

                if not custom_copied:
                    # Detect fused QKV projection modules via regex and slice parameters
                    fused_mod = None
                    for child_name, child_mod in old_attn.named_children():
                        if FUSED_QKV_REGEX.match(child_name) and hasattr(child_mod, "weight"):
                            fused_mod = child_mod
                            break
                    if fused_mod is None:
                        for candidate in ["c_attn", "qkv_proj", "query_key_value", "in_proj", "Wqkv"]:
                            mod = getattr(old_attn, candidate, None)
                            if mod is not None and hasattr(mod, "weight"):
                                fused_mod = mod
                                break

                    if fused_mod is not None:
                        # Sliced Fused QKV handling
                        sliced = _slice_fused_qkv_weights(
                            fused_mod.weight.data,
                            fused_mod.bias.data if hasattr(fused_mod, "bias") and fused_mod.bias is not None else None,
                            num_heads=num_heads if num_heads is not None else 12,
                            num_kv_heads=num_kv_heads or num_heads,
                            head_dim=head_dim,
                            source_mod=fused_mod,
                        )
                        for proj_key in ["q", "k", "v"]:
                            w, b = sliced[proj_key]
                            target_proj = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS[proj_key])
                            if target_proj is not None:
                                _copy_projection_weights(target_proj, w, b)
                            else:
                                lin1 = getattr(new_attn, f"{proj_key}_proj1", None)
                                lin2 = getattr(new_attn, f"{proj_key}_proj2", None)
                                if lin1 is not None and lin2 is not None:
                                    _copy_projection_weights(lin1, w, None)
                                    with torch.no_grad():
                                        W_f = w.detach().float()
                                        if hasattr(lin1, "weight") and hasattr(lin2, "weight"):
                                            if W_f.shape[1] != lin1.weight.shape[1] and W_f.shape[0] == lin1.weight.shape[1]:
                                                W_f = W_f.t()
                                            U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
                                            r = min(lin1.weight.shape[0], lin2.weight.shape[1], U.size(1), Vt.size(0))
                                            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                                            U_b = (U[:, :r] * sqrt_S.unsqueeze(0)).to(device=w.device, dtype=w.dtype)
                                            V_b = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(device=w.device, dtype=w.dtype)
                                            lin1.weight.data[:r, :].copy_(V_b)
                                            lin2.weight.data[:, :r].copy_(U_b)
                                            if b is not None and getattr(lin2, "bias", None) is not None:
                                                lin2.bias.data.copy_(b.to(lin2.bias.device, lin2.bias.dtype))

                        # Output projection
                        try:
                            src_w_o, src_b_o = adapter.get_weight_bias(old_attn, "o")
                            target_o = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS["o"])
                            if target_o is not None:
                                _copy_projection_weights(target_o, src_w_o, src_b_o)
                            else:
                                lin1_o = getattr(new_attn, "out_proj1", getattr(new_attn, "o_proj1", None))
                                lin2_o = getattr(new_attn, "out_proj2", getattr(new_attn, "o_proj2", None))
                                if lin1_o is not None and lin2_o is not None:
                                    with torch.no_grad():
                                        W_f = src_w_o.detach().float()
                                        if W_f.shape[1] != lin1_o.weight.shape[1] and W_f.shape[0] == lin1_o.weight.shape[1]:
                                            W_f = W_f.t()
                                        U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
                                        r = min(lin1_o.weight.shape[0], lin2_o.weight.shape[1], U.size(1), Vt.size(0))
                                        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                                        U_b = (U[:, :r] * sqrt_S.unsqueeze(0)).to(device=src_w_o.device, dtype=src_w_o.dtype)
                                        V_b = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(device=src_w_o.device, dtype=src_w_o.dtype)
                                        min_out = min(lin2_o.weight.shape[0], U_b.shape[0])
                                        min_in = min(lin1_o.weight.shape[1], V_b.shape[1])
                                        lin1_o.weight.data[:r, :min_in].copy_(V_b[:, :min_in])
                                        lin2_o.weight.data[:min_out, :r].copy_(U_b[:min_out, :])
                                        if src_b_o is not None and getattr(lin2_o, "bias", None) is not None:
                                            min_b = min(lin2_o.bias.shape[0], src_b_o.shape[0])
                                            lin2_o.bias.data[:min_b].copy_(src_b_o[:min_b].to(lin2_o.bias.device, lin2_o.bias.dtype))
                        except (ValueError, AttributeError) as e:
                            logger.debug("Could not copy output projection for %s: %s", name, e)
                    else:
                        # Standard decoupled Q, K, V, O projections
                        for proj_key in ["q", "k", "v", "o"]:
                            src_weight, src_bias = None, None
                            try:
                                src_weight, src_bias = adapter.get_weight_bias(old_attn, proj_key)
                            except (ValueError, AttributeError) as e:
                                logger.debug("Could not get %s projection weights for %s: %s", proj_key, name, e)

                            target_proj = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS[proj_key])
                            if target_proj is not None:
                                if src_weight is not None:
                                    _copy_projection_weights(target_proj, src_weight, src_bias)
                                elif proj_key == "o":
                                    # Encoder self-attention (BERT/RoBERTa .self) has no internal out_proj
                                    # because the outer attention container already applies output.dense.
                                    # Initialize new_attn.out_proj to Identity so context passes through transparently.
                                    with torch.no_grad():
                                        if hasattr(target_proj, "weight") and target_proj.weight.ndim == 2:
                                            if target_proj.weight.shape[0] == target_proj.weight.shape[1]:
                                                nn.init.eye_(target_proj.weight)
                                            else:
                                                nn.init.zeros_(target_proj.weight)
                                                min_d = min(target_proj.weight.shape[0], target_proj.weight.shape[1])
                                                target_proj.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                                        elif hasattr(target_proj, "set_svd_factors") or (hasattr(target_proj, "U") and hasattr(target_proj, "V")):
                                            # Low-rank / factorized linear layer: set factors to truncated identity
                                            d_out = getattr(target_proj, "out_features", getattr(target_proj, "d_out", hidden_size or 768))
                                            d_in = getattr(target_proj, "in_features", getattr(target_proj, "d_in", hidden_size or 768))
                                            eye_mat = torch.eye(d_out, d_in, device=target_proj.U.device if hasattr(target_proj, "U") else None)
                                            _copy_projection_weights(target_proj, eye_mat, None)
                                        if hasattr(target_proj, "bias") and target_proj.bias is not None:
                                            nn.init.zeros_(target_proj.bias)
                            else:
                                # Support indexed dual projections (e.g. aegis_hybrid, svd_linear_attention: q_proj1, q_proj2)
                                lin1 = getattr(new_attn, f"{proj_key}_proj1", getattr(new_attn, f"{proj_key}1", None))
                                lin2 = getattr(new_attn, f"{proj_key}_proj2", getattr(new_attn, f"{proj_key}2", None))
                                if proj_key == "o":
                                    lin1 = lin1 or getattr(new_attn, "out_proj1", getattr(new_attn, "o_proj1", None))
                                    lin2 = lin2 or getattr(new_attn, "out_proj2", getattr(new_attn, "o_proj2", None))
                                if lin1 is not None and lin2 is not None:
                                    if src_weight is not None:
                                        with torch.no_grad():
                                            W_f = src_weight.detach().float()
                                            if hasattr(lin1, "weight") and hasattr(lin2, "weight"):
                                                if W_f.shape[1] != lin1.weight.shape[1] and W_f.shape[0] == lin1.weight.shape[1]:
                                                    W_f = W_f.t()
                                                U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
                                                r = min(lin1.weight.shape[0], lin2.weight.shape[1], U.size(1), Vt.size(0))
                                                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                                                U_b = (U[:, :r] * sqrt_S.unsqueeze(0)).to(device=src_weight.device, dtype=src_weight.dtype)
                                                V_b = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(device=src_weight.device, dtype=src_weight.dtype)
                                                min_out = min(lin2.weight.shape[0], U_b.shape[0])
                                                min_in = min(lin1.weight.shape[1], V_b.shape[1])
                                                lin1.weight.data[:r, :min_in].copy_(V_b[:, :min_in])
                                                lin2.weight.data[:min_out, :r].copy_(U_b[:min_out, :])
                                                if src_bias is not None and getattr(lin2, "bias", None) is not None:
                                                    min_b = min(lin2.bias.shape[0], src_bias.shape[0])
                                                    lin2.bias.data[:min_b].copy_(src_bias[:min_b].to(lin2.bias.device, lin2.bias.dtype))
                                    elif proj_key == "o":
                                        # Identity for dual out_proj
                                        with torch.no_grad():
                                            if hasattr(lin1, "weight") and hasattr(lin2, "weight"):
                                                nn.init.zeros_(lin1.weight)
                                                nn.init.zeros_(lin2.weight)
                                                r = min(lin1.weight.shape[0], lin2.weight.shape[1], lin1.weight.shape[1], lin2.weight.shape[0])
                                                lin1.weight.data[:r, :r].fill_diagonal_(1.0)
                                                lin2.weight.data[:r, :r].fill_diagonal_(1.0)
                                                if getattr(lin2, "bias", None) is not None:
                                                    nn.init.zeros_(lin2.bias)

        from framework.utils.replace_attention import UniversalAttentionWrapper
        return_tuple = "self_attn" in name.lower() or "attention" in name.lower() or getattr(old_attn, "is_decoder", False) or family == "gpt" or name.endswith(".attn")
        wrapped_attn = UniversalAttentionWrapper(new_attn, layer_name=name, return_tuple=return_tuple)
        replace_module(model, name, wrapped_attn)
        replaced += 1

    logger.info("Successfully injected '%s' into %d transformer blocks.", attention_type, replaced)
    if not hasattr(model, "compression_stats"):
        model.compression_stats = {}
    model.compression_stats["num_compressed"] = replaced
    model.compression_stats["total_params_after"] = sum(p.numel() for p in model.parameters())
    return model


def available_attention_types() -> list[str]:
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
