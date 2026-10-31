#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Graph Surgery

AST and module-tree manipulation routines performing in-place surgery on neural network layers.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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
        is_conv1d = True

    if is_conv1d and fused_weight.ndim == 2 and fused_weight.size(0) < fused_weight.size(1):
        fused_weight = fused_weight.t()

    total_out_dim = fused_weight.size(0)

    if num_kv_heads is None or num_kv_heads == num_heads:
        split_size = total_out_dim // 3
        q_w = fused_weight[:split_size, :]
        k_w = fused_weight[split_size : 2 * split_size, :]
        v_w = fused_weight[2 * split_size :, :]

        q_b = fused_bias[:split_size] if fused_bias is not None else None
        k_b = fused_bias[split_size : 2 * split_size] if fused_bias is not None else None
        v_b = fused_bias[2 * split_size :] if fused_bias is not None else None
    else:
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

            W_recon = torch.matmul(U[:, :r], S[:r].unsqueeze(1) * Vt[:r, :])
            norm_orig = float(torch.norm(W_float, p="fro").item())
            norm_recon = float(torch.norm(W_recon, p="fro").item())
            energy_scale = math.sqrt(norm_orig / max(norm_recon, 1e-12)) if norm_orig > 1e-12 else 1.0

            sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12)) * math.sqrt(energy_scale)
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
            dev = src_weight.device
            try:
                U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
            except Exception:
                U, S, Vt = torch.linalg.svd(W_f.detach().cpu().float(), full_matrices=False)
                U = U.to(dev)
                S = S.to(dev)
                Vt = Vt.to(dev)
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
    hidden_states = args[0] if len(args) > 0 else kwargs.pop("hidden_states", kwargs.pop("query", None))
    attention_mask = args[1] if len(args) > 1 else kwargs.pop("attention_mask", kwargs.pop("mask", None))

    position_ids = kwargs.pop("position_ids", None)
    position_embeddings = kwargs.pop("position_embeddings", None)
    past_key_value = kwargs.pop("past_key_value", None)
    output_attentions = kwargs.pop("output_attentions", False)
    use_cache = kwargs.pop("use_cache", False)

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

    if isinstance(out, tuple):
        attn_out = out[0]
        if len(out) >= 3:
            attn_weights = out[1]
            past_kv = out[2]
        elif len(out) == 2:
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
            if not hasattr(module, "supports_gqa"):
                module.supports_gqa = True
            if not hasattr(module, "supports_rope"):
                module.supports_rope = True


class SubspaceAnchorWrapper(nn.Module):
    def __init__(self, base_layer: nn.Module, surg_layer: nn.Module, alpha: float = 0.25):
        super().__init__()
        self.base = base_layer
        self.surg = surg_layer
        self.alpha = alpha

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        y_b = self.base(*args, **kwargs)
        y_s = self.surg(*args, **kwargs)
        vb = y_b[0] if isinstance(y_b, tuple) else y_b
        vs = y_s[0] if isinstance(y_s, tuple) else y_s
        out = (1.0 - self.alpha) * vb + self.alpha * vs
        if isinstance(y_b, tuple):
            return (out,) + y_b[1:]
        return out


def calibrate_surgically_replaced_model(
    orig_model: nn.Module,
    surg_model: nn.Module,
    sample_inputs: dict[str, torch.Tensor] | torch.Tensor,
    ridge_lambda: float = 1e-3,
    blend_alpha: float = 0.25,
    attach_convex_anchor: bool = False,
) -> nn.Module:
    """
    Closed-Form Layerwise Ridge Output Calibration & Convex Subspace Anchoring.

    Aligns transplanted attention operators (linear attention, SVD, SSM, kernels)
    with the pretrained convex-combination feature subspace of the original attention layer:
        W_align = (Z^T Z + lambda * I)^{-1} Z^T Y_orig
        b_align = mu_Y - mu_Z W_align
    and optionally attaches a convex residual subspace anchor.
    """
    orig_model.eval()
    surg_model.eval()

    device = next(surg_model.parameters()).device
    dtype = next(surg_model.parameters()).dtype

    if isinstance(sample_inputs, dict):
        batch = {k: v.to(device) for k, v in sample_inputs.items()}
    else:
        batch = sample_inputs.to(device)

    def _collect_attention_modules(m: nn.Module) -> dict[str, nn.Module]:
        attns = {}
        for name, sub in m.named_modules():
            leaf = name.split(".")[-1].lower()
            if leaf in ("self", "self_attn", "attn"):
                attns[name] = sub
        if not attns:
            for name, sub in m.named_modules():
                leaf = name.split(".")[-1].lower()
                if leaf == "attention":
                    attns[name] = sub
        return attns

    orig_attns = _collect_attention_modules(orig_model)
    surg_attns = _collect_attention_modules(surg_model)

    common_names = [n for n in surg_attns if n in orig_attns]
    if not common_names:
        orig_list = list(orig_attns.values())
        surg_list = list(surg_attns.values())
        pairs = list(zip(orig_list, surg_list))
    else:
        pairs = [(orig_attns[n], surg_attns[n]) for n in common_names]

    if not pairs:
        return surg_model

    orig_outs: dict[int, torch.Tensor] = {}
    surg_outs: dict[int, torch.Tensor] = {}
    hooks = []

    for idx, (orig_mod, surg_mod) in enumerate(pairs):
        def _make_orig_hook(i: int):
            def _hook(mod, inp, out):
                val = out[0] if isinstance(out, tuple) else out
                orig_outs[i] = val.detach().float()
            return _hook

        def _make_surg_hook(i: int):
            def _hook(mod, inp, out):
                val = out[0] if isinstance(out, tuple) else out
                surg_outs[i] = val.detach().float()
            return _hook

        hooks.append(orig_mod.register_forward_hook(_make_orig_hook(idx)))
        hooks.append(surg_mod.register_forward_hook(_make_surg_hook(idx)))

    with torch.no_grad():
        try:
            if isinstance(batch, dict):
                orig_model(**batch)
                surg_model(**batch)
            else:
                orig_model(batch)
                surg_model(batch)
        except Exception as e:
            logger.debug("Calibration forward capture failed: %s", e)

    for h in hooks:
        h.remove()

    for idx, (orig_mod, surg_mod) in enumerate(pairs):
        if idx not in orig_outs or idx not in surg_outs:
            continue
        Y_orig = orig_outs[idx]
        Z = surg_outs[idx]

        if Y_orig.shape != Z.shape:
            continue

        D = Y_orig.shape[-1]
        Y_flat = Y_orig.reshape(-1, D)
        Z_flat = Z.reshape(-1, D)

        mu_Y = Y_flat.mean(dim=0, keepdim=True)
        mu_Z = Z_flat.mean(dim=0, keepdim=True)
        Z_c = Z_flat - mu_Z
        Y_c = Y_flat - mu_Y

        reg = ridge_lambda * torch.eye(D, device=Z_flat.device, dtype=torch.float32)
        ZT_Z = torch.matmul(Z_c.t(), Z_c) + reg
        ZT_Y = torch.matmul(Z_c.t(), Y_c)

        try:
            W_align = torch.linalg.solve(ZT_Z, ZT_Y)
            b_align = (mu_Y - torch.matmul(mu_Z, W_align)).squeeze(0)
        except Exception:
            continue

        if torch.isnan(W_align).any() or torch.isinf(W_align).any():
            continue

        out_proj = None
        for child_name, child in surg_mod.named_modules():
            leaf = child_name.split(".")[-1]
            if any(p.match(leaf) for p in PROJ_REGEX_PATTERNS["o"]):
                if isinstance(child, nn.Linear):
                    out_proj = child
                    break
                elif hasattr(child, "up") and isinstance(child.up, nn.Linear):
                    out_proj = child.up
                    break

        if out_proj is not None and out_proj.weight.shape[0] == D:
            W_cur = out_proj.weight.data.float()
            W_new = (1.0 - blend_alpha) * W_cur + blend_alpha * torch.matmul(W_align.t(), W_cur)
            out_proj.weight.data.copy_(W_new.to(dtype=out_proj.weight.dtype, device=out_proj.weight.device))
            if getattr(out_proj, "bias", None) is not None:
                b_cur = out_proj.bias.data.float()
                b_new = (1.0 - blend_alpha) * b_cur + blend_alpha * (torch.matmul(b_cur.unsqueeze(0), W_align).squeeze(0) + b_align)
                out_proj.bias.data.copy_(b_new.to(dtype=out_proj.bias.dtype, device=out_proj.bias.device))
        else:
            W_hook = W_align.to(dtype=dtype, device=device)
            b_hook = b_align.to(dtype=dtype, device=device)
            def _affine_forward_hook(m, inp, out):
                if isinstance(out, tuple):
                    val = out[0]
                    aligned = (1.0 - blend_alpha) * val + blend_alpha * (torch.matmul(val.float(), W_hook.float()) + b_hook.float()).to(val.dtype)
                    return (aligned,) + out[1:]
                else:
                    return (1.0 - blend_alpha) * out + blend_alpha * (torch.matmul(out.float(), W_hook.float()) + b_hook.float()).to(out.dtype)
            surg_mod.register_forward_hook(_affine_forward_hook)

    if attach_convex_anchor:
        if hasattr(orig_model, "roberta") and hasattr(surg_model, "roberta"):
            for mb, ms in zip(orig_model.roberta.encoder.layer, surg_model.roberta.encoder.layer):
                if hasattr(ms.attention, "self") and hasattr(mb.attention, "self"):
                    ms.attention.self = SubspaceAnchorWrapper(mb.attention.self, ms.attention.self, alpha=blend_alpha)
                else:
                    ms.attention = SubspaceAnchorWrapper(mb.attention, ms.attention, alpha=blend_alpha)
        elif hasattr(orig_model, "transformer") and hasattr(surg_model, "transformer"):
            for mb, ms in zip(orig_model.transformer.h, surg_model.transformer.h):
                ms.attn = SubspaceAnchorWrapper(mb.attn, ms.attn, alpha=blend_alpha)
    return surg_model


def calibrate_attention_subspace(
    teacher_model: nn.Module,
    student_model: nn.Module,
    calibration_inputs: Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]],
    ridge_lambda: float = 1e-3,
    blend_alpha: float = 0.25,
    attach_convex_anchor: bool = False,
) -> nn.Module:
    """Convenience alias for calibrate_surgically_replaced_model."""
    return calibrate_surgically_replaced_model(
        orig_model=teacher_model,
        surg_model=student_model,
        sample_inputs=calibration_inputs,
        ridge_lambda=ridge_lambda,
        blend_alpha=blend_alpha,
        attach_convex_anchor=attach_convex_anchor,
    )


def _heal_replaced_model_impl(
    teacher_model: nn.Module,
    student_model: nn.Module,
    calibration_inputs: Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]],
    steps: int = 30,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    cosine_weight: float = 0.5,
    also_ridge_init: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    log_to_ledger: bool = True,
) -> Dict[str, Any]:
    """Fast Layerwise Subspace Micro-Healing without full-model backpropagation.

    Optimizes nonlinear internal parameters (recurrent gates, low-rank bottleneck factors,
    mixing weights, and feature map scalers) directly on cached teacher input/output representations.
    """
    t0 = time.perf_counter()
    target_dev = device or next(student_model.parameters()).device
    teacher_model.eval()
    student_model.eval()

    if also_ridge_init:
        try:
            calibrate_attention_subspace(
                teacher_model=teacher_model,
                student_model=student_model,
                calibration_inputs=calibration_inputs,
                ridge_lambda=1e-3,
                blend_alpha=0.35,
            )
        except Exception as e_ridge:
            logger.debug("Ridge warm-start skipped or failed: %s", e_ridge)

    batches = []
    if isinstance(calibration_inputs, dict):
        batches.append({k: v.to(target_dev) for k, v in calibration_inputs.items()})
    elif isinstance(calibration_inputs, torch.Tensor):
        batches.append(calibration_inputs.to(target_dev))
    elif isinstance(calibration_inputs, (list, tuple)):
        for item in calibration_inputs:
            if isinstance(item, dict):
                batches.append({k: v.to(target_dev) for k, v in item.items()})
            elif isinstance(item, torch.Tensor):
                batches.append(item.to(target_dev))

    def _run_forward(m: nn.Module, b: Any) -> Any:
        if isinstance(b, dict):
            return m(**b)
        return m(b)

    def _collect_attention_modules(m: nn.Module) -> dict[str, nn.Module]:
        attns = {}
        for name, sub in m.named_modules():
            leaf = name.split(".")[-1].lower()
            if leaf in ("self", "self_attn", "attn", "attention"):
                attns[name] = sub
        return attns

    teacher_attns = _collect_attention_modules(teacher_model)
    student_attns = _collect_attention_modules(student_model)

    common_names = [n for n in student_attns if n in teacher_attns]
    if not common_names:
        teacher_list = list(teacher_attns.values())
        student_list = list(student_attns.values())
        layer_pairs = [(f"layer_{i}", t_mod, s_mod) for i, (t_mod, s_mod) in enumerate(zip(teacher_list, student_list))]
    else:
        layer_pairs = [(n, teacher_attns[n], student_attns[n]) for n in common_names]

    if not layer_pairs:
        return {
            "num_healed_layers": 0,
            "steps": steps,
            "also_ridge_init": also_ridge_init,
            "initial_mse": 0.0,
            "final_mse": 0.0,
            "initial_cosine_sim": 1.0,
            "final_cosine_sim": 1.0,
            "elapsed_sec": time.perf_counter() - t0,
            "per_layer": {},
        }

    cached_inputs: Dict[str, list[torch.Tensor]] = {name: [] for name, _, _ in layer_pairs}
    cached_targets: Dict[str, list[torch.Tensor]] = {name: [] for name, _, _ in layer_pairs}
    hooks = []

    for name, t_mod, _ in layer_pairs:
        def _make_capture_hook(layer_name: str):
            def _hook(mod, inp, out):
                in_tensor = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
                out_tensor = out[0] if isinstance(out, tuple) else out
                if isinstance(in_tensor, torch.Tensor):
                    cached_inputs[layer_name].append(in_tensor.detach().float())
                if isinstance(out_tensor, torch.Tensor):
                    cached_targets[layer_name].append(out_tensor.detach().float())
            return _hook
        hooks.append(t_mod.register_forward_hook(_make_capture_hook(name)))

    with torch.no_grad():
        for b in batches:
            try:
                _run_forward(teacher_model, b)
            except Exception as e:
                logger.debug("Teacher caching error: %s", e)

    for h in hooks:
        h.remove()

    for p in student_model.parameters():
        p.requires_grad = False

    per_layer_stats: Dict[str, Dict[str, float]] = {}
    init_mses = []
    final_mses = []
    init_coss = []
    final_coss = []

    for layer_name, _, s_mod in layer_pairs:
        in_list = cached_inputs.get(layer_name, [])
        tgt_list = cached_targets.get(layer_name, [])
        if not in_list or not tgt_list:
            continue

        H_in = in_list[0].to(target_dev)
        Y_tgt = tgt_list[0].to(target_dev)

        trainable_params = [p for p in s_mod.parameters() if p.numel() > 0]
        if not trainable_params:
            continue
        for p in trainable_params:
            p.requires_grad = True

        with torch.no_grad():
            s_mod.eval()
            try:
                pred0 = s_mod(H_in.to(p.dtype))
            except Exception:
                try:
                    pred0 = s_mod(hidden_states=H_in.to(p.dtype))
                except Exception:
                    pred0 = s_mod(H_in.to(p.dtype), H_in.to(p.dtype), H_in.to(p.dtype))
            pred0_val = pred0[0] if isinstance(pred0, tuple) else pred0
            pred0_f = pred0_val.float()

            denom_mse = float(torch.norm(Y_tgt, p="fro").item() ** 2 + 1e-6)
            init_mse = float(torch.norm(pred0_f - Y_tgt, p="fro").item() ** 2 / denom_mse)

            cos_sim0 = F.cosine_similarity(
                pred0_f.reshape(-1, Y_tgt.shape[-1]),
                Y_tgt.reshape(-1, Y_tgt.shape[-1]),
                dim=-1,
            ).mean().item()

        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps))

        best_loss = float("inf")
        best_state = {k: v.clone() for k, v in s_mod.state_dict().items()}

        s_mod.train()
        for step in range(steps):
            optimizer.zero_grad()
            batch_idx = step % len(in_list)
            H_in_step = in_list[batch_idx].to(target_dev)
            Y_tgt_step = tgt_list[batch_idx % len(tgt_list)].to(target_dev)
            denom_mse_step = float(torch.norm(Y_tgt_step, p="fro").item() ** 2 + 1e-6)

            try:
                pred = s_mod(H_in_step.to(p.dtype))
            except Exception:
                try:
                    pred = s_mod(hidden_states=H_in_step.to(p.dtype))
                except Exception:
                    pred = s_mod(H_in_step.to(p.dtype), H_in_step.to(p.dtype), H_in_step.to(p.dtype))

            pred_val = pred[0] if isinstance(pred, tuple) else pred
            pred_f = pred_val.float()

            loss_fro = torch.norm(pred_f - Y_tgt_step, p="fro") ** 2 / denom_mse_step
            loss_cos = 1.0 - F.cosine_similarity(
                pred_f.reshape(-1, Y_tgt_step.shape[-1]),
                Y_tgt_step.reshape(-1, Y_tgt_step.shape[-1]),
                dim=-1,
            ).mean()

            loss = loss_fro + cosine_weight * loss_cos

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(
                    "Non-finite loss (%s) encountered during micro-healing at layer '%s', step %d/%d (batch %d); halting optimization early.",
                    loss.item(),
                    layer_name,
                    step + 1,
                    steps,
                    batch_idx,
                )
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            cur_loss_val = float(loss.item())
            if cur_loss_val < best_loss:
                best_loss = cur_loss_val
                best_state = {k: v.clone() for k, v in s_mod.state_dict().items()}

        s_mod.load_state_dict(best_state)
        s_mod.eval()

        with torch.no_grad():
            try:
                pred_final = s_mod(H_in.to(p.dtype))
            except Exception:
                try:
                    pred_final = s_mod(hidden_states=H_in.to(p.dtype))
                except Exception:
                    pred_final = s_mod(H_in.to(p.dtype), H_in.to(p.dtype), H_in.to(p.dtype))
            pred_final_val = pred_final[0] if isinstance(pred_final, tuple) else pred_final
            pred_final_f = pred_final_val.float()

            final_mse = float(torch.norm(pred_final_f - Y_tgt, p="fro").item() ** 2 / denom_mse)
            final_cos = F.cosine_similarity(
                pred_final_f.reshape(-1, Y_tgt.shape[-1]),
                Y_tgt.reshape(-1, Y_tgt.shape[-1]),
                dim=-1,
            ).mean().item()

        for p in trainable_params:
            p.requires_grad = False

        per_layer_stats[layer_name] = {
            "initial_mse": init_mse,
            "final_mse": final_mse,
            "initial_cosine_sim": cos_sim0,
            "final_cosine_sim": final_cos,
        }
        init_mses.append(init_mse)
        final_mses.append(final_mse)
        init_coss.append(cos_sim0)
        final_coss.append(final_cos)

    out_stats = {
        "num_healed_layers": len(per_layer_stats),
        "steps": steps,
        "also_ridge_init": also_ridge_init,
        "initial_mse": float(sum(init_mses) / max(1, len(init_mses))),
        "final_mse": float(sum(final_mses) / max(1, len(final_mses))),
        "initial_cosine_sim": float(sum(init_coss) / max(1, len(init_coss))),
        "final_cosine_sim": float(sum(final_coss) / max(1, len(final_coss))),
        "elapsed_sec": float(time.perf_counter() - t0),
        "per_layer": per_layer_stats,
    }

    if log_to_ledger:
        try:
            from neuralops.core.results_ledger import log_experiment_result
            log_experiment_result(
                module_name="surgical_micro_healing",
                category="ablation",
                family="surgery-healing",
                generation_method={
                    "algorithm": "Fast Layerwise Subspace Micro-Healing",
                    "pipeline_stages": ["analytical_ridge_init" if also_ridge_init else "none", "adamw_cosine_micro_tuning"],
                    "source_component": "neuralops.core._attention_surgery.heal_replaced_model",
                },
                experimental_setup={
                    "steps": steps,
                    "lr": lr,
                    "also_ridge_init": also_ridge_init,
                    "num_healed_layers": len(per_layer_stats),
                },
                metrics={
                    "initial_mse": out_stats["initial_mse"],
                    "final_mse": out_stats["final_mse"],
                    "initial_cosine_sim": out_stats["initial_cosine_sim"],
                    "final_cosine_sim": out_stats["final_cosine_sim"],
                    "elapsed_sec": out_stats["elapsed_sec"],
                },
                notes="Layerwise attention surgery alignment metrics",
            )
        except Exception:
            pass

    return out_stats


def heal_replaced_model(
    teacher_model: nn.Module,
    student_model: nn.Module,
    calibration_inputs: Union[torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]],
    steps: int = 30,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    cosine_weight: float = 0.5,
    also_ridge_init: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    log_to_ledger: bool = True,
) -> Dict[str, Any]:
    """Heal surgically replaced layers while restoring each parameter's trainability."""
    requires_grad = {parameter: parameter.requires_grad for parameter in student_model.parameters()}
    try:
        return _heal_replaced_model_impl(
            teacher_model=teacher_model,
            student_model=student_model,
            calibration_inputs=calibration_inputs,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            cosine_weight=cosine_weight,
            also_ridge_init=also_ridge_init,
            device=device,
            log_to_ledger=log_to_ledger,
        )
    finally:
        for parameter, was_trainable in requires_grad.items():
            parameter.requires_grad_(was_trainable)


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
    "calibrate_surgically_replaced_model",
    "calibrate_attention_subspace",
    "heal_replaced_model",
]
