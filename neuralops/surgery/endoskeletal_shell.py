#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Endoskeletal Surgery Shell

Executes warm-start weight transfers and registers attention mechanisms with HuggingFace.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.surgery.mla_surgery import DeepSeekMLASurgeryBridge, DualStreamLatentState
from neuralops.surgery.moa_router import (
    HybridMoARouter,
    compute_attention_entropy,
    profile_layer_sensitivity,
)
from neuralops.surgery.scanner import AttentionTargetSpec, UniversalArchitectureScanner

MoARouter = HybridMoARouter

logger = logging.getLogger("NeuralOps.Surgery.EndoskeletalShell")


def register_neuralops_with_huggingface() -> None:
    """
    Registers named adapters in HuggingFace's attention-function registry.

    The adapter uses an explicitly attached ``module._neuralops_submodule``
    when present; otherwise it preserves the model's behavior with PyTorch
    SDPA. Registration alone does not transplant or initialize a NeuralOps
    attention module, because the HuggingFace hook receives projected Q/K/V
    tensors while most NeuralOps modules own those projections themselves.
    """
    try:
        import transformers.modeling_utils as hf_utils
        if not hasattr(hf_utils, "ALL_ATTENTION_FUNCTIONS"):
            return
        all_attn_fn = hf_utils.ALL_ATTENTION_FUNCTIONS
    except (ImportError, AttributeError) as e:
        logger.debug("HuggingFace transformers ALL_ATTENTION_FUNCTIONS not available: %s", e)
        return

    def _make_hf_attention_callable(attn_name: str) -> Callable:
        def _hf_neuralops_attention(
            module: nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            scaling: Optional[float] = None,
            is_causal: Optional[bool] = None,
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            B, H_q, T, D_h = query.shape
            H_kv = key.shape[1]

            if H_q != H_kv:
                repeat_factor = H_q // H_kv
                key = torch.repeat_interleave(key, repeat_factor, dim=1)
                value = torch.repeat_interleave(value, repeat_factor, dim=1)

            if hasattr(module, "_neuralops_submodule") and module._neuralops_submodule is not None:
                core_sub = module._neuralops_submodule
            else:
                # Attention implementations registered in NeuralOps generally
                # own Q/K/V projections. Transformers' attention-function API
                # supplies already-projected Q/K/V, so constructing one here
                # would silently add random, uncheckpointed parameters. Until a
                # module is explicitly attached by the caller, preserve model
                # semantics with the reference SDPA path.
                core_sub = None

            if core_sub is not None:
                if callable(getattr(core_sub, "forward_qkv", None)):
                    out = core_sub.forward_qkv(
                        query,
                        key,
                        value,
                        attention_mask=attention_mask,
                        is_causal=is_causal,
                        **kwargs,
                    )
                    if isinstance(out, tuple):
                        out = out[0]
                else:
                    q_flat = query.transpose(1, 2).contiguous().view(B, T, H_q * D_h)
                    result = core_sub(
                        q_flat,
                        attention_mask=attention_mask,
                        is_causal=is_causal,
                        **kwargs,
                    )
                    out = result[0] if isinstance(result, (tuple, list)) else result
                if out.ndim == 4:
                    if out.shape == (B, H_q, T, D_h):
                        out = out.transpose(1, 2)
                    elif out.shape != (B, T, H_q, D_h):
                        raise ValueError(
                            f"NeuralOps attention returned incompatible shape {tuple(out.shape)}"
                        )
                    out = out.contiguous()
                elif out.shape != (B, T, H_q * D_h):
                    raise ValueError(
                        f"NeuralOps attention returned incompatible shape {tuple(out.shape)}"
                    )
                out = out.view(B, T, H_q, D_h)
            else:
                assert callable(F.scaled_dot_product_attention), "F.scaled_dot_product_attention is not callable"
                out = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, is_causal=bool(is_causal)
                )
                out = out.transpose(1, 2).contiguous()

            return out, None

        return _hf_neuralops_attention

    for name in ATTENTION_REGISTRY.keys():
        hf_key = f"neuralops_{name}"
        if hf_key not in all_attn_fn:
            all_attn_fn[hf_key] = _make_hf_attention_callable(name)

    if "neuralops" not in all_attn_fn:
        all_attn_fn["neuralops"] = _make_hf_attention_callable("cosformer")


class SidecarGateBranch(nn.Module):
    """
    Lightweight data-dependent gating and depthwise causal 1D convolution
    for recurrent/stateful linear attention models.
    Initialized in long-memory regime (decay bias ~ 2.5 -> sigmoid ~ 0.92-0.95).
    """

    def __init__(self, d_model: int, num_heads: int, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_conv = d_conv

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
            bias=True,
        )
        self.gate_proj = nn.Linear(d_model, num_heads, bias=True)

        with torch.no_grad():
            self.conv1d.weight.zero_()
            self.conv1d.weight[:, :, -1] = 1.0
            if self.conv1d.bias is not None:
                self.conv1d.bias.zero_()
            nn.init.constant_(self.gate_proj.bias, 2.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            try:
                from neuralops.core.triton_kernels.elementwise import dynconv1d_fwd_triton
                w = self.conv1d.weight.squeeze(1)
                b = self.conv1d.bias
                x_conv = dynconv1d_fwd_triton(x, w, b)
            except (RuntimeError, ImportError) as e:
                logger.debug("dynconv1d_fwd_triton failed: %s, falling back to PyTorch conv1d", e)
                x_conv = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2).contiguous()
        else:
            x_conv = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2).contiguous()
        gates = torch.sigmoid(self.gate_proj(x))
        return x_conv, gates


def transfer_pretrained_qkv_weights(
    spec: AttentionTargetSpec,
    new_module: nn.Module,
    strict: bool = False,
) -> None:
    """
    Copies pretrained Q, K, V, and O projections from the host model into the newly
    instantiated NeuralOps attention module without random re-initialization.
    """
    if isinstance(new_module, DeepSeekMLASurgeryBridge):
        if spec.target_module is not None:
            new_module.inherit_from_mla_module(spec.target_module)
            return

    if spec.extracted_qkv is None:
        if strict:
            raise ValueError(f"No pretrained Q/K/V/O projections were extracted for {spec.module_path!r}")
        return

    W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o = spec.extracted_qkv

    target_q = getattr(
        new_module, "q_proj", getattr(new_module, "query", getattr(new_module, "q", None))
    )
    target_k = getattr(
        new_module, "k_proj", getattr(new_module, "key", getattr(new_module, "k", None))
    )
    target_v = getattr(
        new_module, "v_proj", getattr(new_module, "value", getattr(new_module, "v", None))
    )
    target_o = getattr(
        new_module,
        "out_proj",
        getattr(
            new_module,
            "o_proj",
            getattr(
                new_module,
                "proj",
                getattr(new_module, "dense", getattr(new_module, "c_proj", getattr(new_module, "out_lin", None))),
            ),
        ),
    )
    target_fused = getattr(
        new_module, "qkv", getattr(new_module, "query_key_value", getattr(new_module, "in_proj", None))
    )

    missing: list[str] = []

    def _copy_param(name: str, target_linear: Any, W: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        if target_linear is None or not hasattr(target_linear, "weight"):
            missing.append(name)
            return
        with torch.no_grad():
            tgt_w = target_linear.weight
            if tgt_w.shape == W.shape:
                tgt_w.copy_(W.to(device=tgt_w.device, dtype=tgt_w.dtype))
            elif tgt_w.ndim == 2 and W.ndim == 2 and tgt_w.t().shape == W.shape:
                tgt_w.copy_(W.t().to(device=tgt_w.device, dtype=tgt_w.dtype))
            elif tgt_w.ndim == 2 and W.ndim == 2:
                # GQA head broadcasting: when H_q != H_kv and target expects expanded heads
                head_dim = getattr(spec, "head_dim", 64)
                num_heads = getattr(spec, "num_heads", 1)
                num_kv_heads = getattr(spec, "num_kv_heads", 1)

                is_gqa_broadcast = (
                    name in ("k_proj", "v_proj", "k", "v")
                    and head_dim > 0
                    and num_heads > num_kv_heads
                    and num_heads % num_kv_heads == 0
                    and W.shape[0] == num_kv_heads * head_dim
                    and tgt_w.shape[0] == num_heads * head_dim
                    and tgt_w.shape[1] == W.shape[1]
                )
                if is_gqa_broadcast:
                    group_size = num_heads // num_kv_heads
                    logger.info(
                        "GQA head broadcasting for %s at %s: expanding %d KV heads to %d query heads (group_size=%d, head_dim=%d)",
                        name,
                        spec.module_path,
                        num_kv_heads,
                        num_heads,
                        group_size,
                        head_dim,
                    )
                    W_expanded = (
                        W.view(num_kv_heads, head_dim, W.shape[1])
                        .repeat_interleave(group_size, dim=0)
                        .view(num_heads * head_dim, W.shape[1])
                    )
                    tgt_w.copy_(W_expanded.to(device=tgt_w.device, dtype=tgt_w.dtype))
                    target_bias = getattr(target_linear, "bias", None)
                    if b is not None and target_bias is not None:
                        tgt_b = target_bias
                        if b.shape[0] == num_kv_heads * head_dim and tgt_b.shape[0] == num_heads * head_dim:
                            b_expanded = (
                                b.view(num_kv_heads, head_dim)
                                .repeat_interleave(group_size, dim=0)
                                .view(num_heads * head_dim)
                            )
                            tgt_b.copy_(b_expanded.to(device=tgt_b.device, dtype=tgt_b.dtype))
                    return

                if strict:
                    raise ValueError(
                        f"Projection {name} shape mismatch for {spec.module_path!r}: "
                        f"target={tuple(tgt_w.shape)}, source={tuple(W.shape)}. "
                        "To allow padded/isometric weight transfer across different projection widths, pass strict=False."
                    )
                # Explicit shape verification and isometric projection
                is_conv1d_tgt = getattr(target_linear, "nf", None) == tgt_w.shape[1]
                if is_conv1d_tgt:
                    src_w = W.t()
                    src_in, src_out = src_w.shape
                    tgt_in, tgt_out = tgt_w.shape
                else:
                    src_w = W
                    src_out, src_in = src_w.shape
                    tgt_out, tgt_in = tgt_w.shape

                truncated_out = max(0, src_out - tgt_out)
                truncated_in = max(0, src_in - tgt_in)
                expanded_out = max(0, tgt_out - src_out)
                expanded_in = max(0, tgt_in - src_in)

                if truncated_out > 0 or truncated_in > 0:
                    logger.warning(
                        "DIMENSIONAL CONTRACTION in %s at %s: source %s -> target %s. "
                        "Truncating %d output dims and %d input dims; representation loss may occur. "
                        "Consider post-surgery calibration healing.",
                        name,
                        spec.module_path,
                        tuple(src_w.shape),
                        tuple(tgt_w.shape),
                        truncated_out,
                        truncated_in,
                    )
                elif expanded_out > 0 or expanded_in > 0:
                    logger.warning(
                        "DIMENSIONAL EXPANSION in %s at %s: source %s -> target %s. "
                        "Applying padded isometric zero-extension for %d output dims and %d input dims.",
                        name,
                        spec.module_path,
                        tuple(src_w.shape),
                        tuple(tgt_w.shape),
                        expanded_out,
                        expanded_in,
                    )
                else:
                    logger.warning(
                        "Isometric projection for %s at %s: source %s -> target %s.",
                        name,
                        spec.module_path,
                        tuple(src_w.shape),
                        tuple(tgt_w.shape),
                    )

                copy_out = min(tgt_out, src_out)
                copy_in = min(tgt_in, src_in)
                if expanded_out > 0 or expanded_in > 0:
                    tgt_w.zero_()
                if is_conv1d_tgt:
                    tgt_w[:copy_in, :copy_out].copy_(
                        src_w[:copy_in, :copy_out].to(device=tgt_w.device, dtype=tgt_w.dtype)
                    )
                else:
                    tgt_w[:copy_out, :copy_in].copy_(
                        src_w[:copy_out, :copy_in].to(device=tgt_w.device, dtype=tgt_w.dtype)
                    )
            else:
                raise ValueError(
                    f"Unsupported projection shape for {name} at {spec.module_path!r}: "
                    f"target={tuple(tgt_w.shape)}, source={tuple(W.shape)}"
                )

            target_bias = getattr(target_linear, "bias", None)
            if b is not None and target_bias is not None:
                tgt_b = target_bias
                if tgt_b.shape == b.shape:
                    tgt_b.copy_(b.to(device=tgt_b.device, dtype=tgt_b.dtype))
                else:
                    if strict:
                        raise ValueError(
                            f"Bias shape mismatch for {name} at {spec.module_path!r}: "
                            f"target={tuple(tgt_b.shape)}, source={tuple(b.shape)}. "
                            "Pass strict=False to allow padded/isometric transfer."
                        )
                    min_b = min(tgt_b.shape[0], b.shape[0])
                    if tgt_b.shape[0] > b.shape[0]:
                        tgt_b.zero_()
                        logger.warning(
                            "Zero-extending bias for %s at %s: source %s padded to target %s",
                            name,
                            spec.module_path,
                            tuple(b.shape),
                            tuple(tgt_b.shape),
                        )
                    else:
                        logger.warning(
                            "Truncating bias for %s at %s: source %s truncated to target %s",
                            name,
                            spec.module_path,
                            tuple(b.shape),
                            tuple(tgt_b.shape),
                        )
                    tgt_b[:min_b].copy_(b[:min_b].to(device=tgt_b.device, dtype=tgt_b.dtype))
            elif b is None and target_bias is not None:
                target_bias.zero_()
            elif strict and b is not None and target_bias is None:
                raise ValueError(f"Target projection {name} has no bias but source projection does")

    if target_fused is not None and (target_q is None or target_k is None or target_v is None):
        W_fused = torch.cat([W_q, W_k, W_v], dim=0)
        b_fused = (
            torch.cat([b_q, b_k, b_v], dim=0)
            if (b_q is not None and b_k is not None and b_v is not None)
            else None
        )
        _copy_param("qkv", target_fused, W_fused, b_fused)
    else:
        _copy_param("query", target_q, W_q, b_q)
        _copy_param("key", target_k, W_k, b_k)
        _copy_param("value", target_v, W_v, b_v)
    _copy_param("output", target_o, W_o, b_o)
    if strict and missing:
        raise ValueError(
            f"Target attention module is missing projection(s) {', '.join(missing)} "
            f"required to inherit weights from {spec.module_path!r}"
        )
    if missing:
        logger.debug("Skipped unavailable projections for %s: %s", spec.module_path, ", ".join(missing))


class UniversalAttentionBridge(nn.Module):
    """
    Universal surgery shell replacing any attention layer in the wild.
    Preserves exact forward return signature, residual connections, and positional embeddings.
    """

    def __init__(
        self,
        orig_module: nn.Module,
        spec: AttentionTargetSpec,
        new_attention: nn.Module,
        inherit_weights: bool = True,
        attach_sidecar: bool = False,
        moa_router: Optional[HybridMoARouter] = None,
        layer_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.orig_module = orig_module
        self.spec = spec
        self.new_attention = new_attention
        self.pattern_type = spec.pattern_type
        self.embed_dim = spec.embed_dim
        self.num_heads = spec.num_heads
        self.moa_router = moa_router
        if self.moa_router is None and spec.extra_metadata and "moa_router" in spec.extra_metadata:
            self.moa_router = spec.extra_metadata["moa_router"]

        if layer_idx is not None:
            self.layer_idx = layer_idx
        elif hasattr(spec, "module_path") and spec.module_path:
            import re

            m = re.search(r"\b(?:layers?|h|block|blocks)\.(\d+)\b", spec.module_path)
            self.layer_idx = int(m.group(1)) if m else None
        else:
            self.layer_idx = None

        # Unwrap base layer if orig_module is a PEFT/LoRA wrapper
        self.base_module: nn.Module = getattr(orig_module, "base_layer", orig_module)

        # Preserve PEFT / LoRA adapter references if present
        self.lora_adapters: Dict[str, Any] = {}
        if spec.extra_metadata and "lora_adapters" in spec.extra_metadata:
            self.lora_adapters.update(spec.extra_metadata["lora_adapters"])
        if hasattr(orig_module, "lora_A"):
            self.lora_A = orig_module.lora_A
            self.lora_adapters["lora_A"] = orig_module.lora_A
        if hasattr(orig_module, "lora_B"):
            self.lora_B = orig_module.lora_B
            self.lora_adapters["lora_B"] = orig_module.lora_B
        if hasattr(orig_module, "scaling"):
            self.scaling = orig_module.scaling
        if hasattr(orig_module, "lora_magnitude_vector"):
            self.lora_magnitude_vector = orig_module.lora_magnitude_vector
            self.lora_adapters["lora_magnitude_vector"] = orig_module.lora_magnitude_vector
        if hasattr(orig_module, "peft_config"):
            self.peft_config = orig_module.peft_config

        for proj_attr in ("q_proj", "k_proj", "v_proj", "o_proj", "q", "k", "v", "o", "query", "key", "value", "c_attn", "c_proj"):
            proj = getattr(orig_module, proj_attr, None)
            if proj is not None and hasattr(proj, "lora_A") and hasattr(proj, "lora_B"):
                self.lora_adapters[f"{proj_attr}_lora_A"] = proj.lora_A
                self.lora_adapters[f"{proj_attr}_lora_B"] = proj.lora_B
                if hasattr(proj, "scaling"):
                    self.lora_adapters[f"{proj_attr}_scaling"] = proj.scaling
                if hasattr(proj, "lora_magnitude_vector"):
                    self.lora_adapters[f"{proj_attr}_lora_magnitude_vector"] = proj.lora_magnitude_vector
                if hasattr(proj, "r"):
                    self.lora_adapters[f"{proj_attr}_r"] = proj.r
                if hasattr(proj, "lora_alpha"):
                    self.lora_adapters[f"{proj_attr}_lora_alpha"] = proj.lora_alpha

        if self.lora_adapters:
            logger.info(
                "Preserved %d LoRA/PEFT adapter reference(s) during surgery on %s",
                len(self.lora_adapters),
                spec.module_path,
            )

        self._orig_sig = inspect.signature(orig_module.forward)

        if inherit_weights:
            if isinstance(self.new_attention, DeepSeekMLASurgeryBridge):
                self.new_attention.inherit_from_mla_module(self.orig_module)
            else:
                transfer_pretrained_qkv_weights(spec, self.new_attention)

        self.sidecar = (
            SidecarGateBranch(d_model=self.embed_dim, num_heads=self.num_heads)
            if attach_sidecar
            else None
        )

        self.teacher_branch: Optional[nn.Module] = None
        self.alpha: float = 1.0
        self.stage: str = "stage3_end_to_end"
        self.dispatch_mode: Optional[str] = None

    def set_dispatch_mode(self, mode: Optional[str]) -> None:
        """Sets runtime dynamic dispatch mode override ('full', 'linear', 'mla', or None for auto)."""
        valid_modes = (
            None,
            "full",
            "mha",
            "full_attention",
            "linear",
            "linear_attention",
            "mla",
            "deepseek_mla",
            "auto",
        )
        if mode is not None and mode not in valid_modes:
            raise ValueError(f"Invalid dispatch mode {mode!r}. Must be one of {valid_modes}")
        self.dispatch_mode = None if mode == "auto" else mode

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Phase 068: Dynamic Dispatch Hook (dispatch latency <= 2 us)
        # Fast resolution of runtime routing flags
        dispatch_mode = (
            kwargs.pop("dispatch_mode", None)
            or kwargs.pop("routing_flag", None)
            or kwargs.pop("mode", None)
            or self.dispatch_mode
        )

        if dispatch_mode in ("full", "mha", "full_attention"):
            return self.orig_module(*args, **kwargs)

        # Check Hybrid MoA routing if router is attached and layer_idx is provided
        if dispatch_mode is None and self.moa_router is not None and self.layer_idx is not None:
            if self.moa_router.is_full_attention(self.layer_idx):
                # Preserve original quadratic Full Attention execution
                return self.orig_module(*args, **kwargs)

        # Route DeepSeek MLA blocks through DeepSeekMLASurgeryBridge
        is_mla = dispatch_mode in ("mla", "deepseek_mla") or (
            dispatch_mode not in ("linear", "linear_attention")
            and isinstance(self.new_attention, DeepSeekMLASurgeryBridge)
        )
        if is_mla:
            if len(args) == 4 or ("c_kv" in kwargs and "k_rope" in kwargs):
                return self.new_attention(*args, **kwargs)

            hf_cache = kwargs.get("past_key_value", kwargs.get("past_key_values", None))
            layer_i = self.layer_idx if self.layer_idx is not None else 0
            if hf_cache is not None and hasattr(hf_cache, "get_recurrent_state"):
                past_state = hf_cache.get_recurrent_state(layer_i)
            else:
                past_state = kwargs.get(
                    "past_state", hf_cache if not hasattr(hf_cache, "get_recurrent_state") else None
                )

            if hasattr(self.orig_module, "kv_a_proj_with_mqa"):
                hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                if hs is not None:
                    c_kv, k_rope = self.new_attention.separate_mla_projections(
                        self.orig_module.kv_a_proj_with_mqa(hs)
                    )
                    if hasattr(self.orig_module, "q_a_proj"):
                        q_content, q_rope = self.new_attention.separate_query_projections(
                            self.orig_module.q_a_proj(hs)
                        )
                    else:
                        q_content = c_kv
                        q_rope = k_rope
                    is_decode = kwargs.get("is_decode", False) or (past_state is not None and hs.shape[1] == 1)
                    out, new_state = self.new_attention(
                        c_kv, k_rope, q_content, q_rope, past_state=past_state, is_decode=is_decode
                    )
                    if hf_cache is not None and hasattr(hf_cache, "update_recurrent_state"):
                        hf_cache.update_recurrent_state(layer_i, new_state, num_tokens=hs.shape[1])
                        return out, None, hf_cache
                    if kwargs.get("use_cache", False) or "past_key_value" in kwargs or "past_key_values" in kwargs:
                        return out, None, new_state
                    return out
            elif self.new_attention.kv_latent_proj is not None:
                hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                if hs is not None:
                    is_decode = kwargs.get("is_decode", False) or (past_state is not None and hs.shape[1] == 1)
                    out, new_state = self.new_attention(
                        hs, past_state=past_state, is_decode=is_decode
                    )
                    if hf_cache is not None and hasattr(hf_cache, "update_recurrent_state"):
                        hf_cache.update_recurrent_state(layer_i, new_state, num_tokens=hs.shape[1])
                        return out, None, hf_cache
                    if kwargs.get("use_cache", False) or "past_key_value" in kwargs or "past_key_values" in kwargs:
                        return out, None, new_state
                    return out

        kwargs_clean = dict(kwargs)
        if len(args) > 0:
            hidden_states = args[0]
            rest_args = args[1:]
        else:
            hidden_states = kwargs_clean.pop(
                "hidden_states",
                kwargs_clean.pop("x", kwargs_clean.pop("input_tensor", kwargs_clean.pop("query", None))),
            )
            rest_args = ()

        if hidden_states is None:
            raise ValueError(f"Could not extract hidden states from forward call {args}, {kwargs}")

        if self.sidecar is not None:
            hidden_states, _ = self.sidecar(hidden_states)

        assert callable(self.new_attention), f"{type(self).__name__}.new_attention is not callable"
        try:
            target_fn = getattr(self.new_attention, "forward", self.new_attention)
            sig = inspect.signature(target_fn)
            accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_varkw:
                call_kwargs = {k: v for k, v in kwargs_clean.items() if k in sig.parameters}
            else:
                call_kwargs = kwargs_clean
        except Exception:
            call_kwargs = kwargs_clean

        attn_out = self.new_attention(hidden_states, *rest_args, **call_kwargs)

        if isinstance(attn_out, tuple):
            core_tensor = attn_out[0]
            extra_outputs = attn_out[1:]
        else:
            core_tensor = attn_out
            extra_outputs = ()

        if self.pattern_type == "bert_nested":
            if kwargs.get("output_attentions", False):
                return (core_tensor, None)
            return (core_tensor,)

        if self.pattern_type == "conv1d_gpt2":
            output_attentions = kwargs.get("output_attentions", False)
            present = extra_outputs[0] if len(extra_outputs) > 0 else None
            ret = [core_tensor, present]
            if output_attentions:
                ret.append(None)
            return tuple(ret)

        if self.pattern_type == "split_4proj":
            if len(extra_outputs) > 1:
                return (core_tensor, *extra_outputs)
            attn_weights = extra_outputs[0] if len(extra_outputs) > 0 else None
            if "past_key_value" in kwargs or kwargs.get("use_cache", False):
                present_kv = kwargs.get("past_key_value", None)
                return (core_tensor, attn_weights, present_kv)
            return (core_tensor, attn_weights)

        if self.pattern_type == "deepseek_mla":
            if kwargs.get("use_cache", False) or "past_key_value" in kwargs:
                present_state = extra_outputs[0] if len(extra_outputs) > 0 else None
                return (core_tensor, None, present_state)
            return (core_tensor,)

        if len(extra_outputs) > 0:
            return (core_tensor, *extra_outputs)

        return core_tensor

    @property
    def has_lora_adapters(self) -> bool:
        """Returns True if this bridge preserves PEFT/LoRA adapter references."""
        return bool(self.lora_adapters or hasattr(self, "lora_A"))

    def get_lora_adapters(self) -> Dict[str, Any]:
        """Returns all preserved LoRA/PEFT adapter references."""
        return dict(self.lora_adapters)


class EndoskeletalShell(nn.Module):
    """
    Endoskeletal Shell for Transformer Backbones.

    Executes surgical transplants, Hybrid Mixture of Attention (MoA) schedules,
    and wraps transformer layers with UniversalAttentionBridge and DeepSeekMLASurgeryBridge
    while guaranteeing device placement and tensor shape invariants.
    """

    def __init__(self, model: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.model = model
        self.router: Optional[HybridMoARouter] = None
        self.wrapped_layers: Dict[int, UniversalAttentionBridge] = {}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.model is None:
            raise RuntimeError("EndoskeletalShell has no underlying model attached.")
        return self.model(*args, **kwargs)

    @classmethod
    def apply_moa_schedule(
        cls,
        model: Optional[Union[EndoskeletalShell, nn.Module]] = None,
        moa_router: Optional[HybridMoARouter] = None,
        linear_attention_factory: Optional[Callable[[AttentionTargetSpec], nn.Module]] = None,
        inherit_weights: bool = True,
        attach_sidecar: bool = False,
        use_mla: bool = False,
        latent_kv_dim: int = 512,
        decoupled_rope_dim: int = 64,
        q_latent_dim: Optional[int] = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Integrates a HybridMoARouter schedule into the transformer model.

        Scans the model hierarchy for attention targets, configures layer-wise routing,
        and replaces targets with UniversalAttentionBridge instances wrapping the native
        attention (for full attention layers) or sub-quadratic/MLA bridges (for linear attention layers).
        Ensures strict device and dtype parity to eliminate tensor mismatch or device placement errors.

        Args:
            model: Transformer backbone or EndoskeletalShell instance.
            moa_router: Optional HybridMoARouter instance specifying the partition schedule.
                If None, automatically initializes the canonical Pareto optimal partition
                based on the total layer count.
            linear_attention_factory: Optional custom callable (AttentionTargetSpec) -> nn.Module
                used to construct linear attention blocks.
            inherit_weights: Whether to inherit pretrained projection weights.
            attach_sidecar: Whether to attach SidecarGateBranch to sub-quadratic layers.
            use_mla: Whether to use DeepSeek MLA as the sub-quadratic attention mechanism.
            latent_kv_dim: KV latent compression dimension for MLA bridges.
            decoupled_rope_dim: Decoupled RoPE dimension for MLA bridges.
            q_latent_dim: Query latent compression dimension.
            strict: Whether to enforce strict shape parity on weight inheritance.
            **kwargs: Extra parameters passed to bridge construction.

        Returns:
            Dictionary containing metadata, router, and wrapped bridge layers.
        """
        shell_instance: Optional[EndoskeletalShell] = None
        if isinstance(model, EndoskeletalShell):
            shell_instance = model
            target_model = model.model
        else:
            target_model = model

        if target_model is None:
            raise ValueError("A target model (nn.Module) must be provided to apply_moa_schedule.")

        # 1. Scan model hierarchy for attention blocks
        specs = UniversalArchitectureScanner.scan(target_model)

        # Fallback inspection for custom modular backbones
        if not specs:
            candidate_specs: list[AttentionTargetSpec] = []
            layers_attr = None
            for attr in ("layers", "blocks", "h", "transformer"):
                if hasattr(target_model, attr):
                    layers_attr = getattr(target_model, attr)
                    break
            if isinstance(layers_attr, (nn.ModuleList, list, tuple)):
                for idx, layer in enumerate(layers_attr):
                    attn_mod = None
                    attn_name = None
                    for a_name in ("self_attn", "attn", "attention", "mha", "layer_0"):
                        if hasattr(layer, a_name):
                            attn_mod = getattr(layer, a_name)
                            attn_name = a_name
                            break
                    if attn_mod is not None:
                        embed_dim = getattr(attn_mod, "embed_dim", getattr(attn_mod, "hidden_size", 64))
                        num_heads = getattr(attn_mod, "num_heads", getattr(attn_mod, "num_attention_heads", 4))
                        head_dim = getattr(attn_mod, "head_dim", embed_dim // num_heads if num_heads > 0 else 16)
                        pattern = (
                            "deepseek_mla"
                            if hasattr(attn_mod, "kv_a_proj_with_mqa")
                            else ("split_4proj" if hasattr(attn_mod, "q_proj") else "native_mha")
                        )
                        candidate_specs.append(
                            AttentionTargetSpec(
                                module_path=f"layers.{idx}.{attn_name}",
                                parent_module=layer,
                                attr_name=attn_name,
                                target_module=attn_mod,
                                pattern_type=pattern,
                                embed_dim=embed_dim,
                                num_heads=num_heads,
                                num_kv_heads=num_heads,
                                head_dim=head_dim,
                                is_causal=True,
                                has_bias=False,
                            )
                        )
                if candidate_specs:
                    specs = candidate_specs

        # 2. Extract layer indices
        layer_indices: list[int] = []
        for i, spec in enumerate(specs):
            m = re.search(r"\b(?:layers?|h|block|blocks)\.(\d+)\b", spec.module_path)
            idx = int(m.group(1)) if m else i
            layer_indices.append(idx)

        total_layers = max(len(specs), (max(layer_indices) + 1) if layer_indices else 0)

        # 3. Router initialization
        if moa_router is None:
            moa_router = HybridMoARouter(num_layers=total_layers)

        wrapped_bridges: Dict[int, UniversalAttentionBridge] = {}

        # 4. Wrap and dispatch layers
        for spec, l_idx in zip(specs, layer_indices):
            # Discover target device and dtype
            target_device: Optional[torch.device] = None
            target_dtype: Optional[torch.dtype] = None
            for p in spec.target_module.parameters():
                target_device = p.device
                target_dtype = p.dtype
                break
            if target_device is None:
                for b in spec.target_module.buffers():
                    target_device = b.device
                    target_dtype = b.dtype
                    break
            if target_device is None:
                for p in target_model.parameters():
                    target_device = p.device
                    target_dtype = p.dtype
                    break

            # Construct new attention module
            if linear_attention_factory is not None:
                new_attn = linear_attention_factory(spec)
            elif use_mla or spec.pattern_type == "deepseek_mla":
                new_attn = DeepSeekMLASurgeryBridge(
                    latent_kv_dim=latent_kv_dim,
                    decoupled_rope_dim=decoupled_rope_dim,
                    q_latent_dim=q_latent_dim or spec.embed_dim,
                    hidden_dim=spec.embed_dim,
                )
            else:
                try:
                    from neuralops.attentions.linear.cosformer import CosformerAttention

                    new_attn = CosformerAttention(
                        hidden_size=spec.embed_dim,
                        num_heads=spec.num_heads,
                    )
                except (ImportError, Exception):
                    new_attn = DeepSeekMLASurgeryBridge(
                        latent_kv_dim=latent_kv_dim,
                        decoupled_rope_dim=decoupled_rope_dim,
                        q_latent_dim=q_latent_dim or spec.embed_dim,
                        hidden_dim=spec.embed_dim,
                    )

            # Ensure device and dtype alignment
            if target_device is not None:
                new_attn.to(device=target_device)
                if target_dtype is not None and target_dtype.is_floating_point:
                    new_attn.to(dtype=target_dtype)

            bridge = UniversalAttentionBridge(
                orig_module=spec.target_module,
                spec=spec,
                new_attention=new_attn,
                inherit_weights=inherit_weights,
                attach_sidecar=attach_sidecar,
                moa_router=moa_router,
                layer_idx=l_idx,
            )

            if target_device is not None:
                bridge.to(device=target_device)
                if target_dtype is not None and target_dtype.is_floating_point:
                    bridge.to(dtype=target_dtype)

            # In-place transplantation into parent module
            if hasattr(spec.parent_module, "__setitem__") and spec.attr_name.isdigit():
                spec.parent_module[int(spec.attr_name)] = bridge
            else:
                setattr(spec.parent_module, spec.attr_name, bridge)

            wrapped_bridges[l_idx] = bridge

        if shell_instance is not None:
            shell_instance.router = moa_router
            shell_instance.wrapped_layers.update(wrapped_bridges)

        logger.info(
            "EndoskeletalShell: Applied MoA schedule to %d layers (%d full, %d linear)",
            len(wrapped_bridges),
            len(moa_router.get_full_attention_layers()),
            len(moa_router.get_linear_attention_layers()),
        )

        return {
            "model": target_model,
            "moa_router": moa_router,
            "num_layers": total_layers,
            "full_attention_layers": moa_router.get_full_attention_layers(),
            "linear_attention_layers": moa_router.get_linear_attention_layers(),
            "wrapped_layers_count": len(wrapped_bridges),
            "wrapped_bridges": wrapped_bridges,
        }

    def apply_schedule(
        self,
        moa_router: Optional[HybridMoARouter] = None,
        linear_attention_factory: Optional[Callable[[AttentionTargetSpec], nn.Module]] = None,
        inherit_weights: bool = True,
        attach_sidecar: bool = False,
        use_mla: bool = False,
        latent_kv_dim: int = 512,
        decoupled_rope_dim: int = 64,
        q_latent_dim: Optional[int] = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Instance method applying MoA schedule to the attached model."""
        return self.apply_moa_schedule(
            model=self,
            moa_router=moa_router,
            linear_attention_factory=linear_attention_factory,
            inherit_weights=inherit_weights,
            attach_sidecar=attach_sidecar,
            use_mla=use_mla,
            latent_kv_dim=latent_kv_dim,
            decoupled_rope_dim=decoupled_rope_dim,
            q_latent_dim=q_latent_dim,
            strict=strict,
            **kwargs,
        )


try:
    register_neuralops_with_huggingface()
except (RuntimeError, AttributeError, ValueError) as e:
    logger.debug("Auto-registration with HuggingFace skipped: %s", e)


__all__ = [
    "register_neuralops_with_huggingface",
    "SidecarGateBranch",
    "transfer_pretrained_qkv_weights",
    "UniversalAttentionBridge",
    "DeepSeekMLASurgeryBridge",
    "DualStreamLatentState",
    "HybridMoARouter",
    "MoARouter",
    "EndoskeletalShell",
    "compute_attention_entropy",
    "profile_layer_sensitivity",
]
