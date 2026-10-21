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
from typing import Any, Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.surgery.scanner import AttentionTargetSpec

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


try:
    register_neuralops_with_huggingface()
except (RuntimeError, AttributeError, ValueError) as e:
    logger.debug("Auto-registration with HuggingFace skipped: %s", e)


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
                if strict:
                    raise ValueError(
                        f"Projection {name} shape mismatch for {spec.module_path!r}: "
                        f"target={tuple(tgt_w.shape)}, source={tuple(W.shape)}"
                    )
                if getattr(target_linear, "nf", None) == tgt_w.shape[1]:
                    min_in = min(tgt_w.shape[0], W.shape[1])
                    min_out = min(tgt_w.shape[1], W.shape[0])
                    tgt_w[:min_in, :min_out].copy_(
                        W[:min_out, :min_in].t().to(device=tgt_w.device, dtype=tgt_w.dtype)
                    )
                else:
                    min_out = min(tgt_w.shape[0], W.shape[0])
                    min_in = min(tgt_w.shape[1], W.shape[1])
                    tgt_w[:min_out, :min_in].copy_(
                        W[:min_out, :min_in].to(device=tgt_w.device, dtype=tgt_w.dtype)
                    )
                logger.warning(
                    "Copied overlapping weights for %s at %s: source %s, target %s",
                    name,
                    spec.module_path,
                    tuple(W.shape),
                    tuple(tgt_w.shape),
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
                            f"target={tuple(tgt_b.shape)}, source={tuple(b.shape)}"
                        )
                    min_b = min(tgt_b.shape[0], b.shape[0])
                    tgt_b[:min_b].copy_(b[:min_b].to(device=tgt_b.device, dtype=tgt_b.dtype))
            elif b is None and target_bias is not None:
                target_bias.zero_()
            elif strict and b is not None and target_bias is None:
                raise ValueError(f"Target projection {name} has no bias but source projection does")

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
    ) -> None:
        super().__init__()
        self.orig_module = orig_module
        self.spec = spec
        self.new_attention = new_attention
        self.pattern_type = spec.pattern_type
        self.embed_dim = spec.embed_dim
        self.num_heads = spec.num_heads

        self._orig_sig = inspect.signature(orig_module.forward)

        if inherit_weights:
            transfer_pretrained_qkv_weights(spec, self.new_attention)

        self.sidecar = (
            SidecarGateBranch(d_model=self.embed_dim, num_heads=self.num_heads)
            if attach_sidecar
            else None
        )

        self.teacher_branch: Optional[nn.Module] = None
        self.alpha: float = 1.0
        self.stage: str = "stage3_end_to_end"

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) > 0:
            hidden_states = args[0]
            rest_args = args[1:]
        else:
            hidden_states = kwargs.get(
                "hidden_states",
                kwargs.get("x", kwargs.get("input_tensor", kwargs.get("query", None))),
            )
            rest_args = ()

        if hidden_states is None:
            raise ValueError(f"Could not extract hidden states from forward call {args}, {kwargs}")

        if self.sidecar is not None:
            hidden_states, _ = self.sidecar(hidden_states)

        assert callable(self.new_attention), f"{type(self).__name__}.new_attention is not callable"
        attn_out = self.new_attention(hidden_states, *rest_args, **kwargs)

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
            attn_weights = extra_outputs[0] if len(extra_outputs) > 0 else None
            return (core_tensor, attn_weights)

        if len(extra_outputs) > 0:
            return (core_tensor, *extra_outputs)

        return core_tensor


__all__ = [
    "register_neuralops_with_huggingface",
    "SidecarGateBranch",
    "transfer_pretrained_qkv_weights",
    "UniversalAttentionBridge",
]
