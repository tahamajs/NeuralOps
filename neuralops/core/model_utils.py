#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transformer Model Utilities

Model loading, parameter freezing, weight inspection, and buffer verification helper functions.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any

import torch
import torch.nn as nn


def run_calibration_forward(
    model: nn.Module,
    tokenizer: Any,
    texts: list[str],
    device: torch.device | None = None,
    max_length: int = 128,
) -> nn.Module:
    """Run a single forward pass on calibration texts; return model (in eval)."""
    model.eval()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)
    return model


ATTENTION_WEIGHT_MAP = {
    "query": ["q_lin", "q_proj", "query", "W_q", "q"],
    "key": ["k_lin", "k_proj", "key", "W_k", "k"],
    "value": ["v_lin", "v_proj", "value", "W_v", "v"],
    "out_proj": ["out_lin", "o_proj", "output", "dense", "out_proj", "c_proj", "wo", "proj"],
}


def copy_attention_weights(
    src_module: nn.Module,
    dst_module: nn.Module,
    source_map: dict[str, list[str]] | None = None,
) -> None:
    """
    Copy pretrained attention weights from src to dst with shape-adaptive head projection.
    Tries dedicated dst_module.copy_weights first, then attribute-by-attribute.
    """
    if hasattr(dst_module, "copy_weights") and callable(dst_module.copy_weights):
        try:
            dst_module.copy_weights(src_module)
            return
        except Exception:
            pass

    if source_map is None:
        source_map = ATTENTION_WEIGHT_MAP

    dst_aliases = {
        "query": ["query", "q_proj", "W_q", "q_lin", "q"],
        "key": ["key", "k_proj", "W_k", "k_lin", "k"],
        "value": ["value", "v_proj", "W_v", "v_lin", "v"],
        "out_proj": ["out_proj", "o_proj", "dense", "out_lin", "c_proj", "wo"],
    }

    head_dim = getattr(dst_module, "head_dim", getattr(src_module, "head_dim", None))

    for target_attr, src_attrs in source_map.items():
        src = None
        for attr in src_attrs:
            if hasattr(src_module, attr):
                cand = getattr(src_module, attr)
                if isinstance(cand, nn.Module) and hasattr(cand, "weight"):
                    src = cand
                    break
                elif hasattr(cand, "dense") and hasattr(cand.dense, "weight"):
                    src = cand.dense
                    break
                elif hasattr(cand, "weight"):
                    src = cand
                    break

        dst = None
        candidates = dst_aliases.get(target_attr, [target_attr])
        for cand_name in candidates:
            if hasattr(dst_module, cand_name):
                c = getattr(dst_module, cand_name)
                if hasattr(c, "weight"):
                    dst = c
                    break

        if src is None:
            if target_attr == "out_proj" and dst is not None:
                with torch.no_grad():
                    min_d = min(dst.weight.shape[0], dst.weight.shape[1])
                    dst.weight.data.zero_()
                    dst.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if dst.bias is not None:
                        nn.init.zeros_(dst.bias)
                if hasattr(dst_module, "is_nested_encoder"):
                    dst_module.is_nested_encoder = True
                continue
            raise ValueError(
                f"Cannot find source for '{target_attr}' in {src_module.__class__.__name__}. Tried: {src_attrs}"
            )

        if dst is None:
            continue

        with torch.no_grad():
            w_src = src.weight.data
            w_dst = dst.weight.data

            if w_src.ndim > 2:
                w_src = w_src.reshape(w_src.shape[0], -1)
            if w_dst.ndim > 2:
                w_dst = w_dst.reshape(w_dst.shape[0], -1)

            if w_src.shape != w_dst.shape and w_dst.shape == w_src.t().shape:
                w_src = w_src.t().contiguous()

            dst_out, dst_in = w_dst.shape
            src_out, src_in = w_src.shape

            h_dim = head_dim
            if h_dim is None:
                for hd in (64, 128, 32, 16, 256):
                    if dst_out % hd == 0 and src_out % hd == 0:
                        h_dim = hd
                        break
                else:
                    h_dim = dst_out

            if dst_out == src_out and dst_in == src_in:
                dst.weight.data.copy_(w_src)
                if src.bias is not None and dst.bias is not None:
                    dst.bias.data.copy_(src.bias.data)
                elif dst.bias is not None:
                    nn.init.zeros_(dst.bias)
            elif dst_out < src_out and dst_in == src_in:
                src_heads = max(1, src_out // h_dim)
                dst_heads = max(1, dst_out // h_dim)
                if src_heads > dst_heads and src_heads % dst_heads == 0:
                    group = src_heads // dst_heads
                    scale = math.sqrt(group)
                    w_pooled = (w_src.view(dst_heads, group, h_dim, src_in).mean(dim=1) * scale).reshape(dst_out, dst_in)
                    dst.weight.data.copy_(w_pooled)
                    if src.bias is not None and dst.bias is not None:
                        b_pooled = (src.bias.data.view(dst_heads, group, h_dim).mean(dim=1) * scale).reshape(dst_out)
                        dst.bias.data.copy_(b_pooled)
                    elif dst.bias is not None:
                        nn.init.zeros_(dst.bias)
                else:
                    dst.weight.data.copy_(w_src[:dst_out])
                    if src.bias is not None and dst.bias is not None:
                        dst.bias.data.copy_(src.bias.data[:dst_out])
                    elif dst.bias is not None:
                        nn.init.zeros_(dst.bias)
            elif dst_out > src_out and dst_out % src_out == 0:
                rep = dst_out // src_out
                dst.weight.data.copy_(w_src.repeat(rep, 1))
                if src.bias is not None and dst.bias is not None:
                    dst.bias.data.copy_(src.bias.data.repeat(rep))
                elif dst.bias is not None:
                    nn.init.zeros_(dst.bias)
            else:
                out_d = min(dst_out, src_out)
                in_d = min(dst_in, src_in)
                dst.weight.data.zero_()
                dst.weight.data[:out_d, :in_d].copy_(w_src[:out_d, :in_d])
                if src.bias is not None and dst.bias is not None:
                    mb = min(dst.bias.numel(), src.bias.numel())
                    dst.bias.data[:mb].copy_(src.bias.data[:mb])
                elif dst.bias is not None:
                    nn.init.zeros_(dst.bias)

            if hasattr(dst, "freeze_base_weights") and callable(dst.freeze_base_weights):
                dst.freeze_base_weights()


def replace_module(model: nn.Module, full_name: str, new_module: nn.Module) -> None:
    """Replace a sub-module identified by its dotted name, supporting ModuleList and ModuleDict."""
    parts = [p for p in full_name.split(".") if p]
    if not parts:
        if hasattr(model, "q_proj"):
            model.q_proj = new_module
        return
    parent = model
    for part in parts[:-1]:
        if isinstance(parent, (nn.ModuleList, list)):
            parent = parent[int(part)]
        elif isinstance(parent, (nn.ModuleDict, dict)):
            parent = parent[part]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if isinstance(parent, (nn.ModuleList, list)):
        parent[int(last)] = new_module
    elif isinstance(parent, (nn.ModuleDict, dict)):
        parent[last] = new_module
    else:
        setattr(parent, last, new_module)


def get_parent_and_child_name(model: nn.Module, full_name: str) -> Any:
    """Return (parent_module, child_attribute_name), supporting ModuleList and ModuleDict."""
    parts = [p for p in full_name.split(".") if p]
    if not parts:
        return model, ""
    parent = model
    for part in parts[:-1]:
        if isinstance(parent, (nn.ModuleList, list)):
            parent = parent[int(part)]
        elif isinstance(parent, (nn.ModuleDict, dict)):
            parent = parent[part]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


count_params = count_parameters


@contextmanager
def activation_hooks(
    model: nn.Module,
    target_module_type: type = nn.Linear,
    collect_input: bool = True,
    collect_output: bool = False,
) -> Any:
    """Context manager that collects activations for all modules of a given type."""
    activations: dict[str, Any] = {}
    hooks = []

    def make_hook(name: str) -> Any:
        """Make hook."""

        def hook(module: nn.Module, input: tuple[Any, ...], output: Any) -> None:
            """Hook."""
            data = {}
            if collect_input:
                data["input"] = input[0].detach()
            if collect_output:
                data["output"] = output.detach() if isinstance(output, torch.Tensor) else output[0].detach()
            activations[name] = data

        return hook

    for name, module in model.named_modules():
        if isinstance(module, target_module_type):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        yield activations
    finally:
        for hook in hooks:
            hook.remove()


def detect_decoder(
    config: Any = None,
    is_decoder: bool | None = None,
    source_attn: nn.Module | None = None,
    kwargs: dict[str, Any] | None = None,
) -> bool:
    """Detect whether the attention layer is being used as a causal decoder."""
    if is_decoder is not None:
        return bool(is_decoder)
    if kwargs:
        if kwargs.get("is_decoder") or kwargs.get("is_causal"):
            return True
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "is_causal", False):
            return True
        mt = str(getattr(config, "model_type", "") or "").lower()
        if any(
            mt.startswith(p)
            for p in (
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
    if source_attn is not None:
        if (
            getattr(source_attn, "is_decoder", False)
            or hasattr(source_attn, "c_attn")
            or "gpt" in source_attn.__class__.__name__.lower()
        ):
            return True
    return False


def detect_return_format(source_attn: nn.Module | None, is_nested_encoder: bool) -> str:
    """Detect the expected output tuple format (bert, gpt2, or llama)."""
    if is_nested_encoder:
        return "bert"
    if source_attn is None:
        return "llama"
    cls = source_attn.__class__.__name__.lower()
    if any(k in cls for k in ("gpt2", "gptne", "gptj", "gptneo", "gpt_neo", "gpt_j")) or hasattr(source_attn, "c_attn"):
        return "gpt2"
    return "llama"


_detect_decoder = detect_decoder
_detect_return_format = detect_return_format
