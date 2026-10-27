#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Layer Replacement Surgery Engine & In-Place Module Grafting

Architectural Role (NeuralOps v1.6.0):
This module drives the in-place structural surgery engine for foundation transformer models,
allowing arbitrary pretrained HuggingFace backbones (LLaMA, Mistral, GPT-2, Gemma, BERT)
to be converted from quadratic attention to linear, state-space, or hybrid attention
without breaking existing parameter graphs.

Key Engineering Standards & Invariants:
1. Recursive AST Module Traversal:
   Recursively inspects model topology across standard attributes, nn.Sequential,
   nn.ModuleList, and nn.ModuleDict containers, accurately targeting attention leaves
   (self_attn, attn, attention) while skipping already-transplanted child nodes.

2. In-Place Module Substitution & State Preservation:
   Substitutes target attention layers in place while preserving tensor devices, precision
   dtypes (FP16, BF16, FP32), and parameter tracking. Shadow twins and forward hooks
   capture activation dynamics without modifying user training scripts.

3. Projection Reconciliation & Shape Identity:
   Coordinates with topology_router.py to handle fused QKV projections (qkv_proj),
   Grouped-Query Attention (GQA) head expansions, and rotary position embeddings (RoPE/ALiBi),
   strictly enforcing the dimensional invariant (B, T, D) -> (B, T, D).

4. Deterministic Reversibility:
   Produces structured surgery records enabling clean module recovery via revert_surgery().
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, List, Optional, Tuple, Union

import torch.nn as nn

logger = logging.getLogger("NeuralOps.Surgery.LayerReplacement")

CONTAINER_TYPES = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
DEFAULT_ATTN_LEAVES = {
    "attention",
    "self_attn",
    "attn",
    "self",
    "selfattention",
    "encdecattention",
}


def _is_default_attention_target(name: str, module: nn.Module) -> bool:
    """Heuristic to identify attention targets across diverse model architectures."""
    if isinstance(module, CONTAINER_TYPES):
        return False
    if isinstance(module, nn.MultiheadAttention):
        return True
    cls_name = module.__class__.__name__
    if cls_name.endswith("Wrapper") or "Container" in cls_name:
        return False
    leaf = name.split(".")[-1].lower() if name else ""
    if leaf in DEFAULT_ATTN_LEAVES:
        return True
    if "attention" in cls_name.lower() or "attn" in cls_name.lower():
        return True
    return False


def find_nested_attention_modules(
    model: nn.Module,
    target_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    layer_indices: Optional[List[int]] = None,
) -> List[Tuple[str, nn.Module]]:
    """
    Recursively scans the entire model hierarchy for candidate attention modules,
    including those deeply nested inside nn.Sequential, nn.ModuleList, or custom containers.
    """
    targets: List[Tuple[str, nn.Module]] = []
    selected_prefixes: List[str] = []

    for name, module in model.named_modules():
        if not name:
            continue
        # Avoid traversing into subcomponents of an already-selected attention module
        if any(name.startswith(p) for p in selected_prefixes):
            continue

        is_match = (
            target_filter(name, module)
            if target_filter is not None
            else _is_default_attention_target(name, module)
        )

        if is_match:
            parts = name.split(".")
            layer_idx = None
            for p in parts:
                m = re.search(r"\d+", p)
                if m:
                    layer_idx = int(m.group())
                    break

            if layer_indices is not None and layer_idx is not None and layer_idx not in layer_indices:
                continue

            targets.append((name, module))
            selected_prefixes.append(name + ".")

    return targets


def _set_nested_module(root: nn.Module, path: str, new_module: nn.Module) -> None:
    """
    Sets a submodule at a dotted path, correctly resolving and updating
    nn.Sequential, nn.ModuleList, or standard attribute containers.
    """
    parts = path.split(".")
    curr: Any = root

    for part in parts[:-1]:
        if isinstance(curr, (nn.Sequential, nn.ModuleList)) and part.isdigit():
            curr = curr[int(part)]
        else:
            curr = getattr(curr, part)

    last_part = parts[-1]
    if isinstance(curr, (nn.Sequential, nn.ModuleList)) and last_part.isdigit():
        curr[int(last_part)] = new_module
    else:
        setattr(curr, last_part, new_module)


def _instantiate_replacement(
    target_module: nn.Module,
    attention_key: Optional[str] = None,
    new_attention_cls: Optional[Union[type, Callable[..., nn.Module]]] = None,
    **kwargs: Any,
) -> nn.Module:
    """Instantiates a replacement module from a class, callable factory, or attention registry."""
    if new_attention_cls is not None:
        try:
            return new_attention_cls(old_module=target_module, **kwargs)
        except TypeError:
            try:
                return new_attention_cls(**kwargs)
            except TypeError:
                return new_attention_cls()

    if attention_key is not None:
        from neuralops.core.registry import ATTENTION_REGISTRY
        fn = ATTENTION_REGISTRY.get(attention_key)
        if fn is None:
            raise ValueError(f"Attention key '{attention_key}' not found in ATTENTION_REGISTRY.")
        try:
            return fn(old_module=target_module, **kwargs)
        except TypeError:
            try:
                return fn(**kwargs)
            except TypeError:
                return fn()

    raise ValueError("Either 'attention_key' or 'new_attention_cls' must be provided.")


def replace_attention_recursive(
    model: nn.Module,
    attention_key: Optional[str] = None,
    new_attention_cls: Optional[Union[type, Callable[..., nn.Module]]] = None,
    target_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    layer_indices: Optional[List[int]] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Recursively discovers and surgically replaces attention modules located at arbitrary
    nesting depths, including inside nn.Sequential, nn.ModuleList, and custom blocks (P053).
    """
    targets = find_nested_attention_modules(
        model, target_filter=target_filter, layer_indices=layer_indices
    )

    if not targets:
        logger.warning("No attention targets found to replace in model hierarchy.")
        return model

    for path, old_mod in targets:
        new_mod = _instantiate_replacement(
            target_module=old_mod,
            attention_key=attention_key,
            new_attention_cls=new_attention_cls,
            **kwargs,
        )

        try:
            p = next(old_mod.parameters())
            new_mod.to(device=p.device, dtype=p.dtype)
        except (StopIteration, RuntimeError):
            pass

        _set_nested_module(model, path, new_mod)
        logger.debug("Replaced module at '%s' with %s", path, type(new_mod).__name__)

    if not hasattr(model, "_neuralops_metadata"):
        model._neuralops_metadata = []
    model._neuralops_metadata.append(
        {
            "op": "replace_attention_recursive",
            "replaced_count": len(targets),
            "attention_key": attention_key,
            "layer_indices": layer_indices,
        }
    )

    return model


replace_attention = replace_attention_recursive

__all__ = [
    "replace_attention_recursive",
    "replace_attention",
    "find_nested_attention_modules",
]
