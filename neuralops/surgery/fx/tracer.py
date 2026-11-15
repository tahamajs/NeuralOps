# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tracer module for FX Attention Surgery.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("neuralops.surgery.fx.tracer")

import torch.nn.functional as F

logger = logging.getLogger("neuralops.surgery.fx")

# Core module types that should never be traced internally to avoid graph explosion
BASE_LEAF_MODULE_TYPES: List[Type[nn.Module]] = [
    nn.LayerNorm,
    nn.Linear,
    nn.Embedding,
    nn.Dropout,
]
if hasattr(nn, "RMSNorm"):
    BASE_LEAF_MODULE_TYPES.append(nn.RMSNorm)

LEAF_MODULE_TYPES: Tuple[Type[nn.Module], ...] = tuple(BASE_LEAF_MODULE_TYPES)

LEAF_MODULE_NAMES: Tuple[str, ...] = (
    "LayerNorm",
    "RMSNorm",
    "LlamaRMSNorm",
    "GemmaRMSNorm",
    "LlamaRotaryEmbedding",
    "RotaryEmbedding",
    "Linear4bit",
    "Linear8bitLt",
    "Linear8bit",
    "WQLinear",
    "QuantLinear",
    "PagedAttention",
    "Embedding",
    "Conv1d",
)

_SDPA_TARGETS: Tuple[Any, ...] = tuple(
    t
    for t in (
        getattr(getattr(torch.ops, "aten", None), "scaled_dot_product_attention", None),
        getattr(F, "scaled_dot_product_attention", None),
    )
    if t is not None
)


def _sanitize_graph_inputs(
    model: Union[nn.Module, Callable[..., Any], Any],
    concrete_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Synthesizes default mock arguments for causal decoder models to bypass dynamic control flow."""
    canonical_defaults: Dict[str, Any] = {
        "attention_mask": None,
        "position_ids": None,
        "past_key_value": None,
        "past_key_values": None,
        "use_cache": False,
        "output_attentions": False,
        "output_hidden_states": False,
        "return_dict": False,
    }

    sanitized: Dict[str, Any] = {}
    target_fn = getattr(model, "forward", model)
    if callable(target_fn):
        try:
            sig = inspect.signature(target_fn)
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            for param_name in sig.parameters.keys():
                if param_name in canonical_defaults:
                    sanitized[param_name] = canonical_defaults[param_name]
            if has_var_keyword:
                for k, v in canonical_defaults.items():
                    if k not in sanitized:
                        sanitized[k] = v
        except (ValueError, TypeError):
            sanitized.update(canonical_defaults)
    else:
        sanitized.update(canonical_defaults)

    if not sanitized and not (callable(target_fn) and hasattr(model, "forward")):
        sanitized.update(canonical_defaults)

    if concrete_args:
        sanitized.update(concrete_args)

    return sanitized


class NeuralOpsAttentionTracer(torch.fx.Tracer):
    """Symbolic FX Tracer preserving dynamic batch and sequence dimensions."""

    def __init__(
        self,
        leaf_modules: Optional[List[Type[nn.Module]]] = None,
        autowrap_modules: Tuple[Any, ...] = (torch.ops.aten,),
        autowrap_functions: Tuple[Callable[..., Any], ...] = (),
        param_shapes_constant: bool = True,
    ) -> None:
        super().__init__(
            autowrap_modules=autowrap_modules,
            autowrap_functions=autowrap_functions,
            param_shapes_constant=param_shapes_constant,
        )
        self.leaf_classes: Set[Type[nn.Module]] = set(LEAF_MODULE_TYPES)
        if leaf_modules:
            self.leaf_classes.update(leaf_modules)
        self.dynamic_dims: Dict[str, Tuple[int, ...]] = {}

    def _is_dynamic_dim(self, dim_idx: int) -> bool:
        """Determines whether a tensor dimension is marked as dynamic (B=0, T=1 or -2) vs static (H, D)."""
        return dim_idx in (0, 1, -2)

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        """Determines whether a module should be treated as an atomic leaf node in the FX graph."""
        if isinstance(m, tuple(self.leaf_classes)):
            return True
        name = m.__class__.__name__
        if any(leaf_name in name for leaf_name in LEAF_MODULE_NAMES):
            return True
        if getattr(m, "_is_leaf_module", False):
            return True
        return super().is_leaf_module(m, module_qualified_name)

    def create_node(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        """Creates a node in the graph and annotates dynamic shape metadata."""
        node = super().create_node(
            kind=kind,
            target=target,
            args=args,
            kwargs=kwargs,
            name=name,
            type_expr=type_expr,
        )
        if "example_value" not in node.meta:
            node.meta["dynamic_dims"] = tuple(self._is_dynamic_dim(i) for i in range(4))
        return node

    def create_proxy(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        proxy_factory_fn: Optional[Callable[[torch.fx.Node], torch.fx.Proxy]] = None,
    ) -> torch.fx.Proxy:
        """Creates a proxy node, ensuring dynamic sequence shape references are preserved."""
        return super().create_proxy(
            kind=kind,
            target=target,
            args=args,
            kwargs=kwargs,
            name=name,
            type_expr=type_expr,
            proxy_factory_fn=proxy_factory_fn,
        )

    def trace(
        self,
        root: Union[nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
    ) -> torch.fx.Graph:
        """Traces the root module with control-flow sanitization and dynamic sequence proxy support."""
        sanitized_args = _sanitize_graph_inputs(root, concrete_args)
        return super().trace(root, concrete_args=sanitized_args if sanitized_args else None)
