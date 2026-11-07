#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Surgery Registry Infrastructure for NeuralOps Model Transformation Operators.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SurgeryMetadata:
    """Metadata describing a registered model surgery operator."""

    name: str
    category: str  # attention | mlp | layers | weights | embedding | recovery
    compat: list[str] = field(default_factory=lambda: ["llama", "qwen", "gpt2"])
    requires_gpu: bool = False
    requires_calibration: bool = False
    reversibility: str = "medium"  # high | medium | low | none
    cost_estimate: str = "medium"  # low | medium | high
    min_model_size_b: float = 0.1
    fn: Callable[..., Any] | None = None
    _experimental: bool = False


SURGERY_REGISTRY: dict[str, SurgeryMetadata] = {}


def register_surgery(
    name: str,
    category: str = "attention",
    compat: list[str] | None = None,
    requires_gpu: bool = False,
    requires_calibration: bool = False,
    reversibility: str = "medium",
    cost_estimate: str = "medium",
    min_model_size_b: float = 0.1,
    _experimental: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a surgery function in SURGERY_REGISTRY."""
    if compat is None:
        compat = ["llama", "qwen", "gpt2"]

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        meta = SurgeryMetadata(
            name=name,
            category=category,
            compat=compat,
            requires_gpu=requires_gpu,
            requires_calibration=requires_calibration,
            reversibility=reversibility,
            cost_estimate=cost_estimate,
            min_model_size_b=min_model_size_b,
            fn=fn,
            _experimental=_experimental,
        )
        SURGERY_REGISTRY[name] = meta

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            exp_flag = kwargs.pop("_experimental", None)
            if exp_flag is True or (exp_flag is None and meta._experimental):
                warnings.warn(
                    f"{name} is experimental - no external user verification yet",
                    UserWarning,
                    stacklevel=2,
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def list_surgeries(category: str | None = None) -> list[str]:
    """Return names of registered surgeries, optionally filtered by category."""
    if category is None:
        return sorted(list(SURGERY_REGISTRY.keys()))
    return sorted([name for name, meta in SURGERY_REGISTRY.items() if meta.category == category])


def get_surgery(name: str) -> SurgeryMetadata:
    """Retrieve metadata for a registered surgery operator."""
    if name not in SURGERY_REGISTRY:
        raise KeyError(
            f"Surgery operator '{name}' not found in SURGERY_REGISTRY. " f"Available surgeries: {list_surgeries()}"
        )
    return SURGERY_REGISTRY[name]


def get_surgeries_for_model(model_family: str) -> list[str]:
    """Return surgeries compatible with a specified model family."""
    fam = model_family.lower()
    return sorted(
        [
            name
            for name, meta in SURGERY_REGISTRY.items()
            if any(c.lower() in fam or fam in c.lower() for c in meta.compat)
        ]
    )


__all__ = [
    "SurgeryMetadata",
    "SURGERY_REGISTRY",
    "register_surgery",
    "list_surgeries",
    "get_surgery",
    "get_surgeries_for_model",
]
