#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Attention Layers Tier (Tier 2)

Houses all subquadratic attention families (linear, svd, ssm, gated, hybrid, sparse, adaptive, modern, fast_kernels, dense, experimental).
"""

from __future__ import annotations

from typing import Any

from neuralops._optional import _require_torch
_require_torch("neuralops.layers")

import neuralops.attentions as _attentions
from neuralops.core.registry import ATTENTION_REGISTRY, get_attention, list_attentions

ATTENTION_REGISTRY = ATTENTION_REGISTRY
get_attention = get_attention
list_attentions = list_attentions


def __getattr__(name: str) -> Any:
    return getattr(_attentions, name)


def __dir__() -> list[str]:
    return list(_attentions.__dir__()) + ["ATTENTION_REGISTRY", "get_attention", "list_attentions"]


__all__ = sorted(list(set(__dir__())))
