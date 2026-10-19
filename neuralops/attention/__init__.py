# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Canonical `neuralops.attention` subpackage.
Re-exports the Attention registry, BaseAttention, and all 8 architecture families.
"""

from __future__ import annotations

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

__all__ = ["ATTENTION_REGISTRY", "BaseAttention"]
