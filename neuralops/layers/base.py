#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
<<<<<<<< HEAD:neuralops/layers/base.py
"""Public facade for the shared attention base contract."""

from __future__ import annotations

from neuralops.core.base import BaseAttention

__all__ = ["BaseAttention"]
========
"""
Low-Rank Linear Transformation Core Module.
"""

from __future__ import annotations

from neuralops.core.low_rank.factorized import FactorisedLinear, FactorizedLinear
from neuralops.core.low_rank.svd_linear import SVDLinear

__all__ = ["SVDLinear", "FactorizedLinear", "FactorisedLinear"]
>>>>>>>> 07bedc4a84:neuralops/core/low_rank/__init__.py
