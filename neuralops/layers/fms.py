#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Compatibility alias for neuralops.layers.svd.factorised_matrix_state.
"""

import neuralops.layers.svd.factorised_matrix_state as _mod
from neuralops.layers.svd.factorised_matrix_state import *  # noqa: F401, F403

__all__ = getattr(_mod, "__all__", [k for k in dir(_mod) if not k.startswith("_")])
