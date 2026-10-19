# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Grouped Query Attention Compatibility Shim

Backward-compatibility redirection module routing legacy GQA imports to neuralops.core.utils.gqa.
"""

import warnings
warnings.warn(
    "neuralops.core.gqa_utils is deprecated. Use neuralops.core.utils.gqa.",
    DeprecationWarning,
    stacklevel=2,
)
from neuralops.core.utils.gqa import *  # noqa: F401,F403
from neuralops.core.utils.gqa import repeat_kv  # noqa: F401

repeat_kv_optimized = repeat_kv
