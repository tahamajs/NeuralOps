#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Historical strict baseline benchmark for 26 Triton kernels (archived)."""

# Legacy paper script archived per Phase 9 of BENCHMARK_CLEANUP_AND_BUG_AUDIT.md.
# See canonical implementation in neuralops.benchmarks.benchmark_all_26_comprehensive.

import warnings

warnings.warn(
    "Deprecated: This script has been quarantined in _legacy and is scheduled for removal. "
    "Please use canonical NeuralOps CLI v2 ('neuralops benchmark') instead.",
    DeprecationWarning,
    stacklevel=2,
)