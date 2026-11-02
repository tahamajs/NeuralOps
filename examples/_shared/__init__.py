#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Shared utilities for NeuralOps examples. All examples import from here so
behavior is consistent across the suite.
"""

from __future__ import annotations

from examples._shared.benchmark import (
    compute_flops_estimate,
    measure_latency,
    measure_vram_if_available,
    print_benchmark_table,
)
from examples._shared.gpu import (
    get_backend_preference,
    gpu_available,
    gpu_name,
    safe_cuda_sync,
)
from examples._shared.mock_models import (
    TinyClassifier,
    baseline_output,
    tiny_inputs,
)
from examples._shared.offline_data import (
    synthetic_tokens,
    toy_calibration_loader,
    toy_dataloader,
)
from examples._shared.report import (
    kv,
    pass_fail,
    section,
    skip,
    table,
)
from examples._shared.verification import (
    assert_parity,
    assert_shape_unchanged,
    skip_if_missing,
)

__all__ = [
    # Benchmark
    "compute_flops_estimate",
    "measure_latency",
    "measure_vram_if_available",
    "print_benchmark_table",
    # GPU & Hardware
    "get_backend_preference",
    "gpu_available",
    "gpu_name",
    "safe_cuda_sync",
    # Mock Models
    "TinyClassifier",
    "baseline_output",
    "tiny_inputs",
    # Offline Data
    "synthetic_tokens",
    "toy_calibration_loader",
    "toy_dataloader",
    # Reporting
    "kv",
    "pass_fail",
    "section",
    "skip",
    "table",
    # Verification
    "assert_parity",
    "assert_shape_unchanged",
    "skip_if_missing",
]
