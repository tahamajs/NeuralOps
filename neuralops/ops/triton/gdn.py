#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gated DeltaNet (GDN) Chunked Scan Operators.

Forwarding interface for Gated DeltaNet state updates and chunked scans.
"""

from __future__ import annotations

from neuralops.ops.triton.gdn_scan import (
    gdn_recurrent_step_triton,
    gdn_scan_pytorch,
    gdn_scan_triton,
    triton_chunked_delta_gate_scan,
    triton_chunked_delta_gate_scan_pytorch_ref,
)

__all__ = [
    "gdn_scan_triton",
    "gdn_scan_pytorch",
    "gdn_recurrent_step_triton",
    "triton_chunked_delta_gate_scan",
    "triton_chunked_delta_gate_scan_pytorch_ref",
]
