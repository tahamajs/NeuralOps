#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Gated Linear Attention (GLA) Recurrent Scan Operators.

Forwarding interface for GLA recurrence, chunked parallel scans, and step updates.
"""

from __future__ import annotations

from neuralops.ops.triton.gla_scan import (
    chunked_gla_scan,
    chunked_gla_scan_pytorch,
    chunked_gla_scan_triton,
    gla_recurrent_step_triton,
    gla_scan,
    gla_scan_pytorch,
    gla_scan_triton,
    kda_hla_chunk_scan_triton,
)

__all__ = [
    "gla_scan",
    "gla_scan_triton",
    "gla_scan_pytorch",
    "chunked_gla_scan",
    "chunked_gla_scan_triton",
    "chunked_gla_scan_pytorch",
    "gla_recurrent_step_triton",
    "kda_hla_chunk_scan_triton",
]
