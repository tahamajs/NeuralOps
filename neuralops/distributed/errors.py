#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Error hierarchy for distributed execution in NeuralOps.
"""

from __future__ import annotations

from neuralops.core.errors import NeuralOpsError

E_DIST_INIT_FAILED = "E_DIST_INIT_FAILED"
E_DIST_NOT_INITIALIZED = "E_DIST_NOT_INITIALIZED"
E_DIST_COMM_FAILED = "E_DIST_COMM_FAILED"
E_FSDP2_CONFIG_FAILED = "E_FSDP2_CONFIG_FAILED"
E_FSDP2_NOT_INITIALIZED = "E_FSDP2_NOT_INITIALIZED"


class DistributedError(NeuralOpsError):
    """Base exception for all distributed runtime and communication errors."""

    def __init__(self, message: str, code: str = E_DIST_COMM_FAILED) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


__all__ = [
    "DistributedError",
    "E_DIST_INIT_FAILED",
    "E_DIST_NOT_INITIALIZED",
    "E_DIST_COMM_FAILED",
    "E_FSDP2_CONFIG_FAILED",
    "E_FSDP2_NOT_INITIALIZED",
]
