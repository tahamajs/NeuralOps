#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Architecture Adapter Errors.
"""

from __future__ import annotations

from typing import Optional

E_ADAPTER_UNKNOWN = "E_ADAPTER_UNKNOWN"
E_ADAPTER_CAPABILITY_MISMATCH = "E_ADAPTER_CAPABILITY_MISMATCH"


class AdapterError(Exception):
    """Base exception for architecture adapter resolution and capability checks."""

    def __init__(self, message: str, error_code: str, family: Optional[str] = None):
        prefix = f"[{error_code}]"
        if family:
            prefix += f" (family: {family})"
        super().__init__(f"{prefix} {message}")
        self.message = message
        self.error_code = error_code
        self.family = family
