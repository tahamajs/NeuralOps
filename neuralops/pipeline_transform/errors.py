#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Exceptions for Pipeline Planning and Transformation Operations.
"""

from __future__ import annotations

from typing import Optional

from neuralops.core.errors import NeuralOpsError


class PlanningError(NeuralOpsError):
    """Raised when recipe planning fails due to unmet preconditions or missing adapters."""

    def __init__(
        self,
        message: str,
        error_code: str = "E_PLAN_FAILED",
        stage: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.stage = stage

    def __str__(self) -> str:
        prefix = f"[{self.error_code}]"
        if self.stage:
            prefix += f" (stage '{self.stage}')"
        return f"{prefix} {self.message}"


class ConfigurationError(NeuralOpsError):
    """Raised when auto-configuration cannot satisfy specified hardware constraints."""

    def __init__(
        self,
        message: str,
        constraints: Optional[dict] = None,
        error_code: str = "E_CONFIG_UNSATISFIABLE",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.constraints = constraints or {}
        self.error_code = error_code

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message} (constraints: {self.constraints})"


__all__ = ["ConfigurationError", "PlanningError"]
