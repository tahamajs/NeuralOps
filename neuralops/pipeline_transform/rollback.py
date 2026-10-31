#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transactional Rollback Exception Definitions.
"""

from __future__ import annotations

from neuralops.core.errors import NeuralOpsError


class RollbackError(NeuralOpsError):
    """Raised when transactional restoration of a model snapshot fails."""

    def __init__(
        self,
        message: str,
        error_code: str = "E_EXEC_ROLLBACK_FAILED",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


__all__ = ["RollbackError"]
