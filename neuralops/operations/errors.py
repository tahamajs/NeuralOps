#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Operation and Registry Error Definitions.
"""

from __future__ import annotations

from typing import Optional

E_REGISTRY_UNKNOWN_OP = "E_REGISTRY_UNKNOWN_OP"
E_REGISTRY_UNKNOWN_VERSION = "E_REGISTRY_UNKNOWN_VERSION"
E_REGISTRY_DUPLICATE = "E_REGISTRY_DUPLICATE"

E_COMP_TYPE_MISMATCH = "E_COMP_TYPE_MISMATCH"
E_COMP_PRECONDITION_UNSATISFIED = "E_COMP_PRECONDITION_UNSATISFIED"
E_COMP_SELF_COMPOSE_DISALLOWED = "E_COMP_SELF_COMPOSE_DISALLOWED"

E_SCHEMA_MISSING_FIELD = "E_SCHEMA_MISSING_FIELD"
E_SCHEMA_UNKNOWN_FIELD = "E_SCHEMA_UNKNOWN_FIELD"
E_SCHEMA_WRONG_TYPE = "E_SCHEMA_WRONG_TYPE"
E_OP_VERSION_DEPRECATED = "E_OP_VERSION_DEPRECATED"


class OperationError(Exception):
    """Base exception for all operation and registry failures."""

    def __init__(self, message: str, error_code: str, operation_name: Optional[str] = None):
        prefix = f"[{error_code}]"
        if operation_name:
            prefix += f" ({operation_name})"
        super().__init__(f"{prefix} {message}")
        self.message = message
        self.error_code = error_code
        self.operation_name = operation_name


class OperationExecutionError(OperationError):
    """Raised when an operation fails during execution."""

    def __init__(self, message: str, operation_name: Optional[str] = None):
        super().__init__(message, error_code="E_OP_EXECUTION_FAILED", operation_name=operation_name)


class PreconditionFailedError(OperationError):
    """Raised when an operation precondition is unsatisfied."""

    def __init__(self, message: str, operation_name: Optional[str] = None):
        super().__init__(message, error_code=E_COMP_PRECONDITION_UNSATISFIED, operation_name=operation_name)
