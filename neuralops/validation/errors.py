#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Validation error definitions and error codes for NRL v1.0.0.
"""

from __future__ import annotations

from typing import List, Optional

# W1-W15 Error Codes
E_W1_INVALID_SYNTAX = "E_W1_INVALID_SYNTAX"
E_W2_MISSING_REQUIRED_FIELD = "E_W2_MISSING_REQUIRED_FIELD"
E_W3_EMPTY_OPERATIONS = "E_W3_EMPTY_OPERATIONS"
E_W4_INCOMPATIBLE_VERSION = "E_W4_INCOMPATIBLE_VERSION"
E_W5_MISSING_TARGET_MODEL = "E_W5_MISSING_TARGET_MODEL"
E_W6_INVALID_TARGET_PATTERN = "E_W6_INVALID_TARGET_PATTERN"
E_W7_INVALID_OPERATION_PARAMS = "E_W7_INVALID_OPERATION_PARAMS"
E_W8_PRECONDITION_FAILED = "E_W8_PRECONDITION_FAILED"
E_W9_POSTCONDITION_FAILED = "E_W9_POSTCONDITION_FAILED"
E_W10_TYPE_CHAIN_MISMATCH = "E_W10_TYPE_CHAIN_MISMATCH"
E_W11_UNSATISFIABLE_PRECONDITION = "E_W11_UNSATISFIABLE_PRECONDITION"
E_W12_COMPOSABILITY_VIOLATION = "E_W12_COMPOSABILITY_VIOLATION"
E_W13_IRREVERSIBLE_OPERATION = "E_W13_IRREVERSIBLE_OPERATION"
E_W14_MUTATION_SAFETY_VIOLATION = "E_W14_MUTATION_SAFETY_VIOLATION"
E_W15_DUPLICATE_OPERATION_ID = "E_W15_DUPLICATE_OPERATION_ID"

# Runtime Gate Error Codes
E_GATE_FATAL = "E_GATE_FATAL"
E_GATE_ERROR = "E_GATE_ERROR"
E_GATE_WARNING = "E_GATE_WARNING"


class ValidationError(Exception):
    """Base exception for all recipe and transformation validation errors."""

    def __init__(
        self,
        message: str,
        error_code: str,
        stage: Optional[str] = None,
        details: Optional[List[str]] = None,
    ):
        prefix = f"[{error_code}]"
        if stage:
            prefix += f" (stage: {stage})"
        super().__init__(f"{prefix} {message}")
        self.message = message
        self.error_code = error_code
        self.stage = stage
        self.details = details or []


class StaticValidationError(ValidationError):
    """Raised when a recipe violates static validation rules W1-W15."""


class RuntimeValidationError(ValidationError):
    """Raised when a transformation fails an active runtime validation gate."""


__all__ = [
    "E_GATE_ERROR",
    "E_GATE_FATAL",
    "E_GATE_WARNING",
    "E_W10_TYPE_CHAIN_MISMATCH",
    "E_W11_UNSATISFIABLE_PRECONDITION",
    "E_W12_COMPOSABILITY_VIOLATION",
    "E_W13_IRREVERSIBLE_OPERATION",
    "E_W14_MUTATION_SAFETY_VIOLATION",
    "E_W15_DUPLICATE_OPERATION_ID",
    "E_W1_INVALID_SYNTAX",
    "E_W2_MISSING_REQUIRED_FIELD",
    "E_W3_EMPTY_OPERATIONS",
    "E_W4_INCOMPATIBLE_VERSION",
    "E_W5_MISSING_TARGET_MODEL",
    "E_W6_INVALID_TARGET_PATTERN",
    "E_W7_INVALID_OPERATION_PARAMS",
    "E_W8_PRECONDITION_FAILED",
    "E_W9_POSTCONDITION_FAILED",
    "RuntimeValidationError",
    "StaticValidationError",
    "ValidationError",
]
