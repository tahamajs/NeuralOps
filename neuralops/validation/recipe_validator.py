#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe and Symplectic Numerical Guardrail Validation Engine.
Consolidated from legacy neuralops.validator into canonical neuralops.validation.
"""

from __future__ import annotations

from neuralops.validation.errors import (
    E_GATE_ERROR,
    E_GATE_FATAL,
    E_GATE_WARNING,
    E_W1_INVALID_SYNTAX,
    E_W2_MISSING_REQUIRED_FIELD,
    E_W3_EMPTY_OPERATIONS,
    E_W4_INCOMPATIBLE_VERSION,
    E_W5_MISSING_TARGET_MODEL,
    E_W6_INVALID_TARGET_PATTERN,
    E_W7_INVALID_OPERATION_PARAMS,
    E_W8_PRECONDITION_FAILED,
    E_W9_POSTCONDITION_FAILED,
    E_W10_TYPE_CHAIN_MISMATCH,
    E_W11_UNSATISFIABLE_PRECONDITION,
    E_W12_COMPOSABILITY_VIOLATION,
    E_W13_IRREVERSIBLE_OPERATION,
    E_W14_MUTATION_SAFETY_VIOLATION,
    E_W15_DUPLICATE_OPERATION_ID,
    RuntimeValidationError,
    StaticValidationError,
    ValidationError,
)
from neuralops.validation.numerical_guardrails import (
    SymplecticNumericalGuard,
    check_condition_number,
    check_numerical_stability,
    sanitize_non_finite,
    symplectic_guarded_softmax,
)
from neuralops.validation.static import validate_recipe_static

__all__ = [
    # Errors
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
    # Guardrails
    "SymplecticNumericalGuard",
    "check_condition_number",
    "check_numerical_stability",
    "sanitize_non_finite",
    "symplectic_guarded_softmax",
    # Static validation
    "validate_recipe_static",
]
