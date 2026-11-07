#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Validation Subpackage.
"""

from .base import Gate, GateResult, GateSeverity
from .data_validator import DataValidator
from .memory_validator import MemoryTracker
from .model_validator import ModelValidator
from .numerical_validator import NumericalValidator
from .profiles import ValidationProfile, get_gates_for_profile
from .recipe_validator import (
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
    SymplecticNumericalGuard,
    ValidationError,
    check_condition_number,
    check_numerical_stability,
    sanitize_non_finite,
    symplectic_guarded_softmax,
    validate_recipe_static,
)
from .report_generator import ReportGenerator
from .result import ValidationResult
from .runner import run_validation_gates
from .speed_validator import SpeedValidator
from .training_validator import TrainingValidator, ValidationStatus

__all__ = [
    # Top-level validators & runners
    "DataValidator",
    "ModelValidator",
    "NumericalValidator",
    "MemoryTracker",
    "SpeedValidator",
    "TrainingValidator",
    "ValidationStatus",
    "ReportGenerator",
    "Gate",
    "GateResult",
    "GateSeverity",
    "ValidationProfile",
    "ValidationResult",
    "get_gates_for_profile",
    "run_validation_gates",
    # Recipe validation errors and guards (F-57)
    "E_GATE_ERROR",
    "E_GATE_FATAL",
    "E_GATE_WARNING",
    "E_W1_INVALID_SYNTAX",
    "E_W2_MISSING_REQUIRED_FIELD",
    "E_W3_EMPTY_OPERATIONS",
    "E_W4_INCOMPATIBLE_VERSION",
    "E_W5_MISSING_TARGET_MODEL",
    "E_W6_INVALID_TARGET_PATTERN",
    "E_W7_INVALID_OPERATION_PARAMS",
    "E_W8_PRECONDITION_FAILED",
    "E_W9_POSTCONDITION_FAILED",
    "E_W10_TYPE_CHAIN_MISMATCH",
    "E_W11_UNSATISFIABLE_PRECONDITION",
    "E_W12_COMPOSABILITY_VIOLATION",
    "E_W13_IRREVERSIBLE_OPERATION",
    "E_W14_MUTATION_SAFETY_VIOLATION",
    "E_W15_DUPLICATE_OPERATION_ID",
    "RuntimeValidationError",
    "StaticValidationError",
    "ValidationError",
    "SymplecticNumericalGuard",
    "check_condition_number",
    "check_numerical_stability",
    "sanitize_non_finite",
    "symplectic_guarded_softmax",
    "validate_recipe_static",
]
