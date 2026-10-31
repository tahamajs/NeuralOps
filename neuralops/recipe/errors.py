#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe Language Error Types and Static Error Codes.
"""

from __future__ import annotations

from typing import Optional

# Error Codes per NRL v1.0.0 Specification Section 12
E_PARSE_JSON_SYNTAX = "E_PARSE_JSON_SYNTAX"
E_PARSE_YAML_SYNTAX = "E_PARSE_YAML_SYNTAX"
E_PARSE_DUP_KEY = "E_PARSE_DUP_KEY"

E_SCHEMA_MISSING_FIELD = "E_SCHEMA_MISSING_FIELD"
E_SCHEMA_UNKNOWN_FIELD = "E_SCHEMA_UNKNOWN_FIELD"
E_SCHEMA_WRONG_TYPE = "E_SCHEMA_WRONG_TYPE"
E_SCHEMA_INVALID_SEMVER = "E_SCHEMA_INVALID_SEMVER"
E_SCHEMA_INVALID_IDENTIFIER = "E_SCHEMA_INVALID_IDENTIFIER"

E_VALIDATION_LEVEL_INVALID = "E_VALIDATION_LEVEL_INVALID"
E_LANG_VERSION_TOO_NEW = "E_LANG_VERSION_TOO_NEW"
E_LANG_VERSION_TOO_OLD = "E_LANG_VERSION_TOO_OLD"


class RecipeError(Exception):
    """Base exception for all recipe parsing, schema, and validation errors."""

    def __init__(self, message: str, error_code: str, path: Optional[str] = None):
        super().__init__(f"[{error_code}] {message}" + (f" (at '{path}')" if path else ""))
        self.message = message
        self.error_code = error_code
        self.path = path


class RecipeParseError(RecipeError):
    """Raised when JSON or YAML deserialization fails."""


class RecipeSchemaError(RecipeError):
    """Raised when recipe content violates the normative EBNF grammar or schema rules."""
