#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Canonical Form Normalization and SHA-256 Recipe Hashing.

Implements Specification Section 11 (Canonical Form and Hashing).
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

from neuralops.recipe.schema import Recipe


def _canonicalize_value(val: Any) -> Any:
    if isinstance(val, str):
        return unicodedata.normalize("NFC", val)
    elif isinstance(val, dict):
        # Sort keys lexicographically by UTF-8 byte value
        return {
            unicodedata.normalize("NFC", k): _canonicalize_value(v)
            for k, v in sorted(val.items(), key=lambda item: item[0].encode("utf-8"))
        }
    elif isinstance(val, (list, tuple)):
        return [_canonicalize_value(x) for x in val]
    elif isinstance(val, float):
        if val.is_integer():
            return int(val)
        return val
    return val


def canonical_bytes(recipe_or_dict: Recipe | dict[str, Any]) -> bytes:
    """Serialize a Recipe into canonical, deterministic UTF-8 bytes."""
    if isinstance(recipe_or_dict, Recipe):
        raw = recipe_or_dict.to_dict()
    else:
        raw = recipe_or_dict

    normalized = _canonicalize_value(raw)
    # separators=(',', ':') eliminates trailing and extra whitespace
    encoded_str = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return encoded_str.encode("utf-8")


def canonicalize(recipe_or_dict: Recipe | dict[str, Any]) -> bytes:
    """Alias for canonical_bytes per specification naming."""
    return canonical_bytes(recipe_or_dict)


def recipe_hash(recipe_or_dict: Recipe | dict[str, Any]) -> str:
    """Compute cryptographic SHA-256 hash over the canonical byte stream."""
    b = canonical_bytes(recipe_or_dict)
    digest = hashlib.sha256(b).hexdigest()
    return f"sha256:{digest}"
