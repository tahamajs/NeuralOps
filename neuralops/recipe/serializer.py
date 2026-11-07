#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe Serializer with YAML/JSON Support and Round-Trip Guarantees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.recipe.schema import Recipe


def dumps_recipe(recipe: Recipe, format: str = "yaml") -> str:
    """Serialize a Recipe into a string representation."""
    d = recipe.to_dict()
    if format.lower() == "json":
        return json.dumps(d, indent=2, ensure_ascii=False)
    elif format.lower() in ("yaml", "yml"):
        try:
            import yaml
            return yaml.safe_dump(d, sort_keys=False, default_flow_style=False)
        except ImportError as exc:
            raise IntegrationNotImplementedError(
                "PyYAML is required to serialize recipe to YAML format. "
                "Install with 'pip install pyyaml' or serialize to JSON with format='json'."
            ) from exc
    else:
        raise ValueError(f"Unsupported recipe serialization format: '{format}'. Expected 'yaml' or 'json'.")


def save_recipe(recipe: Recipe, path: str | Path, format: Optional[str] = None) -> None:
    """Save a Recipe instance to a filesystem path."""
    dest = Path(path)
    if format is None:
        format = "json" if dest.suffix.lower() == ".json" else "yaml"

    content = dumps_recipe(recipe, format=format)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
