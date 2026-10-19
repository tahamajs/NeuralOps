# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Core Configuration Parser

Lightweight YAML and dictionary configuration loader for core execution settings and kernel options.
"""

from __future__ import annotations

from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Load config from disk or config."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str) -> None:
    """Persist config to disk."""
    with open(path, "w") as f:
        yaml.safe_dump(config, f)


__all__ = ["load_config", "save_config"]

