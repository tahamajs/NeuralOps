#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Configuration Utilities (Delegating wrapper around neuralops.config.loader).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuralops.config.loader import load_config as _canonical_load_config
import yaml


def load_config(config_path: str | Path | None = None, **kwargs: Any) -> Any:
    """Delegate configuration loading to canonical neuralops.config.loader.load_config."""
    return _canonical_load_config(yaml_path=config_path, **kwargs)


def save_config(config: dict[str, Any], config_path: str) -> None:
    """Persist config to disk."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in (".yaml", ".yml"):
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(path, "w") as f:
            json.dump(config, f, indent=2)


def update_config(config: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Update config."""
    for key, value in updates.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    return config


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge configs."""
    result = base.copy()
    return update_config(result, override)


__all__ = ["load_config", "save_config", "update_config", "merge_configs"]
