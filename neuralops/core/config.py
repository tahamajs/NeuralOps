"""
Core Configuration Parser

Lightweight YAML and dictionary configuration loader for core execution settings and kernel options.

Author: Taha Majlesi
"""

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
