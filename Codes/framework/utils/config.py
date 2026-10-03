# framework/utils/config.py
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load config from disk or config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix in (".yaml", ".yml"):
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        with open(path) as f:
            return json.load(f)


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
