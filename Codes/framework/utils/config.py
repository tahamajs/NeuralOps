# framework/utils/config.py
import json
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix in ('.yaml', '.yml'):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(path, 'r') as f:
            return json.load(f)

def save_config(config: Dict[str, Any], config_path: str):
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in ('.yaml', '.yml'):
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    return config

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = base.copy()
    return update_config(result, override)