"""Simple YAML config loader."""

import yaml
from typing import Dict, Any

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config(config: Dict[str, Any], path: str):
    with open(path, "w") as f:
        yaml.safe_dump(config, f)