# framework/utils/config.py
"""
Configuration management utilities.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to config file (.json or .yaml)
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        with open(config_path, 'r') as f:
            return json.load(f)


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to JSON or YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Output path
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a configuration dictionary.
    
    Args:
        config: Original config
        updates: Updates to apply
        
    Returns:
        Updated config
    """
    for key, value in updates.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configs with override taking precedence."""
    result = base.copy()
    result = update_config(result, override)
    return result