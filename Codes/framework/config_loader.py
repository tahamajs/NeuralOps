"""Load configuration from YAML file and update CompressionConfig / ExperimentConfig."""

import yaml
import os
from pathlib import Path
from dataclasses import asdict
from framework.config import CompressionConfig

def load_yaml_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def apply_config_to_dataclass(config_dataclass, yaml_dict: dict, prefix: str = "") -> None:
    """
    Recursively apply YAML values to a dataclass instance.
    Only updates fields that exist in the dataclass.
    """
    for key, value in yaml_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if hasattr(config_dataclass, key):
            setattr(config_dataclass, key, value)
        # If the value is a dict and the field is a nested dataclass, we would
        # need a recursive approach; for now we only handle flat fields.

def merge_yaml_with_args(yaml_config: dict, args) -> CompressionConfig:
    """
    Build a CompressionConfig from YAML + CLI arguments.
    CLI arguments take precedence over YAML values.
    """
    # Start with defaults
    comp_config = CompressionConfig()

    # Override with YAML sections (if present)
    yaml_hardware = yaml_config.get('hardware', {})
    yaml_data = yaml_config.get('data', {})
    yaml_training = yaml_config.get('training', {})
    yaml_lora = yaml_config.get('lora', {})
    yaml_rdrop = yaml_config.get('rdrop', {})
    yaml_r3f = yaml_config.get('r3f', {})
    yaml_mixout = yaml_config.get('mixout', {})
    yaml_llrd = yaml_config.get('llrd', {})

    # Apply YAML values to CompressionConfig (flat fields)
    for section in [yaml_data, yaml_training, yaml_lora, yaml_rdrop, yaml_r3f, yaml_mixout, yaml_llrd]:
        apply_config_to_dataclass(comp_config, section)

    # CLU arguments override YAML
    for arg_name, value in vars(args).items():
        if hasattr(comp_config, arg_name) and value is not None:
            setattr(comp_config, arg_name, value)

    return comp_config