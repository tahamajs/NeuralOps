#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Configuration Loader

Hierarchical configuration parser supporting YAML specifications, CLI overrides, and validation.
"""

import argparse
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

import yaml

from .config import Config

T = TypeVar("T")


def load_yaml(path: str) -> dict:
    """Load yaml from disk or config."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli_args(cfg_dict: dict, args: argparse.Namespace) -> dict:
    """Override YAML values with CLI arguments where not None."""
    cli = vars(args)
    for key, val in cli.items():
        if val is not None and key in cfg_dict:
            cfg_dict[key] = val
    return cfg_dict


def _dict_to_dataclass(cls: type[T], data: dict) -> T:
    """Recursively convert dict → dataclass."""
    kwargs = {}
    for f in fields(cls):
        key = f.name
        if key in data:
            val = data[key]
            if is_dataclass(f.type) and isinstance(val, dict):
                kwargs[key] = _dict_to_dataclass(f.type, val)
            else:
                kwargs[key] = val
    return cls(**kwargs)


def load_config(
    yaml_path: str | None = None,
    cli_args: argparse.Namespace | None = None,
) -> Config:
    """
    Load configuration from optional YAML file + CLI overrides.
    Always returns a fully populated Config dataclass with defaults.
    """
    cfg_dict: dict[str, Any] = {}
    if yaml_path:
        cfg_dict = load_yaml(yaml_path)
    if cli_args:
        cfg_dict = merge_cli_args(cfg_dict, cli_args)

    return _dict_to_dataclass(Config, cfg_dict)
