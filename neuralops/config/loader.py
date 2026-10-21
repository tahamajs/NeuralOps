#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Typed YAML loading for the NeuralOps configuration schemas."""

from __future__ import annotations

import argparse
import csv
import copy
import types
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from .config import Config
from .experiment_config import ExperimentConfig
from .policy_config import BlocklistEntry
from .sweep_config import AblationPipelineConfig, AttentionSweepConfig, QualitySweepConfig

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping, rejecting empty or non-mapping documents."""
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration root in {path!s} must be a mapping")
    return dict(data)


def merge_cli_args(cfg_dict: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply non-None CLI values, including dotted nested keys, to a config mapping."""
    merged = copy.deepcopy(cfg_dict)
    for raw_key, value in vars(args).items():
        if value is None:
            continue
        keys = raw_key.split(".")
        target = merged
        for key in keys[:-1]:
            child = target.get(key)
            if child is None:
                child = {}
                target[key] = child
            if not isinstance(child, dict):
                raise ValueError(f"Cannot apply CLI override {raw_key!r}: {key!r} is not a mapping")
            target = child
        target[keys[-1]] = value
    return merged


def _coerce_value(type_hint: Any, value: Any, path: str) -> Any:
    """Convert nested YAML values according to resolved dataclass annotations."""
    if type_hint is Any:
        return value

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        failures = []
        for option in args:
            if option is type(None):
                continue
            try:
                return _coerce_value(option, value, path)
            except (TypeError, ValueError) as exc:
                failures.append(str(exc))
        raise TypeError(f"{path} does not match {type_hint}: {'; '.join(failures)}")

    if isinstance(type_hint, type) and is_dataclass(type_hint):
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping for {type_hint.__name__}")
        return _dict_to_dataclass(type_hint, value, path)

    if origin in (list, Sequence):
        if not isinstance(value, list):
            raise TypeError(f"{path} must be a list")
        item_type = args[0] if args else Any
        return [_coerce_value(item_type, item, f"{path}[{index}]") for index, item in enumerate(value)]

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path} must be a list or tuple")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value(args[0], item, f"{path}[{index}]") for index, item in enumerate(value))
        if args and len(value) != len(args):
            raise ValueError(f"{path} must contain exactly {len(args)} items")
        return tuple(
            _coerce_value(item_type, item, f"{path}[{index}]")
            for index, (item_type, item) in enumerate(zip(args, value))
        )

    if origin is dict:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping")
        key_type, value_type = args if len(args) == 2 else (Any, Any)
        return {
            _coerce_value(key_type, key, f"{path}.<key>"): _coerce_value(value_type, item, f"{path}.{key}")
            for key, item in value.items()
        }

    if type_hint is int and isinstance(value, float) and not value.is_integer():
        raise TypeError(f"{path} must be an integer, got fractional value {value!r}")
    if type_hint in (str, int, float, bool) and not isinstance(value, type_hint):
        if type_hint is bool or isinstance(value, bool):
            raise TypeError(f"{path} must be {type_hint.__name__}, got {type(value).__name__}")
        try:
            return type_hint(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{path} must be coercible to {type_hint.__name__}") from exc
    return value


def _dict_to_dataclass(cls: type[T], data: Mapping[str, Any], path: str = "config") -> T:
    """Build a dataclass recursively and fail on misspelled/unsupported keys."""
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must be a mapping")
    dataclass_fields = {item.name: item for item in fields(cls)}
    unknown = set(data) - dataclass_fields.keys()
    if unknown:
        formatted = ", ".join(sorted(map(str, unknown)))
        raise ValueError(f"Unknown {cls.__name__} field(s) at {path}: {formatted}")

    hints = get_type_hints(cls)
    kwargs = {
        key: _coerce_value(hints.get(key, item.type), value, f"{path}.{key}")
        for key, value in data.items()
        for item in (dataclass_fields[key],)
    }
    return cls(**kwargs)


def _load_typed_config(
    config_type: type[T],
    yaml_path: str | Path | None,
    cli_args: argparse.Namespace | None,
) -> T:
    config_data = load_yaml(yaml_path) if yaml_path is not None else {}
    if cli_args is not None:
        config_data = merge_cli_args(config_data, cli_args)
    return _dict_to_dataclass(config_type, config_data)


def load_config(
    yaml_path: str | Path | None = None,
    cli_args: argparse.Namespace | None = None,
) -> Config:
    """Load the canonical nested ``Config`` schema from YAML and CLI overrides."""
    return _load_typed_config(Config, yaml_path, cli_args)


def load_experiment_config(
    yaml_path: str | Path | None = None,
    cli_args: argparse.Namespace | None = None,
) -> ExperimentConfig:
    """Load the flat experiment/CLI schema used by ``neuralops.main``."""
    return _load_typed_config(ExperimentConfig, yaml_path, cli_args)


def load_attention_sweep_config(yaml_path: str | Path) -> AttentionSweepConfig:
    """Load and validate the attention-sweep preset schema."""
    return _load_typed_config(AttentionSweepConfig, yaml_path, None)


def load_quality_sweep_config(yaml_path: str | Path) -> QualitySweepConfig:
    """Load and validate architecture-specific quality sweep settings."""
    return _load_typed_config(QualitySweepConfig, yaml_path, None)


def load_ablation_pipeline(yaml_path: str | Path) -> AblationPipelineConfig:
    """Load an ordered ablation pipeline and its typed stage records."""
    return _load_typed_config(AblationPipelineConfig, yaml_path, None)


def load_blocklist(path: str | Path) -> dict[str, BlocklistEntry]:
    """Read a CSV-formatted benchmark policy list with line-aware errors."""
    entries: dict[str, BlocklistEntry] = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = next(csv.reader([line], skipinitialspace=True))
            except csv.Error as exc:
                raise ValueError(f"Invalid blocklist CSV at {path}:{line_number}: {exc}") from exc
            if len(row) != 3 or not all(value.strip() for value in row):
                raise ValueError(f"Expected module, reason code, and justification at {path}:{line_number}")
            name, reason_code, justification = (value.strip() for value in row)
            if name in entries:
                raise ValueError(f"Duplicate blocklist module {name!r} at {path}:{line_number}")
            entries[name] = BlocklistEntry(name, reason_code, justification)
    return entries


__all__ = [
    "load_yaml",
    "merge_cli_args",
    "load_config",
    "load_experiment_config",
    "load_attention_sweep_config",
    "load_quality_sweep_config",
    "load_ablation_pipeline",
    "load_blocklist",
]
