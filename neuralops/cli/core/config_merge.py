#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Deterministic Configuration Precedence and Merge Engine for NeuralOps CLI.

Precedence Contract:
  Built-in Defaults < Config File < Environment Variables < Explicit CLI Flags
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from neuralops.benchmarks.config import BenchmarkConfig

from .errors import ConfigurationError


def parse_csv_list(value: Any, cast_type: type = str) -> list[Any]:
    """Parse comma-separated values into a typed list."""
    if isinstance(value, list):
        return [cast_type(x) for x in value]
    if isinstance(value, str):
        tokens = [t.strip() for t in value.split(",") if t.strip()]
        if not tokens:
            raise ConfigurationError("Comma-separated list must contain at least one element.")
        try:
            return [cast_type(t) for t in tokens]
        except ValueError as exc:
            raise ConfigurationError(f"Cannot cast token to {cast_type.__name__} in '{value}': {exc}") from exc
    return [cast_type(value)]


def extract_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Extracts non-None CLI arguments to override configuration settings."""
    raw = vars(args)
    overrides: dict[str, Any] = {}
    aliases = {
        "dtype": "precision",
        "batch_size": "batch_sizes",
        "seq_len": "seq_lens",
        "iters": "iters",
        "warmup": "warmup",
        "output": "output_dir",
    }

    for key, value in raw.items():
        if value is None or key in {"config", "subcommand", "group", "command"}:
            continue
        canon_key = aliases.get(key, key)
        overrides[canon_key] = value

    if "batch_sizes" in overrides and isinstance(overrides["batch_sizes"], str):
        overrides["batch_sizes"] = parse_csv_list(overrides["batch_sizes"], int)
    if "seq_lens" in overrides and isinstance(overrides["seq_lens"], str):
        overrides["seq_lens"] = parse_csv_list(overrides["seq_lens"], int)
    if "categories" in overrides and isinstance(overrides["categories"], str):
        overrides["categories"] = parse_csv_list(overrides["categories"], str)

    return overrides


def load_effective_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Merges built-in defaults, config file, environment variables, and explicit CLI flags.

    Raises:
        ConfigurationError: If unknown config keys or malformed parameters are provided.
    """
    config_path = getattr(args, "config", None)
    if config_path:
        p = Path(config_path)
        if not p.is_file():
            raise ConfigurationError(
                f"Configuration file not found: '{config_path}'",
                hint="Verify file path or omit --config to use canonical presets.",
            )
        try:
            if p.suffix.lower() in (".yaml", ".yml"):
                try:
                    import yaml

                    with open(p, "r", encoding="utf-8") as f:
                        file_data = yaml.safe_load(f) or {}
                except ImportError as exc:
                    raise ConfigurationError(
                        "PyYAML is required to parse YAML config files.",
                        hint="Install pyyaml via 'pip install pyyaml' or use JSON configs.",
                    ) from exc
            else:
                with open(p, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
        except Exception as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed parsing configuration file '{config_path}': {exc}") from exc

        valid_fields = set(BenchmarkConfig.__dataclass_fields__.keys()) | {"dtype"}
        unknown_keys = sorted(set(file_data.keys()) - valid_fields)
        if unknown_keys:
            raise ConfigurationError(
                f"Unknown configuration key(s) in '{config_path}': {', '.join(unknown_keys)}",
                hint=f"Valid keys include: {', '.join(sorted(valid_fields))}",
            )
        base_cfg = BenchmarkConfig.from_dict(file_data)
    else:
        base_cfg = BenchmarkConfig()

    values = base_cfg.to_dict()

    # Environment variable overrides
    if "NEURALOPS_DEVICE" in os.environ:
        values["device"] = os.environ["NEURALOPS_DEVICE"]
    if "NEURALOPS_PRECISION" in os.environ:
        values["precision"] = os.environ["NEURALOPS_PRECISION"]
    if "NEURALOPS_OUTPUT_DIR" in os.environ:
        values["output_dir"] = Path(os.environ["NEURALOPS_OUTPUT_DIR"])

    # Explicit CLI argument overrides
    cli_opts = extract_cli_overrides(args)
    values.update(cli_opts)

    return BenchmarkConfig.from_dict(values)
