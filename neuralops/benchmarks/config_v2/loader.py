"""
Benchmark YAML configuration loader.
The single sanctioned entrypoint for loading benchmark YAMLs (Rule 3.3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml
from pydantic import ValidationError

from neuralops.benchmarks.config_v2.schema import BenchmarkConfig
from neuralops.core.exceptions import ConfigError


class ConfigValidationError(ConfigError):
    """Raised when benchmark configuration loading or schema validation fails."""


__all__ = [
    "ConfigValidationError",
    "load_benchmark_config",
    "parse_yaml_string",
]


def parse_yaml_string(content: str) -> BenchmarkConfig:
    """Parse a YAML string into a strictly-validated BenchmarkConfig."""
    try:
        raw: Any = yaml.safe_load(content)
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                f"Expected YAML mapping at root, got {type(raw).__name__}"
            )
        return BenchmarkConfig(**raw)
    except ValidationError as e:
        raise ConfigValidationError(f"Configuration validation failed:\n{e}") from e
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Invalid YAML syntax: {e}") from e


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and parse a benchmark YAML config file from disk."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark configuration file not found: {file_path}")

    content = file_path.read_text(encoding="utf-8")
    return parse_yaml_string(content)
