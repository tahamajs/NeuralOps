#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Phase 1: Strongly Typed Benchmark Configuration Specification.

Provides validation, serialization, and default configuration contracts
for all NeuralOps microbenchmark sweeps, training loops, and serving matrices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Graceful fallback if pydantic is not installed
    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> dict[str, Any]:
            return self.__dict__.copy()

        def model_dump_json(self, indent: int = 2) -> str:
            return json.dumps(self.model_dump(), indent=indent)

    def Field(default: Any = None, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return default


class BenchmarkConfig(BaseModel):
    """Authoritative benchmark run configuration contract."""

    device: Literal["cuda", "mps", "cpu", "auto"] = Field(default="auto", description="Target accelerator backend.")
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32", description="Tensor computation precision."
    )
    batch_sizes: list[int] = Field(default=[2, 4, 8], description="Batch sizes to evaluate.")
    seq_lens: list[int] = Field(default=[16, 64, 256], description="Sequence lengths to evaluate.")
    warmup: int = Field(default=3, description="Number of untimed warmup iterations.")
    iters: int = Field(default=10, description="Number of timed benchmark trials.")
    seed: int = Field(default=42, description="Random seed for reproducible tensors.")
    categories: list[str] = Field(
        default=["attention", "compressor", "kernel"],
        description="Benchmark categories to run.",
    )
    output_dir: Path = Field(default=Path("results"), description="Root directory for telemetry artifacts.")
    parallel_workers: int = Field(default=4, description="Concurrency level for CPU/MPS evaluation sweeps.")
    enable_backward: bool = Field(default=True, description="Whether to execute backward autograd gradient checks.")
    enable_vram_tracking: bool = Field(default=True, description="Track peak accelerator memory allocations.")

    def to_canonical(self):
        """Convert the legacy script config to the package config contract."""
        from neuralops.benchmarks.config import BenchmarkConfig as CanonicalConfig

        data = self.model_dump() if hasattr(self, "model_dump") else self.__dict__.copy()
        data["precision"] = data.pop("dtype", "float32")
        data["output_dir"] = Path(data.get("output_dir", "results"))
        allowed = set(CanonicalConfig.__dataclass_fields__)
        extra = {
            "batch_sizes": data.get("batch_sizes", [2, 4, 8]),
            "seq_lens": data.get("seq_lens", [16, 64, 256]),
            "parallel_workers": data.get("parallel_workers", 4),
            "enable_backward": data.get("enable_backward", True),
            "enable_vram_tracking": data.get("enable_vram_tracking", True),
        }
        filtered = {}
        for key, value in data.items():
            if key in allowed:
                filtered[key] = value
            else:
                extra[key] = value
        filtered["extra"] = extra
        return CanonicalConfig(**filtered)

    @classmethod
    def from_file(cls, path: str | Path) -> BenchmarkConfig:
        """Load configuration from JSON or YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]

                data = yaml.safe_load(text)
            except ImportError:
                # Basic JSON fallback if YAML parser unavailable
                data = json.loads(text)
        else:
            data = json.loads(text)
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """Serialize configuration to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        dump_fn = getattr(self, "model_dump_json", None)
        if callable(dump_fn):
            content = dump_fn(indent=2)
        else:
            content = json.dumps(self.__dict__, indent=2)
        p.write_text(content, encoding="utf-8")
