"""
Pydantic v2 schemas for benchmark configurations.
Follows AGENTS.md Section 3 rules (frozen, extra='forbid', tuple defaults).
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class SuiteConfig(BaseModel):
    """Configuration for an individual benchmark suite."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=False,
        validate_assignment=True,
    )

    name: str = "suite"
    type: str = "attention"
    attentions: tuple[str, ...] = Field(default_factory=tuple)
    kernels: tuple[str, ...] = Field(default_factory=tuple)
    batch_sizes: tuple[int, ...] = Field(default_factory=tuple)
    seq_lens: tuple[int, ...] = Field(default_factory=tuple)
    head_dims: tuple[int, ...] = Field(default_factory=tuple)
    num_heads: tuple[int, ...] = Field(default_factory=tuple)
    dtypes: tuple[str, ...] = Field(default_factory=tuple)
    warmup_iters: int = Field(default=10, ge=0)
    num_iters: int = Field(default=50, ge=1)


class RunConfig(BaseModel):
    """Runtime configuration options."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=False,
        validate_assignment=True,
    )

    run_id: str
    output_dir: str = "results"
    device: str = "cpu"
    seed: int = 42
    record_telemetry: bool = True


class BenchmarkConfig(BaseModel):
    """Top-level configuration for a benchmark run."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=False,
        validate_assignment=True,
    )

    run_id: str
    output_dir: str = "results"
    device: str = "cpu"
    suites: tuple[SuiteConfig, ...] = Field(default_factory=tuple)
    suite: str | None = None
    kernels: tuple[str, ...] = Field(default_factory=tuple)
    seed: int = 42
    tags: tuple[str, ...] = Field(default_factory=tuple)
