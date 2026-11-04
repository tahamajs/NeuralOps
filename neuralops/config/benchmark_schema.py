#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Authoritative Pydantic V2 benchmark configuration schemas for NeuralOps.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HardwareExecutionTarget(BaseModel):
    """Execution hardware configuration and precision rules."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    device: str = Field(
        default="cuda:0",
        description="Target compute device string ('cuda:0', 'cpu', 'mps')",
    )
    backend_preference: Literal["auto", "cuda", "triton", "torch"] = Field(
        default="auto",
        description="Backend priority in the 3-level fallback hierarchy",
    )
    enforce_deterministic: bool = Field(
        default=False,
        description="Enforce deterministic CUDA algorithms",
    )
    allow_tf32: bool = Field(
        default=True,
        description="Allow TensorFloat-32 for GEMM and convolutions",
    )
    target_compute_capability: Optional[str] = Field(
        default=None,
        description="Optional target compute capability, e.g. '8.9'",
    )
    memory_fraction_limit: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Maximum fraction of physical GPU VRAM to allocate",
    )

    @field_validator("device")
    @classmethod
    def validate_device_string(cls, v: str) -> str:
        if re.match(r"^\d+$", v):
            v = f"cuda:{v}"
        pattern = r"^(cuda(:\d+)?|cpu|mps)$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid compute device '{v}'. Must match pattern: {pattern}")
        return v


class TrackingSubsystemConfig(BaseModel):
    """Telemetry recording and lifecycle tracking configuration."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    enabled: bool = Field(
        default=True,
        description="Master switch for experiment tracking",
    )
    enable_tensorboard: bool = Field(
        default=True,
        description="Generate SummaryWriter events in run directory",
    )
    enable_local_jsonl: bool = Field(
        default=True,
        description="Stream step iterations to metrics.jsonl",
    )
    log_frequency_steps: int = Field(
        default=1,
        ge=1,
        description="Telemetry logging interval in steps",
    )
    capture_gpu_telemetry: bool = Field(
        default=True,
        description="Sample GPU temperature, power, clocks, and VRAM via NVML",
    )
    telemetry_sample_interval_ms: int = Field(
        default=250,
        ge=50,
        le=10000,
        description="Background hardware polling frequency in milliseconds",
    )
    tensorboard_flush_secs: int = Field(
        default=10,
        ge=1,
        description="Flush interval for TensorBoard events",
    )
    export_formats: List[str] = Field(
        default_factory=lambda: ["jsonl", "csv", "summary"],
        description="Formats to export metrics and summaries",
    )


class AttentionSweepParameters(BaseModel):
    """Multi-dimensional grid sweep parameters for attention mechanism benchmarking."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    mechanisms: List[str] = Field(
        default_factory=lambda: ["all"],
        description="List of attention keys or ['all'] to sweep all 131 registered attentions",
    )
    batch_sizes: List[int] = Field(
        default_factory=lambda: [1, 4, 16],
        description="Batch sizes for latency/throughput matrix",
    )
    sequence_lengths: List[int] = Field(
        default_factory=lambda: [512, 1024, 2048, 4096, 8192],
        description="Sequence length token counts",
    )
    head_dimensions: List[int] = Field(
        default_factory=lambda: [64, 128],
        description="Head dimensions (d_k and d_v)",
    )
    num_heads: int = Field(
        default=32,
        ge=1,
        description="Number of query attention heads",
    )
    num_kv_heads: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of key/value heads for GQA/MQA models",
    )
    dtypes: List[Literal["fp16", "bf16", "fp32"]] = Field(
        default_factory=lambda: ["fp16", "bf16"],
        description="Floating-point precisions to evaluate",
    )
    causal_modes: List[bool] = Field(
        default_factory=lambda: [True, False],
        description="Evaluate causal autoregressive masking and bidirectional attention",
    )
    warmup_iterations: int = Field(
        default=10,
        ge=1,
        description="Number of un-timed warmup passes to stabilize GPU clocks",
    )
    benchmark_iterations: int = Field(
        default=50,
        ge=5,
        description="Number of timed measurement iterations per configuration",
    )

    @field_validator("batch_sizes", "sequence_lengths", "head_dimensions")
    @classmethod
    def validate_positive_integers(cls, v: List[int]) -> List[int]:
        if not v or any(x <= 0 for x in v):
            raise ValueError("All dimensional quantities must be strictly positive integers.")
        return sorted(list(set(v)))


class CompressionSweepParameters(BaseModel):
    """Grid sweep parameters for model compression algorithm benchmarking."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    compressors: List[str] = Field(
        default_factory=lambda: ["all"],
        description="List of compressor keys or ['all'] to sweep all 72 registered algorithms",
    )
    target_layers: List[str] = Field(
        default_factory=lambda: ["all"],
        description="Target model layer patterns (e.g., ['q_proj', 'v_proj', 'all'])",
    )
    target_ratios: List[float] = Field(
        default_factory=lambda: [0.25, 0.50, 0.75],
        description="Compression retention ratios",
    )
    evaluation_datasets: List[str] = Field(
        default_factory=lambda: ["wikitext2"],
        description="Evaluation datasets for post-compression perplexity",
    )
    eval_context_length: int = Field(
        default=2048,
        ge=128,
        description="Context window for perplexity evaluation",
    )

    @field_validator("target_ratios")
    @classmethod
    def validate_ratios(cls, v: List[float]) -> List[float]:
        for r in v:
            if not (0.0 < r < 1.0):
                raise ValueError(f"Compression ratio {r} must be strictly between 0.0 and 1.0")
        return sorted(list(set(v)))


class BenchmarkRunConfig(BaseModel):
    """Canonical root configuration model governing a benchmark execution."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    version: str = Field(
        default="1.0.0",
        description="Schema specification version",
    )
    benchmark_suite: Literal[
        "attention",
        "compression",
        "cuda_ops",
        "triton_ops",
        "e2e_serving",
        "niah",
        "roofline",
        "qwen_kd",
    ] = Field(
        ...,
        description="Identifier of the benchmark suite to run",
    )
    name: Optional[str] = Field(
        default=None,
        description="Human-readable run name or semantic label",
    )
    description: Optional[str] = Field(
        default=None,
        description="Purpose and hypothesis of the benchmark execution",
    )
    tag: Optional[str] = Field(
        default=None,
        description="Semantic categorization tag for SQLite filtering",
    )
    output_root_directory: Path = Field(
        default=Path("results/benchmarks"),
        description="Filesystem root where per-run isolated directories are created",
    )
    hardware: HardwareExecutionTarget = Field(
        default_factory=HardwareExecutionTarget,
        description="Hardware placement and compute rules",
    )
    tracking: TrackingSubsystemConfig = Field(
        default_factory=TrackingSubsystemConfig,
        description="Telemetry and tracking parameters",
    )
    attention_sweep: Optional[AttentionSweepParameters] = Field(
        default=None,
        description="Parameters for attention sweep benchmark",
    )
    compression_sweep: Optional[CompressionSweepParameters] = Field(
        default=None,
        description="Parameters for compressor sweep benchmark",
    )
    custom_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional suite-specific parameters",
    )

    def model_dump_yaml(self) -> str:
        """Serialize configuration model to YAML string."""
        import yaml

        return yaml.dump(self.model_dump(mode="json"), sort_keys=False)
