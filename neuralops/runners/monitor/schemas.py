#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Monitor Schemas and DTOs

Pydantic schemas and data transfer objects for benchmark telemetry and dashboard APIs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GpuMetrics(BaseModel):
    name: str = Field(default="NVIDIA GeForce RTX 4070 Laptop GPU", description="GPU Model Name")
    gpu_util: float = Field(default=0.0, ge=0.0, le=100.0, description="GPU Core Utilization %")
    mem_used_mb: float = Field(default=0.0, ge=0.0, description="Memory used in Megabytes")
    mem_total_mb: float = Field(default=8192.0, gt=0.0, description="Total memory in Megabytes")
    mem_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Memory used percentage")
    power_w: float = Field(default=0.0, ge=0.0, description="Power draw in Watts")
    temp_c: float = Field(default=0.0, description="GPU temperature in Celsius")
    clock_mhz: Optional[float] = Field(default=0.0, description="Graphics clock in MHz")
    status: str = Field(default="online", description="GPU device status")


class AttentionModuleItem(BaseModel):
    name: str = Field(..., description="Canonical module identifier")
    family: Optional[str] = Field(default="subquadratic", description="Algorithmic family (e.g. linear, state_space, chunked, toeplitz)")
    complexity_time: Optional[str] = Field(default="O(N)", description="Asymptotic time complexity")
    complexity_space: Optional[str] = Field(default="O(N)", description="Asymptotic memory complexity")
    latency_ms: Optional[float] = Field(default=None, description="Triton kernel latency in ms")
    accuracy_pct: Optional[float] = Field(default=None, description="GLUE / downstream accuracy percentage")
    perplexity: Optional[float] = Field(default=None, description="Wikitext-2 / language modeling perplexity")
    status: str = Field(default="OK", description="Execution or verification status")


class CompressorBenchmarkItem(BaseModel):
    name: str = Field(..., description="Compression engine identifier")
    compress_ms: Optional[float] = Field(default=None, description="Time to compress in milliseconds")
    rel_err: Optional[float] = Field(default=None, description="Relative Frobenius / matrix reconstruction error")
    compression_ratio: Optional[float] = Field(default=None, description="Memory compression ratio (e.g. 4.0x)")
    status: str = Field(default="OK", description="Benchmark status")


class PagedKVCacheBlock(BaseModel):
    block_id: int
    seq_id: Optional[int] = None
    is_free: bool = True
    token_count: int = 0
    capacity: int = 16


class PagedKVCacheAllocation(BaseModel):
    num_blocks: int = 64
    block_size: int = 16
    allocated_blocks: int = 0
    free_blocks: int = 64
    active_sequences: int = 0
    utilization_pct: float = 0.0
    fragmentation_pct: float = 0.0
    continuous_batching: Dict[str, Any] = Field(default_factory=dict)
    blocks: List[PagedKVCacheBlock] = Field(default_factory=list)


class ContinuousBatchingState(BaseModel):
    active_requests: int = 0
    queued_requests: int = 0
    iteration_step: int = 0
    avg_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    prefill_inflight: int = 0
    decode_inflight: int = 0


class GammaAcceptanceRate(BaseModel):
    gamma: int
    acceptance_rate: float
    effective_speedup: float


class SpeculativeDecodingMetrics(BaseModel):
    enabled: bool = True
    draft_model: str = "SVD-Linear-Student (alpha=1.0)"
    target_model: str = "Dense RoBERTa / GPT-2 (alpha=0.0)"
    draft_k: int = 5
    acceptance_rate: float = Field(default=0.785, ge=0.0, le=1.0, description="Token acceptance rate")
    mean_accepted_tokens: float = Field(default=3.92, description="Mean accepted tokens per draft-verify cycle")
    speedup_factor: float = Field(default=2.45, description="Effective wall-clock speedup vs autoregressive target")
    total_tokens_generated: int = Field(default=14520, description="Cumulative tokens produced")
    zero_divergence_verified: bool = Field(default=True, description="Mathematical equivalence guarantee")
    gamma_curves: List[GammaAcceptanceRate] = Field(
        default_factory=lambda: [
            GammaAcceptanceRate(gamma=1, acceptance_rate=0.912, effective_speedup=1.78),
            GammaAcceptanceRate(gamma=2, acceptance_rate=0.865, effective_speedup=2.15),
            GammaAcceptanceRate(gamma=4, acceptance_rate=0.798, effective_speedup=2.52),
            GammaAcceptanceRate(gamma=8, acceptance_rate=0.710, effective_speedup=2.74),
        ]
    )
    history_acceptance_rates: List[float] = Field(
        default_factory=lambda: [
            0.72, 0.74, 0.76, 0.81, 0.79, 0.78, 0.83, 0.82, 0.85, 0.78,
            0.79, 0.80, 0.81, 0.84, 0.78, 0.77, 0.82, 0.86, 0.79, 0.83,
        ]
    )


class NotebookInfo(BaseModel):
    id: str
    title: str
    description: str
    filename: str
    path: str
    status: str = "verified"
    tags: List[str] = Field(default_factory=list)


class ExperimentLaunchRequest(BaseModel):
    module: Optional[str] = Field(default="hgrn2_gated_deltanet", description="Module to sweep or run")
    rank: Optional[int] = Field(default=16, ge=1, le=1024, description="SVD decomposition rank")
    batch_size: Optional[int] = Field(default=8, ge=1, le=128, description="Batch size")
    seq_len: Optional[int] = Field(default=512, ge=16, le=32768, description="Sequence length")
    epochs: Optional[int] = Field(default=3, ge=1, le=100, description="Training epochs")
    learning_rate: Optional[float] = Field(default=3e-4, gt=0.0, description="Learning rate")


class ApiResponse(BaseModel):
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Any] = None
