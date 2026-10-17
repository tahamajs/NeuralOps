"""Pydantic schemas and data transfer objects for NeuralOps Web Dashboard & Monitor API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GpuMetrics(BaseModel):
    name: str = Field(default="NVIDIA GeForce RTX 4070 (8GB)", description="GPU Model Name")
    gpu_util: float = Field(default=0.0, ge=0.0, le=100.0, description="GPU Core Utilization %")
    mem_used_mb: float = Field(default=0.0, ge=0.0, description="Memory used in Megabytes")
    mem_total_mb: float = Field(default=8192.0, gt=0.0, description="Total memory in Megabytes")
    mem_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Memory used percentage")
    power_w: float = Field(default=0.0, ge=0.0, description="Power draw in Watts")
    temp_c: float = Field(default=0.0, description="GPU temperature in Celsius")


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
    blocks: List[PagedKVCacheBlock] = Field(default_factory=list)


class ContinuousBatchingState(BaseModel):
    active_requests: int = 0
    queued_requests: int = 0
    iteration_step: int = 0
    avg_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    prefill_inflight: int = 0
    decode_inflight: int = 0


class SpeculativeDecodingMetrics(BaseModel):
    enabled: bool = True
    draft_model: str = "SVD-Linear-Student (alpha=1.0)"
    target_model: str = "Dense RoBERTa / GPT-2 (alpha=0.0)"
    draft_k: int = 5
    acceptance_rate: float = Field(default=0.785, ge=0.0, le=1.0, description="Token acceptance rate")
    mean_accepted_tokens: float = Field(default=3.92, description="Mean accepted tokens per draft-verify cycle")
    speedup_factor: float = Field(default=2.45, description="Effective wall-clock speedup vs autoregressive target")
    total_tokens_generated: int = Field(default=12400, description="Cumulative tokens produced")
    history_acceptance_rates: List[float] = Field(default_factory=lambda: [0.72, 0.75, 0.79, 0.82, 0.78, 0.80, 0.77, 0.81, 0.79, 0.83])


class RooflineProfile(BaseModel):
    module_name: str
    achieved_tflops: float
    arithmetic_intensity: float
    peak_tflops: float
    peak_bandwidth_gb_s: float
    ridge_point: float
    bottleneck: str


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
