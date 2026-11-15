#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention benchmark suite sweeping registered attention mechanisms.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from neuralops.benchmarks.core.base import AbstractBaseBenchmark, ExecutionSummary, _sanitize_float
from neuralops.config.benchmark_schema import AttentionSweepParameters

try:
    from neuralops.attentions import ATTENTION_REGISTRY
except ImportError:
    from neuralops.core.registry import ATTENTION_REGISTRY

from neuralops.core.registry import create_attention

logger = logging.getLogger(__name__)


class AttentionBenchmark(AbstractBaseBenchmark):
    """
    Empirical benchmark suite sweeping attention mechanisms across dimensions,
    batch sizes, sequence lengths, and precision types.
    """

    def setup(self) -> None:
        sweep_cfg = self.config.attention_sweep or AttentionSweepParameters()
        custom = self.config.custom_parameters or {}

        cfg_mechs = custom.get("mechanisms", sweep_cfg.mechanisms)
        if cfg_mechs == ["all"] or not cfg_mechs:
            all_keys = sorted(list(ATTENTION_REGISTRY.keys()))
            try:
                from neuralops.core.registry import DEPRECATED_ALIASES

                self.mechanisms = [k for k in all_keys if k not in DEPRECATED_ALIASES]
            except ImportError:
                self.mechanisms = all_keys
        else:
            self.mechanisms = list(cfg_mechs)

        limit = custom.get("limit")
        if limit is not None and isinstance(limit, int) and limit > 0:
            self.mechanisms = self.mechanisms[:limit]

        self.batch_sizes = custom.get("batch_sizes", sweep_cfg.batch_sizes)
        self.sequence_lengths = custom.get("sequence_lengths", sweep_cfg.sequence_lengths)
        self.head_dimensions = custom.get("head_dimensions", sweep_cfg.head_dimensions)
        self.num_heads = custom.get("num_heads", sweep_cfg.num_heads)
        self.causal_modes = custom.get("causal_modes", sweep_cfg.causal_modes)
        self.warmup_iterations = custom.get("warmup_iterations", sweep_cfg.warmup_iterations)
        self.benchmark_iterations = custom.get("benchmark_iterations", sweep_cfg.benchmark_iterations)

        if self.device.type == "cuda":
            self.dtypes = custom.get("dtypes", sweep_cfg.dtypes)
        else:
            self.dtypes = ["fp32"]

        self.start_time = time.time()
        self.results_table: List[Dict[str, Any]] = []

    def _resolve_torch_dtype(self, dtype_str: str) -> torch.dtype:
        mapping = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        val = mapping.get(dtype_str.lower())
        if val is None:
            raise ValueError(f"Unknown dtype '{dtype_str}'. Supported: {sorted(mapping.keys())}")
        return val

    def run_iteration(
        self,
        step: int,
        mechanism: str,
        batch_size: int = 1,
        seq_len: int = 128,
        head_dim: int = 64,
        num_heads: int = 2,
        is_causal: bool = False,
        dtype_str: str = "fp32",
        **kwargs: Any,
    ) -> Dict[str, float]:
        dtype = self._resolve_torch_dtype(dtype_str)
        dim = head_dim * num_heads

        self.flush_l2_cache()

        try:
            mod = create_attention(
                mechanism,
                dim=dim,
                num_heads=num_heads,
                is_decoder=is_causal,
            ).to(device=self.device)
            if dtype != torch.float32 and self.device.type == "cuda":
                mod = mod.to(dtype=dtype)
            mod.eval()
        except Exception as e:
            logger.warning("Instantiation failed for attention '%s': %s", mechanism, e)
            self.results_table.append(
                {
                    "step": step,
                    "mechanism_or_kernel": mechanism,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "dtype": dtype_str,
                    "latency_ms": 0.0,
                    "latency_p50_ms": 0.0,
                    "latency_p90_ms": 0.0,
                    "latency_p99_ms": 0.0,
                    "tflops": 0.0,
                    "peak_vram_mb": 0.0,
                    "status": f"FAILED_INIT: {e}",
                }
            )
            self.tracker.log_metric("latency_ms", 0.0, step=step)
            return {"latency_ms": 0.0, "tflops": 0.0, "peak_vram_mb": 0.0}

        x = torch.randn(batch_size, seq_len, dim, device=self.device, dtype=dtype)

        def forward_fn() -> None:
            with torch.no_grad():
                out = mod(x)
                if isinstance(out, tuple):
                    out = out[0]
                if not torch.isfinite(out).all():
                    raise ValueError(f"Non-finite output detected in {mechanism}")

        try:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            mean_ms, all_latencies = self.measure_latency(
                forward_fn,
                warmup=self.warmup_iterations,
                repeat=self.benchmark_iterations,
            )

            sorted_latencies = sorted(all_latencies)
            n = len(sorted_latencies)
            p50 = sorted_latencies[int(n * 0.50)] if n else mean_ms
            p90 = sorted_latencies[min(int(n * 0.90), n - 1)] if n else mean_ms
            p99 = sorted_latencies[min(int(n * 0.99), n - 1)] if n else mean_ms

            flops = 4.0 * batch_size * num_heads * (seq_len**2) * head_dim
            tflops = _sanitize_float((flops / 1e12) / max(mean_ms / 1000.0, 1e-9))

            peak_vram_mb = 0.0
            if self.device.type == "cuda" and torch.cuda.is_available():
                peak_vram_mb = _sanitize_float(torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0))

            self.tracker.log_metric("latency_ms", mean_ms, step=step)

            result_entry = {
                "step": step,
                "mechanism_or_kernel": mechanism,
                "batch_size": batch_size,
                "seq_len": seq_len,
                "dtype": dtype_str,
                "latency_ms": mean_ms,
                "latency_p50_ms": p50,
                "latency_p90_ms": p90,
                "latency_p99_ms": p99,
                "tflops": tflops,
                "peak_vram_mb": peak_vram_mb,
                "status": "PASS",
            }
            self.results_table.append(result_entry)
            return {"latency_ms": mean_ms, "tflops": tflops, "peak_vram_mb": peak_vram_mb}

        except (torch.cuda.OutOfMemoryError, Exception) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                logger.warning("OOM encountered for %s at seq_len=%d: %s", mechanism, seq_len, e)
                if self.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.results_table.append(
                    {
                        "step": step,
                        "mechanism_or_kernel": mechanism,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "dtype": dtype_str,
                        "latency_ms": 0.0,
                        "latency_p50_ms": 0.0,
                        "latency_p90_ms": 0.0,
                        "latency_p99_ms": 0.0,
                        "tflops": 0.0,
                        "peak_vram_mb": 0.0,
                        "status": "FAILED_OOM",
                    }
                )
                self.tracker.log_metric("latency_ms", 0.0, step=step)
                return {"latency_ms": 0.0, "tflops": 0.0, "peak_vram_mb": 0.0}
            else:
                logger.error("Runtime error in attention %s: %s", mechanism, e)
                self.results_table.append(
                    {
                        "step": step,
                        "mechanism_or_kernel": mechanism,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "dtype": dtype_str,
                        "latency_ms": 0.0,
                        "latency_p50_ms": 0.0,
                        "latency_p90_ms": 0.0,
                        "latency_p99_ms": 0.0,
                        "tflops": 0.0,
                        "peak_vram_mb": 0.0,
                        "status": f"FAILED_RUNTIME: {e}",
                    }
                )
                self.tracker.log_metric("latency_ms", 0.0, step=step)
                return {"latency_ms": 0.0, "tflops": 0.0, "peak_vram_mb": 0.0}

    def run(self) -> None:
        step = 0
        b = self.batch_sizes[0] if self.batch_sizes else 1
        s = self.sequence_lengths[0] if self.sequence_lengths else 128
        h_dim = self.head_dimensions[0] if self.head_dimensions else 64
        causal = self.causal_modes[0] if self.causal_modes else False
        dtype_str = self.dtypes[0] if self.dtypes else "fp32"

        for mech in self.mechanisms:
            self.run_iteration(
                step=step,
                mechanism=mech,
                batch_size=b,
                seq_len=s,
                head_dim=h_dim,
                num_heads=min(2, self.num_heads),
                is_causal=causal,
                dtype_str=dtype_str,
            )
            step += 1

    def generate_summary(self) -> ExecutionSummary:
        duration = _sanitize_float(time.time() - getattr(self, "start_time", time.time()))
        total = len(self.results_table)
        passed = sum(1 for r in self.results_table if r.get("status") == "PASS")
        failed_oom = sum(1 for r in self.results_table if r.get("status") == "FAILED_OOM")
        failed_runtime = total - passed - failed_oom

        passed_latencies = [
            r["latency_ms"] for r in self.results_table if r.get("status") == "PASS" and r.get("latency_ms", 0) > 0
        ]
        mean_lat = sum(passed_latencies) / len(passed_latencies) if passed_latencies else 0.0
        sorted_lat = sorted(passed_latencies)
        n = len(sorted_lat)
        p50 = sorted_lat[int(n * 0.50)] if n else 0.0
        p90 = sorted_lat[min(int(n * 0.90), n - 1)] if n else 0.0
        p99 = sorted_lat[min(int(n * 0.99), n - 1)] if n else 0.0

        peak_vram = max((r.get("peak_vram_mb", 0.0) for r in self.results_table), default=0.0)
        tflops_list = [r.get("tflops", 0.0) for r in self.results_table if r.get("status") == "PASS"]
        mean_tflops = sum(tflops_list) / len(tflops_list) if tflops_list else 0.0

        status = "COMPLETED" if (passed > 0 or total == 0) else "FAILED"

        return ExecutionSummary(
            run_id=self.context.run_id,
            benchmark_name="attention",
            status=status,
            total_iterations=total,
            passed_iterations=passed,
            failed_oom_iterations=failed_oom,
            failed_runtime_iterations=failed_runtime,
            mean_latency_ms=_sanitize_float(mean_lat),
            p50_latency_ms=_sanitize_float(p50),
            p90_latency_ms=_sanitize_float(p90),
            p99_latency_ms=_sanitize_float(p99),
            peak_vram_mb=_sanitize_float(peak_vram),
            mean_tflops=_sanitize_float(mean_tflops),
            duration_seconds=duration,
            results_table=self.results_table,
            extra_metadata={"mechanisms_count": len(self.mechanisms)},
        )

    def teardown(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class AttentionBenchmarkSuite:
    """End-to-end benchmark suite for attention mechanisms emitting CanonicalRecord instances."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def run(self) -> list[Any]:
        import datetime
        import hashlib
        import subprocess
        from neuralops.benchmarks.results_v2.record import BenchmarkStatus, CanonicalRecord

        records: list[CanonicalRecord] = []
        run_id = getattr(self.config, "run_id", "default_run")
        device_str = getattr(self.config, "device", "cpu")
        device = torch.device(device_str)

        git_commit = "unknown"
        git_dirty = False
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                git_commit = res.stdout.strip()[:10]
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            )
            git_dirty = len(status_res.stdout.strip()) > 0
        except Exception:
            pass

        suites = getattr(self.config, "suites", ())
        if not suites:
            suites = [self.config]

        for s in suites:
            suite_name = getattr(s, "name", "attention_sweep")
            attentions = getattr(s, "attentions", ("cosformer",)) or ("cosformer",)
            batch_sizes = getattr(s, "batch_sizes", (1,)) or (1,)
            seq_lens = getattr(s, "seq_lens", (64,)) or (64,)
            warmup_iters = getattr(s, "warmup_iters", 2)
            num_iters = getattr(s, "num_iters", 3)

            for mech in attentions:
                for b in batch_sizes:
                    for seq in seq_lens:
                        dim = 64
                        num_heads = 2
                        try:
                            if mech == "cosformer":
                                from neuralops.layers.linear.cosformer import CosformerAttention

                                mod = CosformerAttention(dim=dim, num_heads=num_heads).to(device=device)
                            else:
                                from neuralops.core.registry import create_attention

                                mod = create_attention(mech, dim=dim, num_heads=num_heads).to(device=device)
                            mod.eval()

                            x = torch.randn(b, seq, dim, device=device)

                            # Warmup
                            with torch.no_grad():
                                for _ in range(warmup_iters):
                                    _ = mod(x)

                            # Timing measurement
                            if device.type == "cuda" and torch.cuda.is_available():
                                start_event = torch.cuda.Event(enable_timing=True)
                                end_event = torch.cuda.Event(enable_timing=True)
                                torch.cuda.synchronize()
                                start_event.record()
                                with torch.no_grad():
                                    for _ in range(num_iters):
                                        _ = mod(x)
                                end_event.record()
                                torch.cuda.synchronize()
                                total_ms = start_event.elapsed_time(end_event)
                                latency_ms = total_ms / max(num_iters, 1)
                            else:
                                t_start = time.perf_counter()
                                with torch.no_grad():
                                    for _ in range(num_iters):
                                        _ = mod(x)
                                total_s = time.perf_counter() - t_start
                                latency_ms = (total_s / max(num_iters, 1)) * 1000.0

                            status = BenchmarkStatus.SUCCESS.value
                            metrics = {
                                "latency_ms": max(latency_ms, 0.001),
                                "throughput": float(b * seq) / max(latency_ms / 1000.0, 1e-6),
                            }
                        except Exception as e:
                            logger.error("Attention benchmark failed for %s: %s", mech, e)
                            status = BenchmarkStatus.FAILED.value
                            metrics = {"latency_ms": 0.0}

                        checksum_data = f"{run_id}:{git_commit}:{suite_name}:{mech}:{b}:{seq}"
                        checksum = hashlib.sha256(checksum_data.encode()).hexdigest()

                        record = CanonicalRecord(
                            run_id=run_id,
                            timestamp=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            git_commit=git_commit,
                            git_dirty=git_dirty,
                            suite=suite_name,
                            module=mech,
                            model="synthetic",
                            dataset="synthetic",
                            device=device_str,
                            status=status,
                            metrics=metrics,
                            config={"batch_size": b, "seq_len": seq, "dim": dim, "num_heads": num_heads},
                            telemetry={},
                            checksum=checksum,
                        )
                        records.append(record)
        return records

