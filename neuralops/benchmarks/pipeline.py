#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Universal Benchmark Pipeline and Sweep Controller.

Implements BenchmarkPipeline to orchestrate evaluation across registered benchmarks
and SweepController to iterate sequentially across attention mechanisms (from 131
ATTENTION_REGISTRY keys) and compressors (from 72 COMPRESSOR_REGISTRY keys).
Provides fail-closed dynamic memory sanitation (gc.collect() and torch.cuda.empty_cache()).
"""

from __future__ import annotations

import copy
import gc
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

from neuralops.benchmarks.adapters.model import (
    BaseModelAdapter,
    UniversalModelAdapter,
)
from neuralops.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BaseBenchmark,
    get_benchmark,
)
from neuralops.benchmarks.results.saver import MultiFormatSaver
from neuralops.benchmarks.results.schema import BenchmarkResult
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


def sanitize_memory() -> None:
    """Perform fail-closed garbage collection and accelerator memory deallocation."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            logger.debug("CUDA memory sanitization encountered non-fatal error: %s", e)


class BenchmarkPipeline:
    """Orchestrator feeding evaluation tasks to model adapters and saving telemetry."""

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        task_keys: Optional[List[str]] = None,
        saver: Optional[MultiFormatSaver] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model_adapter = model_adapter
        self.task_keys = task_keys or []
        self.saver = saver
        if device is not None:
            self.model_adapter.to(device)

    def execute_task(self, task_key: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute a single registered benchmark task against the model adapter."""
        target = get_benchmark(task_key)
        if target is None:
            # Check if task_key is directly registered in BENCHMARK_REGISTRY
            target = BENCHMARK_REGISTRY.get(task_key)

        if target is None:
            raise KeyError(
                f"Benchmark task '{task_key}' is not registered in BENCHMARK_REGISTRY. "
                f"Available tasks: {sorted(list(BENCHMARK_REGISTRY.keys()))}"
            )

        logger.info("Executing benchmark task '%s' on %s", task_key, self.model_adapter.model_name)
        t0 = time.perf_counter()

        # Handle class vs instance vs function
        if isinstance(target, type) and issubclass(target, BaseBenchmark):
            runner = target(**kwargs)
            metrics = runner.run(self.model_adapter, **kwargs)
        elif isinstance(target, BaseBenchmark):
            metrics = target.run(self.model_adapter, **kwargs)
        elif callable(target):
            # Callable may expect model_adapter or kwargs
            try:
                metrics = target(model_adapter=self.model_adapter, **kwargs)
            except TypeError:
                try:
                    metrics = target(model=self.model_adapter.model, **kwargs)
                except TypeError:
                    metrics = target(**kwargs)
        elif isinstance(target, dict) and "target" in target:
            subtarget = target["target"]
            if callable(subtarget):
                try:
                    metrics = subtarget(model_adapter=self.model_adapter, **kwargs)
                except TypeError:
                    metrics = subtarget(**kwargs)
            else:
                raise TypeError(f"Registered benchmark target for '{task_key}' is not callable.")
        else:
            raise TypeError(f"Unknown benchmark runner type for '{task_key}': {type(target)}")

        duration = time.perf_counter() - t0
        if not isinstance(metrics, dict):
            metrics = {"result": metrics}
        metrics.setdefault("wall_clock_s", duration)

        # Multi-sink persistence if saver provided
        if self.saver is not None:
            try:
                res = BenchmarkResult(
                    metrics=metrics,
                    model=self.model_adapter.get_model_metadata(),
                )
                res.run_metadata.run_name = f"{self.model_adapter.model_name}_{task_key}"
                self.saver.save(res)
            except Exception as e:
                logger.warning("Auto-persistence of benchmark result failed: %s", e)

        return metrics

    def run(self, **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        """Run all configured tasks in sequence and aggregate results."""
        results: Dict[str, Dict[str, Any]] = {}
        for task_key in self.task_keys:
            try:
                results[task_key] = self.execute_task(task_key, **kwargs)
            except Exception as e:
                logger.error("Task '%s' failed: %s", task_key, e)
                results[task_key] = {"error": str(e), "status": "failed"}
            finally:
                sanitize_memory()
        return results


class SweepController:
    """Controller to iterate systematically across attention mechanisms or compressors."""

    def __init__(
        self,
        base_model_or_adapter: Optional[Union[str, nn.Module, BaseModelAdapter]] = None,
        output_dir: Optional[str] = None,
        saver: Optional[MultiFormatSaver] = None,
        device: Optional[Union[str, torch.device]] = None,
        base_model: Optional[Union[str, nn.Module, BaseModelAdapter]] = None,
    ) -> None:
        target = base_model_or_adapter if base_model_or_adapter is not None else base_model
        if target is None:
            raise ValueError("Either base_model_or_adapter or base_model must be provided.")

        if isinstance(target, BaseModelAdapter):
            self.base_adapter = target
        else:
            self.base_adapter = UniversalModelAdapter.from_pretrained(
                target, device=device
            )
        self.output_dir = output_dir or "results/sweep/"
        os.makedirs(self.output_dir, exist_ok=True)
        self.saver = saver or MultiFormatSaver(output_dir=self.output_dir)
        self.device = device or self.base_adapter.device

    def sweep_attentions(
        self,
        attention_keys: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
        auto_calibrate: bool = True,
        damping: float = 1e-4,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Screen and evaluate a list of attention keys (defaulting to all 131 registry keys)."""
        from neuralops.api import transplant_attention

        keys_to_sweep = attention_keys or sorted(list(ATTENTION_REGISTRY.keys()))
        tasks = benchmarks or []
        sweep_results: Dict[str, Dict[str, Any]] = {}

        logger.info("Beginning attention sweep across %d keys...", len(keys_to_sweep))

        for key in keys_to_sweep:
            if key not in ATTENTION_REGISTRY:
                logger.warning("Skipping unknown attention key: %s", key)
                continue

            logger.info("--- [Attention Sweep] Evaluating candidate: %s ---", key)
            try:
                # Deepcopy model to ensure complete memory isolation
                candidate_model = copy.deepcopy(self.base_adapter.model)
                transplanted = transplant_attention(
                    model=candidate_model,
                    attention=key,
                )

                surgery_manifest = {
                    "is_transplanted": True,
                    "transplanted_attention": key,
                    "attention_key": key,
                    "calibrated": auto_calibrate,
                    "calibration_method": "ridge_closed_form" if auto_calibrate else "none",
                    "damping": damping,
                    "base_model": self.base_adapter.model_name,
                }

                candidate_adapter = UniversalModelAdapter.from_pretrained(
                    transplanted,
                    model_name=f"{self.base_adapter.model_name}_{key}",
                    device=self.device,
                )
                if hasattr(candidate_adapter, "surgery_manifest"):
                    candidate_adapter.surgery_manifest = surgery_manifest

                pipeline = BenchmarkPipeline(
                    model_adapter=candidate_adapter,
                    task_keys=tasks,
                    saver=self.saver,
                    device=self.device,
                )
                sweep_results[key] = pipeline.run(**kwargs)

            except Exception as e:
                logger.error("Failed evaluation of attention candidate '%s': %s", key, e)
                sweep_results[key] = {"error": str(e), "status": "failed"}
            finally:
                sanitize_memory()

        return sweep_results

    def sweep_compressors(
        self,
        compressor_keys: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
        rank: int = 32,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Any]]:
        """Screen and evaluate a list of compressor keys (defaulting to all 72 registry keys)."""
        from neuralops.api import compress

        keys_to_sweep = compressor_keys or sorted(list(COMPRESSOR_REGISTRY.keys()))
        tasks = benchmarks or []
        sweep_results: Dict[str, Dict[str, Any]] = {}

        logger.info("Beginning compression sweep across %d keys...", len(keys_to_sweep))

        for key in keys_to_sweep:
            if key not in COMPRESSOR_REGISTRY:
                logger.warning("Skipping unknown compressor key: %s", key)
                continue

            logger.info("--- [Compressor Sweep] Evaluating candidate: %s ---", key)
            try:
                candidate_model = copy.deepcopy(self.base_adapter.model)
                compressed = compress(
                    model=candidate_model,
                    method=key,
                    rank=rank,
                )

                surgery_manifest = {
                    "is_compressed": True,
                    "compressor_key": key,
                    "compression_method": key,
                    "rank": rank,
                    "base_model": self.base_adapter.model_name,
                }

                candidate_adapter = UniversalModelAdapter.from_pretrained(
                    compressed,
                    model_name=f"{self.base_adapter.model_name}_{key}",
                    device=self.device,
                )
                if hasattr(candidate_adapter, "surgery_manifest"):
                    candidate_adapter.surgery_manifest = surgery_manifest

                pipeline = BenchmarkPipeline(
                    model_adapter=candidate_adapter,
                    task_keys=tasks,
                    saver=self.saver,
                    device=self.device,
                )
                sweep_results[key] = pipeline.run(**kwargs)

            except Exception as e:
                logger.error("Failed evaluation of compressor candidate '%s': %s", key, e)
                sweep_results[key] = {"error": str(e), "status": "failed"}
            finally:
                sanitize_memory()

        return sweep_results


__all__ = [
    "BenchmarkPipeline",
    "SweepController",
    "sanitize_memory",
]
