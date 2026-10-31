#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Recipe Benchmark Runner for Transformation Pipelines.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from neuralops.pipeline_transform.executor import execute_plan
from neuralops.pipeline_transform.planner import plan_recipe
from neuralops.recipe.loader import load_recipe
from neuralops.recipe.schema import Recipe
from neuralops.validation.gates.empirical.accuracy import _evaluate_accuracy
from neuralops.validation.gates.numerical.forward_pass import _get_dummy_input


def _measure_latency_and_throughput(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 2,
    seq_len: int = 16,
    repeats: int = 3,
    warmup: int = 1,
) -> Dict[str, Any]:
    """Measure inference latency and throughput on the specified device."""
    model.eval()
    dummy_in = _get_dummy_input(model, batch_size=batch_size, seq_len=seq_len)
    if isinstance(dummy_in, torch.Tensor):
        dummy_in = dummy_in.to(device)
    elif isinstance(dummy_in, dict):
        dummy_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dummy_in.items()}

    with torch.no_grad():
        for _ in range(warmup):
            if isinstance(dummy_in, dict):
                _ = model(**dummy_in)
            else:
                _ = model(dummy_in)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    times_ms: List[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            if isinstance(dummy_in, dict):
                _ = model(**dummy_in)
            else:
                _ = model(dummy_in)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = statistics.median(times_ms)
    total_tokens = batch_size * seq_len * repeats
    total_time_sec = sum(times_ms) / 1000.0
    throughput = float(total_tokens / max(total_time_sec, 1e-9))

    return {
        "median_latency_ms": median_ms,
        "throughput_tok_per_sec": throughput,
        "runs": repeats,
    }


def _measure_vram(
    model: nn.Module,
    device: torch.device,
    batch_size: int = 2,
    seq_len: int = 16,
) -> Dict[str, Any]:
    """Measure peak VRAM on CUDA device or return skipped on CPU."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "skipped": True,
            "reason": "CUDA device unavailable; peak VRAM measurement skipped (zero-fabrication policy)",
            "peak_vram_mb": None,
        }

    try:
        dummy_in = _get_dummy_input(model, batch_size=batch_size, seq_len=seq_len)
        if isinstance(dummy_in, torch.Tensor):
            dummy_in = dummy_in.to(device)
        elif isinstance(dummy_in, dict):
            dummy_in = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dummy_in.items()}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.eval()

        with torch.no_grad():
            if isinstance(dummy_in, dict):
                _ = model(**dummy_in)
            else:
                _ = model(dummy_in)
            torch.cuda.synchronize(device)

        peak_bytes = torch.cuda.max_memory_allocated(device)
        return {
            "skipped": False,
            "peak_vram_mb": float(peak_bytes / (1024 * 1024)),
            "peak_vram_bytes": peak_bytes,
        }
    except Exception as e:
        return {
            "skipped": True,
            "reason": f"VRAM measurement failed: {e}",
            "peak_vram_mb": None,
        }


def run_recipe_benchmark(
    recipe: Union[str, Recipe, Dict[str, Any]],
    device: str = "cpu",
    repeats: int = 3,
    model: Optional[nn.Module] = None,
    dataset: Optional[Any] = None,
    batch_size: int = 2,
    seq_len: int = 16,
) -> Dict[str, Any]:
    """Run an end-to-end recipe benchmark measuring latency, throughput, VRAM, and accuracy.

    Follows strict zero-fabrication standards (AGENTS.md §6.2):
    - Unmeasured metrics due to missing hardware or dataset are explicitly returned with skipped=True.
    - All reported timings and throughput are measured using wall-clock timers.
    """
    if isinstance(recipe, str):
        recipe_obj = load_recipe(recipe)
    elif isinstance(recipe, dict):
        recipe_obj = Recipe.from_dict(recipe)
    else:
        recipe_obj = recipe

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        target_device = torch.device("cpu")

    # Fallback toy model if none provided
    if model is None:
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    model = model.to(target_device)

    # 1. Baseline measurements
    base_perf = _measure_latency_and_throughput(
        model,
        target_device,
        batch_size=batch_size,
        seq_len=seq_len,
        repeats=repeats,
    )
    base_vram = _measure_vram(model, target_device, batch_size=batch_size, seq_len=seq_len)

    base_accuracy: Dict[str, Any]
    if dataset is not None:
        try:
            acc = _evaluate_accuracy(model, dataset)
            base_accuracy = {"skipped": False, "accuracy": acc}
        except Exception as e:
            base_accuracy = {"skipped": True, "reason": str(e), "accuracy": None}
    else:
        base_accuracy = {
            "skipped": True,
            "reason": "No evaluation dataset provided; accuracy benchmark skipped",
            "accuracy": None,
        }

    # 2. Plan and execute recipe
    plan = plan_recipe(recipe_obj, model)
    exec_result = execute_plan(plan, model)

    transformed_model = exec_result.final_model.to(target_device)

    # 3. Transformed measurements
    trans_perf = _measure_latency_and_throughput(
        transformed_model,
        target_device,
        batch_size=batch_size,
        seq_len=seq_len,
        repeats=repeats,
    )
    trans_vram = _measure_vram(transformed_model, target_device, batch_size=batch_size, seq_len=seq_len)

    trans_accuracy: Dict[str, Any]
    if dataset is not None:
        try:
            acc = _evaluate_accuracy(transformed_model, dataset)
            trans_accuracy = {"skipped": False, "accuracy": acc}
        except Exception as e:
            trans_accuracy = {"skipped": True, "reason": str(e), "accuracy": None}
    else:
        trans_accuracy = {
            "skipped": True,
            "reason": "No evaluation dataset provided; accuracy benchmark skipped",
            "accuracy": None,
        }

    # 4. Compute comparative metrics
    base_lat = base_perf["median_latency_ms"]
    trans_lat = trans_perf["median_latency_ms"]
    speedup = float(base_lat / max(trans_lat, 1e-9))

    # Per-stage records
    stage_summaries = []
    for s in exec_result.stage_records:
        stage_summaries.append({
            "stage_id": s.stage_id,
            "operation": s.operation_name,
            "duration_seconds": s.duration,
            "success": s.success,
        })

    return {
        "recipe_name": recipe_obj.metadata.name,
        "device": str(target_device),
        "success": exec_result.success,
        "baseline": {
            "latency_ms": base_lat,
            "throughput_tok_per_sec": base_perf["throughput_tok_per_sec"],
            "vram": base_vram,
            "accuracy": base_accuracy,
        },
        "transformed": {
            "latency_ms": trans_lat,
            "throughput_tok_per_sec": trans_perf["throughput_tok_per_sec"],
            "vram": trans_vram,
            "accuracy": trans_accuracy,
        },
        "comparison": {
            "speedup": speedup,
            "latency_delta_ms": trans_lat - base_lat,
        },
        "stages": stage_summaries,
    }


__all__ = ["run_recipe_benchmark"]
