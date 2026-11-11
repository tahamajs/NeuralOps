#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Canonical 5-format benchmark ledger exporter for the NeuralOps Multi-Zoo pipelines.

Reads up to 6 zoo JSONL ledgers:
  1. Qwen Attention Zoo (`results/qwen_attention_zoo/metrics.jsonl`)
  2. BERT Compressor Zoo (`results/bert_compressor_zoo/multirank_lora_ledger.jsonl`)
  3. BERT Attention Zoo (`results/bert_attention_zoo/metrics.jsonl`)
  4. Qwen Compressor Zoo (`results/qwen_compressor_zoo/metrics.jsonl`)
  5. VLM LLaVA ScienceQA Zoo (`results/vlm_llava_scienceqa_zoo/metrics.jsonl`)
  6. TinyLlama 1.1B Zoo (`results/tinyllama_1.1b_zoo/metrics.jsonl`)

Maps every verified entry into canonical `BenchmarkResult` records with exact
recorded hardware, metrics, deltas, timestamps, and fail-closed statuses, then
exports all 5 canonical formats via `save_all_formats` (JSON, JSONL, CSV,
Markdown, and Parquet) alongside an aggregated `zoo_overview.json`.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sys
from typing import Any
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralops.benchmarks.results.saver import save_all_formats
from neuralops.benchmarks.results.schema import (
    BaselineComparison,
    BenchmarkResult,
    BenchmarkStatus,
    EnvironmentMetadata,
    HardwareUsage,
    RunMetadata,
)

LOG = logging.getLogger("export_dual_zoo_5formats")

EXIT_SUCCESS = 0
EXIT_USAGE_ERROR = 2
EXIT_NUMERICAL_FAILURE = 4
EXIT_OUTPUT_ERROR = 6

_INDEX_FILENAMES = (
    "results.json",
    "results.jsonl",
    "results.csv",
    "summary.md",
    "results.parquet",
    "zoo_overview.json",
)


def _validate_safe_segment(value: Any, field_name: str, *, allow_slashes: bool = False) -> str:
    """Validate that a path-building identifier is non-empty and free of traversal."""
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Missing required field '{field_name}'.")
    if "\x00" in text or ".." in Path(text).parts:
        raise ValueError(f"Unsafe path segment in field '{field_name}': {text!r}")
    if "\\" in text or Path(text).is_absolute():
        raise ValueError(f"Unsafe path separator in field '{field_name}': {text!r}")
    if not allow_slashes:
        if "/" in text:
            raise ValueError(f"Unsafe path separator in field '{field_name}': {text!r}")
    elif text.startswith("/") or text.endswith("/") or "//" in text:
        raise ValueError(f"Unsafe slash placement in field '{field_name}': {text!r}")
    return text


def _extract_finite_float(
    record: dict[str, Any],
    key: str,
    numeric_errors: list[str],
) -> float | None:
    """Extract a finite float value from a record or record a numerical error."""
    if key not in record or record[key] is None:
        return None
    raw = record[key]
    if isinstance(raw, bool):
        numeric_errors.append(f"Field '{key}' has invalid boolean value {raw!r}.")
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        numeric_errors.append(f"Field '{key}' is non-numeric: {raw!r}.")
        return None
    if not math.isfinite(val):
        numeric_errors.append(f"Field '{key}' is non-finite: {raw!r}.")
        return None
    return val


def _extract_int(
    record: dict[str, Any],
    key: str,
    numeric_errors: list[str],
) -> int | None:
    """Extract an integer value from a record or record a validation error."""
    if key not in record or record[key] is None:
        return None
    raw = record[key]
    if isinstance(raw, bool):
        numeric_errors.append(f"Field '{key}' has invalid boolean value {raw!r}.")
        return None
    try:
        if isinstance(raw, float) and not raw.is_integer():
            numeric_errors.append(f"Field '{key}' is not an integer: {raw!r}.")
            return None
        return int(raw)
    except (TypeError, ValueError):
        numeric_errors.append(f"Field '{key}' is not a valid integer: {raw!r}.")
        return None


def _normalize_zoo_status(raw_status: Any) -> tuple[BenchmarkStatus, list[str]]:
    """Map zoo ledger status strings onto canonical BenchmarkStatus values."""
    warnings: list[str] = []
    if raw_status is None or str(raw_status).strip() == "":
        warnings.append("Source record omitted 'status'; marked as failed.")
        return BenchmarkStatus.FAILED, warnings

    raw_str = str(raw_status).strip()
    upper = raw_str.upper()

    if upper in {"PASSED", "PASS", "SUCCESS", "OK"}:
        return BenchmarkStatus.SUCCESS, warnings
    if upper in {"SKIPPED_OOM", "SKIPPED", "SKIP"}:
        if upper != "SKIPPED":
            warnings.append(f"Normalized source status '{raw_str}' to 'skipped'.")
        return BenchmarkStatus.SKIPPED, warnings
    if upper in {"TIMEOUT", "TIMED_OUT"}:
        return BenchmarkStatus.TIMEOUT, warnings
    if upper in {"FAIL_NUMERICS", "FAIL_DELTA", "NUMERICAL_ERROR"}:
        return BenchmarkStatus.FAIL_NUMERICS, warnings
    if upper in {"DEPENDENCY_ERROR", "MISSING_DEPENDENCY"}:
        return BenchmarkStatus.DEPENDENCY_ERROR, warnings
    if upper in {"HARDWARE_ERROR", "CUDA_ERROR"}:
        return BenchmarkStatus.HARDWARE_ERROR, warnings
    if upper in {"FAIL_TRAINING", "FAIL_CRASH", "FAILED", "FAIL", "ERROR"} or upper.startswith("FAIL"):
        if upper != "FAILED":
            warnings.append(f"Normalized source status '{raw_str}' to 'failed'.")
        return BenchmarkStatus.FAILED, warnings

    warnings.append(f"Unrecognized source status '{raw_str}'; fail-closed as 'failed'.")
    return BenchmarkStatus.FAILED, warnings


def _build_environment_and_hardware(
    record: dict[str, Any],
    default_device: str | None,
    numeric_errors: list[str],
) -> tuple[EnvironmentMetadata, HardwareUsage, str, dict[str, Any]]:
    """Construct EnvironmentMetadata and HardwareUsage strictly from recorded fields."""
    env_dict = record.get("environment") if isinstance(record.get("environment"), dict) else {}

    peak_vram_mb = _extract_finite_float(record, "peak_vram_mb", numeric_errors)
    avg_vram_mb = _extract_finite_float(record, "avg_vram_mb", numeric_errors)
    peak_ram_mb = _extract_finite_float(record, "peak_ram_mb", numeric_errors)
    gpu_util = _extract_finite_float(record, "gpu_utilization_avg", numeric_errors)
    gpu_temp = _extract_finite_float(record, "gpu_temperature_max", numeric_errors)
    gpu_power = _extract_finite_float(record, "gpu_power_avg", numeric_errors)

    hardware_usage = HardwareUsage(
        peak_vram_mb=peak_vram_mb if peak_vram_mb is not None else 0.0,
        avg_vram_mb=avg_vram_mb if avg_vram_mb is not None else 0.0,
        peak_ram_mb=peak_ram_mb if peak_ram_mb is not None else 0.0,
        gpu_utilization_avg=gpu_util,
        gpu_temperature_max=gpu_temp,
        gpu_power_avg=gpu_power,
    )

    device_type = (
        record.get("device_type")
        or record.get("device")
        or env_dict.get("device_type")
        or env_dict.get("device")
        or default_device
        or ("cuda" if (peak_vram_mb is not None and peak_vram_mb > 0.0) else "unknown")
    )

    gpu_model = record.get("gpu_model") or env_dict.get("gpu_model")
    raw_gpu_count = record.get("gpu_count") if record.get("gpu_count") is not None else env_dict.get("gpu_count")
    gpu_count = _extract_int({"gpu_count": raw_gpu_count}, "gpu_count", numeric_errors) if raw_gpu_count is not None else (1 if gpu_model else 0)
    raw_gpu_vram = record.get("gpu_vram_gb") if record.get("gpu_vram_gb") is not None else env_dict.get("gpu_vram_gb")
    gpu_vram_gb = _extract_finite_float({"gpu_vram_gb": raw_gpu_vram}, "gpu_vram_gb", numeric_errors) if raw_gpu_vram is not None else 0.0
    cuda_version = record.get("cuda_version") or env_dict.get("cuda_version")

    # Never call EnvironmentMetadata.capture() because the exporter host may differ
    # from the training/evaluation hardware recorded in the ledger.
    env_kwargs: dict[str, Any] = {
        "python_version": str(env_dict.get("python_version", "unknown")),
        "python_executable": str(env_dict.get("python_executable", "")),
        "platform": str(env_dict.get("platform", "unknown")),
        "platform_version": str(env_dict.get("platform_version", "unknown")),
        "cpu_model": str(env_dict.get("cpu_model", "unknown")),
        "cpu_count": int(env_dict.get("cpu_count", 0) or 0),
        "ram_gb": float(env_dict.get("ram_gb", 0.0) or 0.0),
        "gpu_model": str(gpu_model) if gpu_model is not None else None,
        "gpu_count": int(gpu_count) if gpu_count is not None else 0,
        "gpu_vram_gb": float(gpu_vram_gb) if gpu_vram_gb is not None else 0.0,
        "cuda_compute_capability": env_dict.get("cuda_compute_capability"),
        "cuda_version": str(cuda_version) if cuda_version is not None else None,
        "cudnn_version": env_dict.get("cudnn_version"),
        "torch_version": str(record.get("torch_version") or env_dict.get("torch_version") or "unknown"),
        "triton_version": env_dict.get("triton_version"),
    }
    environment = EnvironmentMetadata(**env_kwargs)

    resources: dict[str, Any] = {}
    if peak_vram_mb is not None:
        resources["peak_vram_mb"] = peak_vram_mb
    if avg_vram_mb is not None:
        resources["avg_vram_mb"] = avg_vram_mb
    if peak_ram_mb is not None:
        resources["peak_ram_mb"] = peak_ram_mb

    return environment, hardware_usage, str(device_type), resources


def _build_run_metadata(
    record: dict[str, Any],
    zoo_name: str,
    run_name: str,
    identity_seed: str,
    duration_s: float | None,
    warnings: list[str],
) -> tuple[RunMetadata, dict[str, Any]]:
    """Build deterministic RunMetadata using exact recorded timestamps when available."""
    run_id = str(record.get("run_id") or uuid.uuid5(uuid.NAMESPACE_URL, f"neuralops:{zoo_name}:{identity_seed}"))

    ts_start = record.get("timestamp_start") or record.get("timestamp")
    ts_end = record.get("timestamp_end") or record.get("timestamp") or ts_start
    meta_extra: dict[str, Any] = {}

    rm_kwargs: dict[str, Any] = {
        "run_id": run_id,
        "run_name": run_name,
    }
    if ts_start:
        rm_kwargs["timestamp_start"] = str(ts_start)
    if ts_end:
        rm_kwargs["timestamp_end"] = str(ts_end)
    if not ts_start and not ts_end:
        warnings.append("Missing source timestamp; exporter fallback timestamp recorded.")
        meta_extra["source_timestamp_missing"] = True

    if duration_s is not None:
        rm_kwargs["wall_clock_duration_s"] = duration_s
        rm_kwargs["total_duration_s"] = duration_s
        rm_kwargs["training_duration_s"] = duration_s

    if record.get("git_commit"):
        commit = str(record["git_commit"])
        rm_kwargs["git_commit"] = commit
        rm_kwargs["git_commit_short"] = commit[:10]
        meta_extra["git_commit_source"] = "record"
    else:
        meta_extra["git_commit_source"] = "exporter_repo"

    if "git_dirty" in record and isinstance(record["git_dirty"], bool):
        rm_kwargs["git_dirty"] = record["git_dirty"]

    return RunMetadata(**rm_kwargs), meta_extra


def _finalize_status_and_errors(
    status: BenchmarkStatus,
    numeric_errors: list[str],
    record: dict[str, Any],
    required_error: str | None,
) -> tuple[BenchmarkStatus, str | None]:
    """Enforce fail-closed status transitions for numerical and schema errors."""
    recorded_error = str(record["error"]) if record.get("error") is not None else None

    if numeric_errors:
        combined = "; ".join(numeric_errors)
        if recorded_error:
            combined = f"{recorded_error}; {combined}"
        return BenchmarkStatus.FAIL_NUMERICS, combined

    if status == BenchmarkStatus.SUCCESS and required_error:
        combined = f"{recorded_error}; {required_error}" if recorded_error else required_error
        return BenchmarkStatus.FAILED, combined

    return status, recorded_error


def map_qwen_attention_record(
    record: dict[str, Any],
    line_no: int,
    default_model: str | None = None,
    default_dataset: str | None = None,
    default_device: str | None = None,
) -> BenchmarkResult:
    """Map one Qwen Attention Zoo `metrics.jsonl` entry to `BenchmarkResult`."""
    attn_key = _validate_safe_segment(record.get("attn_key"), "attn_key")
    model_name = _validate_safe_segment(record.get("model") or default_model or "unknown", "model", allow_slashes=True)
    dataset_name = _validate_safe_segment(record.get("dataset") or default_dataset or "unknown", "dataset")

    numeric_errors: list[str] = []
    status, warnings = _normalize_zoo_status(record.get("status"))

    ppl_before = _extract_finite_float(record, "ppl_before", numeric_errors)
    ppl_after = _extract_finite_float(record, "ppl_after", numeric_errors)
    ppl_delta = _extract_finite_float(record, "ppl_delta", numeric_errors)
    train_secs = _extract_finite_float(record, "train_secs", numeric_errors)
    if train_secs is None:
        train_secs = _extract_finite_float(record, "duration_s", numeric_errors)
    tokens_per_sec = _extract_finite_float(record, "tokens_per_sec", numeric_errors)
    sst2_acc = _extract_finite_float(record, "sst2_accuracy", numeric_errors)
    if sst2_acc is None:
        sst2_acc = _extract_finite_float(record, "accuracy", numeric_errors)

    steps_s1 = _extract_int(record, "steps_s1", numeric_errors)
    steps_s2 = _extract_int(record, "steps_s2", numeric_errors)
    total_steps = _extract_int(record, "total_steps", numeric_errors)

    environment, hardware_usage, device_type, resources = _build_environment_and_hardware(
        record=record,
        default_device=default_device,
        numeric_errors=numeric_errors,
    )

    if train_secs is not None:
        resources["duration_s"] = train_secs
        resources["train_duration_s"] = train_secs
    if steps_s1 is not None:
        resources["steps_s1"] = steps_s1
    if steps_s2 is not None:
        resources["steps_s2"] = steps_s2
    if total_steps is not None:
        resources["total_steps"] = total_steps

    metrics: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] = {}
    delta_metrics: dict[str, Any] = {}

    if ppl_before is not None:
        metrics["pre_ppl"] = ppl_before
        baseline_metrics["ppl"] = ppl_before
    if ppl_after is not None:
        metrics["post_ppl"] = ppl_after
    if ppl_delta is not None:
        metrics["ppl_delta"] = ppl_delta
        delta_metrics["ppl_delta"] = ppl_delta
    elif ppl_before is not None and ppl_after is not None:
        computed_delta = round(ppl_after - ppl_before, 4)
        metrics["ppl_delta"] = computed_delta
        delta_metrics["ppl_delta"] = computed_delta
    if tokens_per_sec is not None:
        metrics["tokens_per_sec"] = tokens_per_sec
    if sst2_acc is not None:
        metrics["accuracy"] = sst2_acc

    required_error = None
    if status == BenchmarkStatus.SUCCESS and ppl_after is None:
        required_error = "Missing required metric 'ppl_after' for PASSED Qwen Attention Zoo record."

    final_status, error_msg = _finalize_status_and_errors(status, numeric_errors, record, required_error)

    pattern = record.get("pattern")
    config: dict[str, Any] = {"zoo": "qwen_attention_zoo", "rank": 0}
    if pattern:
        config["pattern"] = str(pattern)
    for cfg_key in ("seq_len", "batch_size", "grad_accum", "topk", "train_secs"):
        if cfg_key in record and record[cfg_key] is not None:
            config[cfg_key] = record[cfg_key]

    ts_token = str(record.get("timestamp") or record.get("timestamp_end") or f"line_{line_no}")
    identity_seed = f"{attn_key}|{model_name}|{dataset_name}|{pattern or ''}|{ts_token}|{line_no}"
    run_meta, meta_extra = _build_run_metadata(
        record=record,
        zoo_name="qwen_attention_zoo",
        run_name=f"qwen_attention_{attn_key}",
        identity_seed=identity_seed,
        duration_s=train_secs,
        warnings=warnings,
    )

    return BenchmarkResult(
        run_metadata=run_meta,
        module={
            "name": attn_key,
            "type": "attention",
            "module_type": "attention",
            "registry": "attention",
        },
        model={"name": model_name, "family": "qwen"},
        dataset={"name": dataset_name, "task": "causal_lm"},
        config=config,
        environment=environment,
        hardware_usage=hardware_usage,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        delta_metrics=delta_metrics,
        resources=resources,
        status=final_status.value,
        error=error_msg,
        warnings=warnings,
        training_type="distillation",
        benchmark_suite="qwen_attention_zoo",
        device_type=device_type,
        metadata={
            "zoo": "qwen_attention_zoo",
            "source_line": line_no,
            "raw_status": str(record.get("status", "")),
            **meta_extra,
        },
    )


def map_bert_compressor_record(
    record: dict[str, Any],
    line_no: int,
    default_model: str = "bert-base-uncased",
    default_dataset: str = "sst2",
    default_device: str | None = None,
) -> BenchmarkResult:
    """Map one BERT Compressor Zoo (`multirank_lora_ledger.jsonl` or `metrics.jsonl`) entry to `BenchmarkResult`."""
    comp_key = _validate_safe_segment(
        record.get("compressor") or record.get("comp_key") or record.get("module"),
        "compressor",
    )
    model_name = _validate_safe_segment(record.get("model") or default_model, "model", allow_slashes=True)
    dataset_name = _validate_safe_segment(record.get("dataset") or default_dataset, "dataset")

    numeric_errors: list[str] = []
    status, warnings = _normalize_zoo_status(record.get("status"))

    rank = _extract_int(record, "rank", numeric_errors)
    if rank is not None and rank < 0:
        numeric_errors.append(f"Field 'rank' must be non-negative, got {rank}.")
        rank = None
    effective_rank = rank if rank is not None else 0

    ratio = _extract_first_finite_float(
        record,
        ("ratio", "param_reduction", "compression_ratio", "effective_ratio"),
        numeric_errors,
    )
    orig_params = _extract_int(record, "orig_params", numeric_errors)
    comp_params = _extract_int(record, "comp_params", numeric_errors)
    zero_shot_acc = _extract_first_finite_float(record, ("zero_shot_acc", "pre_kd_acc", "acc_before"), numeric_errors)
    zero_shot_loss = _extract_first_finite_float(record, ("zero_shot_loss", "pre_kd_loss", "loss_before"), numeric_errors)
    delta_zero = _extract_first_finite_float(record, ("delta_zero", "zero_delta"), numeric_errors)
    healed_acc = _extract_first_finite_float(
        record,
        ("healed_acc", "post_kd_acc", "acc_after", "accuracy", "sst2_accuracy"),
        numeric_errors,
    )
    healed_loss = _extract_first_finite_float(
        record,
        ("healed_loss", "post_kd_loss", "loss_after", "loss"),
        numeric_errors,
    )
    delta_healed = _extract_first_finite_float(record, ("delta_healed", "acc_gain", "acc_delta"), numeric_errors)
    ppl_before = _extract_first_finite_float(record, ("ppl_before", "pre_kd_ppl"), numeric_errors)
    ppl_after = _extract_first_finite_float(record, ("ppl_after", "post_kd_ppl"), numeric_errors)
    ppl_delta = _extract_first_finite_float(record, ("ppl_delta", "ppl_gain"), numeric_errors)
    duration_s = _extract_first_finite_float(
        record,
        ("duration_s", "eval_time", "train_secs"),
        numeric_errors,
    )

    lora_healed = record.get("lora_healed")
    if lora_healed is not None and not isinstance(lora_healed, bool):
        numeric_errors.append(f"Field 'lora_healed' must be boolean, got {lora_healed!r}.")
        lora_healed = None

    environment, hardware_usage, device_type, resources = _build_environment_and_hardware(
        record=record,
        default_device=default_device,
        numeric_errors=numeric_errors,
    )
    if duration_s is not None:
        resources["duration_s"] = duration_s

    metrics: dict[str, Any] = {}
    if zero_shot_acc is not None:
        metrics["zero_shot_acc"] = zero_shot_acc
    if zero_shot_loss is not None:
        metrics["zero_shot_loss"] = zero_shot_loss
    if healed_acc is not None:
        metrics["healed_acc"] = healed_acc
    if healed_loss is not None:
        metrics["healed_loss"] = healed_loss
    if ratio is not None:
        metrics["compression_ratio"] = ratio
    if ppl_before is not None:
        metrics["pre_ppl"] = ppl_before
    if ppl_after is not None:
        metrics["post_ppl"] = ppl_after
    if ppl_delta is not None:
        metrics["ppl_delta"] = ppl_delta

    # Populate canonical accuracy/loss fields consumed by saver.py indices.
    if lora_healed and healed_acc is not None:
        metrics["accuracy"] = healed_acc
        if healed_loss is not None:
            metrics["loss"] = healed_loss
    elif healed_acc is not None:
        metrics["accuracy"] = healed_acc
        if healed_loss is not None:
            metrics["loss"] = healed_loss
        elif zero_shot_loss is not None:
            metrics["loss"] = zero_shot_loss
    elif zero_shot_acc is not None:
        metrics["accuracy"] = zero_shot_acc
        if zero_shot_loss is not None:
            metrics["loss"] = zero_shot_loss

    baseline_metrics: dict[str, Any] = {}
    if orig_params is not None:
        baseline_metrics["orig_params"] = orig_params
    if record.get("teacher_acc") is not None:
        teacher_acc = _extract_finite_float(record, "teacher_acc", numeric_errors)
        if teacher_acc is not None:
            baseline_metrics["accuracy"] = teacher_acc
    if ppl_before is not None:
        baseline_metrics["ppl"] = ppl_before

    delta_metrics: dict[str, Any] = {}
    if delta_zero is not None:
        delta_metrics["delta_zero"] = delta_zero
    if delta_healed is not None:
        delta_metrics["delta_healed"] = delta_healed
    if ppl_delta is not None:
        delta_metrics["ppl_delta"] = ppl_delta

    params_reduction_pct = 0.0
    if orig_params is not None and comp_params is not None and orig_params > 0:
        params_reduction_pct = round((1.0 - (comp_params / orig_params)) * 100.0, 4)

    baseline_comparison = BaselineComparison(
        baseline_accuracy=baseline_metrics.get("accuracy"),
        accuracy_drop=delta_healed if delta_healed is not None else delta_zero,
        compression_ratio=ratio if ratio is not None else 1.0,
        params_reduction_pct=params_reduction_pct,
    )

    required_error = None
    if status == BenchmarkStatus.SUCCESS:
        if "accuracy" not in metrics and "post_ppl" not in metrics:
            required_error = "Missing required accuracy or perplexity metric for PASSED compressor record."
        elif ratio is None or ratio <= 0.0:
            required_error = "Missing or non-positive 'ratio' metric for PASSED compressor record."

    final_status, error_msg = _finalize_status_and_errors(status, numeric_errors, record, required_error)

    config: dict[str, Any] = {"zoo": "bert_compressor_zoo", "rank": effective_rank}
    if lora_healed is not None:
        config["lora_healed"] = lora_healed

    model_dict: dict[str, Any] = {"name": model_name, "family": "bert"}
    if orig_params is not None:
        model_dict["num_params"] = orig_params
    if comp_params is not None:
        model_dict["num_params_after"] = comp_params

    ts_token = str(record.get("timestamp") or record.get("timestamp_end") or f"line_{line_no}")
    identity_seed = f"{comp_key}|{model_name}|{dataset_name}|rank{effective_rank}|{ts_token}|{line_no}"
    run_meta, meta_extra = _build_run_metadata(
        record=record,
        zoo_name="bert_compressor_zoo",
        run_name=f"bert_compressor_{comp_key}_r{effective_rank}",
        identity_seed=identity_seed,
        duration_s=duration_s,
        warnings=warnings,
    )

    return BenchmarkResult(
        run_metadata=run_meta,
        module={
            "name": comp_key,
            "type": "compressor",
            "module_type": "compressor",
            "registry": "compressor",
            "rank": effective_rank,
        },
        model=model_dict,
        dataset={"name": dataset_name, "task": "classification", "split": "validation"},
        config=config,
        environment=environment,
        hardware_usage=hardware_usage,
        metrics=metrics,
        baseline_comparison=baseline_comparison,
        baseline_metrics=baseline_metrics,
        delta_metrics=delta_metrics,
        resources=resources,
        status=final_status.value,
        error=error_msg,
        warnings=warnings,
        training_type="lora_recovery" if lora_healed else "zero_shot_compression",
        benchmark_suite="bert_compressor_zoo",
        device_type=device_type,
        metadata={
            "zoo": "bert_compressor_zoo",
            "source_line": line_no,
            "raw_status": str(record.get("status", "")),
            **meta_extra,
        },
    )


def _extract_first_finite_float(
    record: dict[str, Any],
    keys: tuple[str, ...],
    numeric_errors: list[str],
) -> float | None:
    """Extract the first non-None finite float from candidate keys."""
    for key in keys:
        if key in record and record[key] is not None:
            return _extract_finite_float(record, key, numeric_errors)
    return None


@dataclass(frozen=True)
class _ZooMapperSpec:
    """Declarative mapping specification for a Zoo JSONL ledger."""

    zoo_name: str
    run_prefix: str
    module_keys: tuple[str, ...]
    module_field_label: str
    module_type: str
    registry: str
    default_model: str
    model_family: str
    default_dataset: str
    default_task: str
    training_type: str
    require_any_metric: tuple[str, ...]
    allow_dataset_slashes: bool = False
    extract_compression: bool = False
    dataset_split: str | None = None


_BERT_ATTN_SPEC = _ZooMapperSpec(
    zoo_name="bert_attention_zoo",
    run_prefix="bert_attention",
    module_keys=("attn_key", "module"),
    module_field_label="attn_key",
    module_type="attention",
    registry="attention",
    default_model="bert-base-uncased",
    model_family="bert",
    default_dataset="sst2",
    default_task="classification",
    training_type="distillation",
    require_any_metric=("post_ppl", "accuracy"),
)

_QWEN_COMP_SPEC = _ZooMapperSpec(
    zoo_name="qwen_compressor_zoo",
    run_prefix="qwen_compressor",
    module_keys=("compressor", "comp_key", "module"),
    module_field_label="compressor",
    module_type="compressor",
    registry="compressor",
    default_model="Qwen/Qwen2.5-0.5B",
    model_family="qwen",
    default_dataset="wikitext-2-raw-v1",
    default_task="causal_lm",
    training_type="compression_distillation",
    require_any_metric=("post_ppl", "accuracy"),
    extract_compression=True,
)

_VLM_SPEC = _ZooMapperSpec(
    zoo_name="vlm_llava_scienceqa_zoo",
    run_prefix="vlm_attention",
    module_keys=("attn_key", "module"),
    module_field_label="attn_key",
    module_type="attention",
    registry="attention",
    default_model="llava-hf/llava-interleave-qwen-0.5b-hf",
    model_family="vlm",
    default_dataset="derek-thomas/ScienceQA",
    default_task="vqa_multimodal",
    training_type="distillation",
    require_any_metric=("accuracy", "post_ppl"),
    allow_dataset_slashes=True,
)

_TINYLLAMA_SPEC = _ZooMapperSpec(
    zoo_name="tinyllama_1.1b_zoo",
    run_prefix="tinyllama_attention",
    module_keys=("attn_key", "module"),
    module_field_label="attn_key",
    module_type="attention",
    registry="attention",
    default_model="TinyLlama/TinyLlama_v1.1",
    model_family="llama",
    default_dataset="wikitext-2",
    default_task="causal_lm",
    training_type="distillation",
    require_any_metric=("post_ppl", "accuracy"),
)

_LLAMA_ATTN_SPEC = _ZooMapperSpec(
    zoo_name="llama_attention_zoo",
    run_prefix="llama_attention",
    module_keys=("attn_key", "module"),
    module_field_label="attn_key",
    module_type="attention",
    registry="attention",
    default_model="TinyLlama/TinyLlama_v1.1",
    model_family="llama",
    default_dataset="wikitext-2",
    default_task="causal_lm",
    training_type="distillation",
    require_any_metric=("post_ppl", "accuracy"),
)


def _map_zoo_distillation_record(
    record: dict[str, Any],
    line_no: int,
    spec: _ZooMapperSpec,
    default_model: str | None = None,
    default_dataset: str | None = None,
    default_device: str | None = None,
) -> BenchmarkResult:
    """Map one attention/compressor zoo JSONL record strictly into BenchmarkResult."""
    raw_mod_key = next((record.get(k) for k in spec.module_keys if record.get(k) is not None), None)
    mod_key = _validate_safe_segment(raw_mod_key, spec.module_field_label)
    model_name = _validate_safe_segment(
        record.get("model") or default_model or spec.default_model,
        "model",
        allow_slashes=True,
    )

    inferred_dataset = record.get("dataset") or default_dataset
    inferred_task = spec.default_task
    if not inferred_dataset:
        if (
            spec.zoo_name == "bert_attention_zoo"
            and any(record.get(k) is not None for k in ("ppl_after", "post_kd_ppl", "ppl_before", "pre_kd_ppl"))
            and record.get("teacher_baseline_acc") is None
        ):
            inferred_dataset = "wikitext-2"
            inferred_task = "masked_lm"
        else:
            inferred_dataset = spec.default_dataset

    dataset_name = _validate_safe_segment(
        inferred_dataset,
        "dataset",
        allow_slashes=spec.allow_dataset_slashes,
    )

    numeric_errors: list[str] = []
    status, warnings = _normalize_zoo_status(record.get("status"))

    ppl_before = _extract_first_finite_float(record, ("ppl_before", "pre_kd_ppl"), numeric_errors)
    ppl_after = _extract_first_finite_float(record, ("ppl_after", "post_kd_ppl"), numeric_errors)
    ppl_delta = _extract_finite_float(record, "ppl_delta", numeric_errors)

    acc_before = _extract_first_finite_float(record, ("acc_before", "pre_kd_acc"), numeric_errors)
    acc_after = _extract_first_finite_float(
        record,
        ("acc_after", "post_kd_acc", "sst2_accuracy", "accuracy", "post_kd_sst2_acc"),
        numeric_errors,
    )
    acc_delta = _extract_first_finite_float(record, ("acc_delta", "acc_gain"), numeric_errors)
    post_kd_sst2_acc = (
        _extract_finite_float(record, "post_kd_sst2_acc", numeric_errors)
        if record.get("post_kd_acc") is not None and record.get("post_kd_sst2_acc") is not None
        else None
    )

    loss_val = _extract_first_finite_float(record, ("post_kd_loss", "loss"), numeric_errors)
    teacher_acc = _extract_first_finite_float(
        record,
        ("teacher_acc", "teacher_baseline_acc", "teacher_sst2_acc"),
        numeric_errors,
    )
    teacher_ppl = _extract_finite_float(record, "teacher_ppl", numeric_errors)

    train_secs = _extract_first_finite_float(record, ("train_secs", "duration_s", "eval_time"), numeric_errors)
    tokens_per_sec = _extract_finite_float(record, "tokens_per_sec", numeric_errors)
    steps_s1 = _extract_int(record, "steps_s1", numeric_errors)
    steps_s2 = _extract_int(record, "steps_s2", numeric_errors)
    total_steps = _extract_int(record, "total_steps", numeric_errors)
    exit_code = _extract_int(record, "exit_code", numeric_errors)

    rank: int | None = None
    ratio: float | None = None
    sparsity_pct: float | None = None
    orig_params: int | None = None
    comp_params: int | None = None
    if spec.extract_compression:
        rank = _extract_int(record, "rank", numeric_errors)
        if rank is not None and rank < 0:
            numeric_errors.append(f"Field 'rank' must be non-negative, got {rank}.")
            rank = None
        ratio = _extract_first_finite_float(
            record,
            ("param_reduction", "ratio", "compression_ratio"),
            numeric_errors,
        )
        sparsity_pct = _extract_finite_float(record, "sparsity_pct", numeric_errors)
        orig_params = _extract_int(record, "orig_params", numeric_errors)
        comp_params = _extract_int(record, "comp_params", numeric_errors)

    environment, hardware_usage, device_type, resources = _build_environment_and_hardware(
        record=record,
        default_device=default_device,
        numeric_errors=numeric_errors,
    )

    if train_secs is not None:
        resources["duration_s"] = train_secs
        resources["train_duration_s"] = train_secs
    if steps_s1 is not None:
        resources["steps_s1"] = steps_s1
    if steps_s2 is not None:
        resources["steps_s2"] = steps_s2
    if total_steps is not None:
        resources["total_steps"] = total_steps
    if exit_code is not None:
        resources["exit_code"] = exit_code

    metrics: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] = {}
    delta_metrics: dict[str, Any] = {}

    if ppl_before is not None:
        metrics["pre_ppl"] = ppl_before
        baseline_metrics["ppl"] = ppl_before
    if ppl_after is not None:
        metrics["post_ppl"] = ppl_after
    if ppl_delta is not None:
        metrics["ppl_delta"] = ppl_delta
        delta_metrics["ppl_delta"] = ppl_delta
    elif ppl_before is not None and ppl_after is not None:
        computed_ppl_delta = round(ppl_after - ppl_before, 4)
        metrics["ppl_delta"] = computed_ppl_delta
        delta_metrics["ppl_delta"] = computed_ppl_delta

    if acc_before is not None:
        metrics["pre_acc"] = acc_before
        baseline_metrics["pre_acc"] = acc_before
    if acc_after is not None:
        metrics["accuracy"] = acc_after
    if post_kd_sst2_acc is not None:
        metrics["sst2_accuracy"] = post_kd_sst2_acc
    if acc_delta is not None:
        metrics["acc_delta"] = acc_delta
        delta_metrics["acc_delta"] = acc_delta
    elif acc_before is not None and acc_after is not None:
        computed_acc_delta = round(acc_after - acc_before, 4)
        metrics["acc_delta"] = computed_acc_delta
        delta_metrics["acc_delta"] = computed_acc_delta

    if loss_val is not None:
        metrics["loss"] = loss_val
    if tokens_per_sec is not None:
        metrics["tokens_per_sec"] = tokens_per_sec
    if teacher_acc is not None:
        baseline_metrics["accuracy"] = teacher_acc
    if teacher_ppl is not None:
        baseline_metrics["teacher_ppl"] = teacher_ppl

    if ratio is not None:
        metrics["compression_ratio"] = ratio
    if sparsity_pct is not None:
        metrics["sparsity_pct"] = sparsity_pct
    if orig_params is not None:
        baseline_metrics["orig_params"] = orig_params

    params_reduction_pct = 0.0
    if orig_params is not None and comp_params is not None and orig_params > 0:
        params_reduction_pct = round((1.0 - (comp_params / orig_params)) * 100.0, 4)

    baseline_comparison = BaselineComparison(
        baseline_accuracy=baseline_metrics.get("accuracy"),
        accuracy_drop=delta_metrics.get("acc_delta"),
        compression_ratio=ratio if ratio is not None else 1.0,
        params_reduction_pct=params_reduction_pct,
    )

    required_error: str | None = None
    if status == BenchmarkStatus.SUCCESS:
        if not any(k in metrics for k in spec.require_any_metric):
            expected = " or ".join(f"'{k}'" for k in spec.require_any_metric)
            required_error = f"Missing required metric ({expected}) for PASSED {spec.zoo_name} record."
        elif spec.extract_compression and ratio is not None and ratio <= 0.0:
            required_error = f"Non-positive compression ratio ({ratio}) for PASSED {spec.zoo_name} record."

    final_status, error_msg = _finalize_status_and_errors(status, numeric_errors, record, required_error)

    config_rank = rank if rank is not None else 0
    config: dict[str, Any] = {"zoo": spec.zoo_name, "rank": config_rank}
    for cfg_key in (
        "pattern",
        "seq_len",
        "batch_size",
        "grad_accum",
        "topk",
        "train_secs",
        "total_layers",
        "surgeried_layers",
        "anchor_layers",
        "hf_org",
    ):
        if cfg_key in record and record[cfg_key] is not None:
            config[cfg_key] = record[cfg_key]

    module_dict: dict[str, Any] = {
        "name": mod_key,
        "type": spec.module_type,
        "module_type": spec.module_type,
        "registry": spec.registry,
    }
    if rank is not None:
        module_dict["rank"] = rank

    model_dict: dict[str, Any] = {"name": model_name, "family": spec.model_family}
    if orig_params is not None:
        model_dict["num_params"] = orig_params
    if comp_params is not None:
        model_dict["num_params_after"] = comp_params

    dataset_dict: dict[str, Any] = {"name": dataset_name, "task": inferred_task}
    if spec.dataset_split:
        dataset_dict["split"] = spec.dataset_split

    pattern = record.get("pattern")
    ts_token = str(record.get("timestamp") or record.get("timestamp_end") or f"line_{line_no}")
    rank_token = f"|rank{rank}" if rank is not None else ""
    identity_seed = f"{mod_key}|{model_name}|{dataset_name}{rank_token}|{pattern or ''}|{ts_token}|{line_no}"
    run_name = f"{spec.run_prefix}_{mod_key}" if rank is None else f"{spec.run_prefix}_{mod_key}_r{rank}"
    run_meta, meta_extra = _build_run_metadata(
        record=record,
        zoo_name=spec.zoo_name,
        run_name=run_name,
        identity_seed=identity_seed,
        duration_s=train_secs,
        warnings=warnings,
    )

    metadata: dict[str, Any] = {
        "zoo": spec.zoo_name,
        "source_line": line_no,
        "raw_status": str(record.get("status", "")),
        **meta_extra,
    }
    if exit_code is not None:
        metadata["exit_code"] = exit_code

    return BenchmarkResult(
        run_metadata=run_meta,
        module=module_dict,
        model=model_dict,
        dataset=dataset_dict,
        config=config,
        environment=environment,
        hardware_usage=hardware_usage,
        metrics=metrics,
        baseline_comparison=baseline_comparison,
        baseline_metrics=baseline_metrics,
        delta_metrics=delta_metrics,
        resources=resources,
        status=final_status.value,
        error=error_msg,
        warnings=warnings,
        training_type=spec.training_type,
        benchmark_suite=spec.zoo_name,
        device_type=device_type,
        metadata=metadata,
    )


def map_bert_attention_record(
    record: dict[str, Any],
    line_no: int,
    default_model: str | None = None,
    default_dataset: str | None = None,
    default_device: str | None = None,
) -> BenchmarkResult:
    """Map one BERT Attention Zoo `metrics.jsonl` entry to `BenchmarkResult`."""
    return _map_zoo_distillation_record(
        record=record,
        line_no=line_no,
        spec=_BERT_ATTN_SPEC,
        default_model=default_model,
        default_dataset=default_dataset,
        default_device=default_device,
    )


def map_qwen_compressor_record(
    record: dict[str, Any],
    line_no: int,
    default_model: str | None = None,
    default_dataset: str | None = None,
    default_device: str | None = None,
) -> BenchmarkResult:
    """Map one Qwen Compressor Zoo `metrics.jsonl` entry to `BenchmarkResult`."""
    return _map_zoo_distillation_record(
        record=record,
        line_no=line_no,
        spec=_QWEN_COMP_SPEC,
        default_model=default_model,
        default_dataset=default_dataset,
        default_device=default_device,
    )


def map_vlm_attention_record(
    record: dict[str, Any],
    line_no: int,
    default_model: str | None = None,
    default_dataset: str | None = None,
    default_device: str | None = None,
) -> BenchmarkResult:
    """Map one VLM LLaVA ScienceQA Zoo `metrics.jsonl` entry to `BenchmarkResult`."""
    return _map_zoo_distillation_record(
        record=record,
        line_no=line_no,
        spec=_VLM_SPEC,
        default_model=default_model,
        default_dataset=default_dataset,
        default_device=default_device,
    )


def map_tinyllama_attention_record(
    record: dict[str, Any],
    line_no: int,
    default_model: str | None = None,
    default_dataset: str | None = None,
    default_device: str | None = None,
) -> BenchmarkResult:
    """Map one TinyLlama 1.1B Zoo `metrics.jsonl` entry to `BenchmarkResult`."""
    return _map_zoo_distillation_record(
        record=record,
        line_no=line_no,
        spec=_TINYLLAMA_SPEC,
        default_model=default_model,
        default_dataset=default_dataset,
        default_device=default_device,
    )


def map_llama_attention_record(
    record: dict[str, Any],
    line_no: int,
    default_model: str | None = None,
    default_dataset: str | None = None,
    default_device: str | None = None,
) -> BenchmarkResult:
    """Map one LLaMA Attention Zoo `metrics.jsonl` entry to `BenchmarkResult`."""
    return _map_zoo_distillation_record(
        record=record,
        line_no=line_no,
        spec=_LLAMA_ATTN_SPEC,
        default_model=default_model,
        default_dataset=default_dataset,
        default_device=default_device,
    )


def _read_jsonl_records(jsonl_path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read and strictly validate every non-empty line in a JSONL ledger."""
    if not jsonl_path.exists() or not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL ledger not found: {jsonl_path}")

    entries: list[tuple[int, dict[str, Any]]] = []
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line_no, raw_line in enumerate(fp, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {jsonl_path}:{line_no}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object in {jsonl_path}:{line_no}, got {type(parsed).__name__}.")
            entries.append((line_no, parsed))
    return entries


def _summarize_finite_metric(
    passed_results: list[BenchmarkResult],
    metric_key: str,
    *,
    best_mode: str | None = None,
) -> dict[str, Any] | None:
    """Compute min/max/mean over finite recorded values without fabricating defaults."""
    pairs: list[tuple[str, float]] = []
    for r in passed_results:
        raw = (r.metrics or {}).get(metric_key)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            mod_name = r.module.get("name", "") if isinstance(r.module, dict) else str(r.module)
            pairs.append((mod_name, val))
    if not pairs:
        return None
    vals = [v for _, v in pairs]
    summary: dict[str, Any] = {
        "count": len(vals),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "mean": round(sum(vals) / len(vals), 4),
    }
    if best_mode == "max":
        best_mod, best_val = max(pairs, key=lambda item: item[1])
        summary["best_module"] = best_mod
        summary["best_value"] = round(best_val, 4)
    elif best_mode == "min":
        best_mod, best_val = min(pairs, key=lambda item: item[1])
        summary["best_module"] = best_mod
        summary["best_value"] = round(best_val, 4)
    return summary


def _build_zoo_overview(
    results: list[BenchmarkResult],
    source_ledgers: dict[str, Path],
    out_path: Path,
    index_files: dict[str, str],
    json_files: list[str],
    markdown_files: list[str],
    verification_failures: int,
) -> dict[str, Any]:
    """Build a fail-closed cross-zoo overview grounded strictly in exported records."""
    grouped: dict[str, list[BenchmarkResult]] = {}
    for res in results:
        grouped.setdefault(res.benchmark_suite, []).append(res)

    failed_statuses = {
        BenchmarkStatus.FAILED.value,
        BenchmarkStatus.FAIL_NUMERICS.value,
        BenchmarkStatus.HARDWARE_ERROR.value,
        BenchmarkStatus.DEPENDENCY_ERROR.value,
        BenchmarkStatus.TIMEOUT.value,
    }

    zoos_section: dict[str, Any] = {}
    for suite, zoo_results in grouped.items():
        zoo_statuses = dict(Counter(r.status for r in zoo_results))
        passed_results = [r for r in zoo_results if r.status == BenchmarkStatus.SUCCESS.value]

        metrics_summary: dict[str, Any] = {}
        for key, mode in (
            ("accuracy", "max"),
            ("post_ppl", "min"),
            ("compression_ratio", "max"),
            ("tokens_per_sec", "max"),
        ):
            m_sum = _summarize_finite_metric(passed_results, key, best_mode=mode)
            if m_sum is not None:
                metrics_summary[key] = m_sum

        vram_vals = [
            float(r.resources["peak_vram_mb"])
            for r in zoo_results
            if isinstance(r.resources, dict)
            and r.resources.get("peak_vram_mb") is not None
            and not isinstance(r.resources.get("peak_vram_mb"), bool)
            and math.isfinite(float(r.resources["peak_vram_mb"]))
        ]
        dur_vals = [
            float(r.resources["duration_s"])
            for r in zoo_results
            if isinstance(r.resources, dict)
            and r.resources.get("duration_s") is not None
            and not isinstance(r.resources.get("duration_s"), bool)
            and math.isfinite(float(r.resources["duration_s"]))
        ]
        resources_summary: dict[str, Any] = {}
        if vram_vals:
            resources_summary["max_peak_vram_mb"] = round(max(vram_vals), 2)
        if dur_vals:
            resources_summary["total_duration_s"] = round(sum(dur_vals), 2)

        zoo_entry: dict[str, Any] = {
            "total_records": len(zoo_results),
            "passed_count": zoo_statuses.get(BenchmarkStatus.SUCCESS.value, 0),
            "failed_count": sum(zoo_statuses.get(st, 0) for st in failed_statuses),
            "skipped_count": zoo_statuses.get(BenchmarkStatus.SKIPPED.value, 0),
            "status_counts": zoo_statuses,
            "unique_modules": len(
                {r.module.get("name", "") for r in zoo_results if isinstance(r.module, dict) and r.module.get("name")}
            ),
            "models": sorted(
                {r.model.get("name", "") for r in zoo_results if isinstance(r.model, dict) and r.model.get("name")}
            ),
            "datasets": sorted(
                {r.dataset.get("name", "") for r in zoo_results if isinstance(r.dataset, dict) and r.dataset.get("name")}
            ),
            "metrics_summary": metrics_summary,
            "resources_summary": resources_summary,
        }
        if suite in source_ledgers:
            zoo_entry["source_ledger"] = str(source_ledgers[suite])
        zoos_section[suite] = zoo_entry

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_path),
        "total_records": len(results),
        "zoos_included": list(grouped.keys()),
        "status_counts": dict(Counter(r.status for r in results)),
        "verification_failures": verification_failures,
        "zoos": zoos_section,
        "index_files": index_files,
        "artifact_counts": {
            "json": len(json_files),
            "markdown": len(markdown_files),
        },
    }


def _write_json_atomic(target_path: Path, payload: Any) -> Path:
    """Write JSON atomically using a unique temporary file and rename."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(f".tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=str)
    tmp_path.replace(target_path)
    return target_path


def _write_text_atomic(target_path: Path, content: str) -> Path:
    """Write text atomically using a unique temporary file and rename."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(f".tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}")
    with tmp_path.open("w", encoding="utf-8") as fp:
        fp.write(content)
    tmp_path.replace(target_path)
    return target_path


def _generate_markdown_summary(
    results: list[BenchmarkResult],
    overview: dict[str, Any],
    out_path: Path,
) -> str:
    """Generate publication-grade multi-zoo Markdown comparison card."""
    lines: list[str] = [
        "# NeuralOps Multi-Zoo Benchmark Summary Card\n",
        f"- **Generated At**: `{overview.get('generated_at', '')}`",
        f"- **Output Directory**: `{out_path}`",
        f"- **Total Records**: {len(results)}",
        "- **Status Counts**: "
        + ", ".join(f"`{k}`: {v}" for k, v in sorted(overview.get("status_counts", {}).items())),
        f"- **Verification Failures**: {overview.get('verification_failures', 0)}",
        "",
        "## 1. Cross-Zoo Performance Matrix\n",
        "| Zoo Suite | Records | Passed | Failed | Skipped | Models | Datasets | Best Accuracy | Min Perplexity | Max Ratio | Max VRAM (MB) |",
        "| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for suite, zinfo in overview.get("zoos", {}).items():
        m_sum = zinfo.get("metrics_summary", {})
        r_sum = zinfo.get("resources_summary", {})

        best_acc_str = "N/A"
        if "accuracy" in m_sum:
            acc_val = m_sum["accuracy"].get("max")
            acc_mod = m_sum["accuracy"].get("best_module", "")
            best_acc_str = f"{acc_val:.2f}% (`{acc_mod}`)" if acc_val is not None else "N/A"

        min_ppl_str = "N/A"
        if "post_ppl" in m_sum:
            ppl_val = m_sum["post_ppl"].get("min")
            ppl_mod = m_sum["post_ppl"].get("best_module", "")
            min_ppl_str = f"{ppl_val:.2f} (`{ppl_mod}`)" if ppl_val is not None else "N/A"

        max_ratio_str = "N/A"
        if "compression_ratio" in m_sum:
            ratio_val = m_sum["compression_ratio"].get("max")
            ratio_mod = m_sum["compression_ratio"].get("best_module", "")
            max_ratio_str = f"{ratio_val:.2f}x (`{ratio_mod}`)" if ratio_val is not None else "N/A"

        vram_str = f"{r_sum.get('max_peak_vram_mb', 0.0):.1f}" if r_sum.get("max_peak_vram_mb") else "N/A"
        models_str = ", ".join(f"`{m}`" for m in zinfo.get("models", [])) or "unknown"
        datasets_str = ", ".join(f"`{d}`" for d in zinfo.get("datasets", [])) or "unknown"

        lines.append(
            f"| `{suite}` | {zinfo.get('total_records', 0)} | {zinfo.get('passed_count', 0)} | "
            f"{zinfo.get('failed_count', 0)} | {zinfo.get('skipped_count', 0)} | {models_str} | "
            f"{datasets_str} | {best_acc_str} | {min_ppl_str} | {max_ratio_str} | {vram_str} |"
        )

    lines.extend([
        "",
        "## 2. Top-Performing Modules by Zoo\n",
    ])

    for suite, zinfo in overview.get("zoos", {}).items():
        zoo_results = [r for r in results if r.benchmark_suite == suite]
        passed_runs = [r for r in zoo_results if r.status == BenchmarkStatus.SUCCESS.value]
        if not passed_runs:
            continue

        lines.append(f"### `{suite}`\n")
        lines.append(
            "| Rank / Module | Type | Model | Dataset | Pre Metric | Post Metric | Delta | Ratio | VRAM (MB) | Duration (s) |"
        )
        lines.append(
            "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        )

        def _sort_key(res: BenchmarkResult) -> tuple[float, float]:
            acc = (res.metrics or {}).get("accuracy")
            ppl = (res.metrics or {}).get("post_ppl")
            acc_val = (
                float(acc)
                if acc is not None and not isinstance(acc, bool) and math.isfinite(float(acc))
                else -1e9
            )
            ppl_val = (
                -float(ppl)
                if ppl is not None and not isinstance(ppl, bool) and math.isfinite(float(ppl))
                else -1e9
            )
            return (acc_val, ppl_val)

        sorted_runs = sorted(passed_runs, key=_sort_key, reverse=True)[:10]
        for res in sorted_runs:
            mod_name = res.module.get("name", "") if isinstance(res.module, dict) else str(res.module)
            mod_type = res.module.get("type", "") if isinstance(res.module, dict) else ""
            rank_val = res.config.get("rank") if isinstance(res.config, dict) else None
            label = f"`{mod_name}`" + (f" (r={rank_val})" if rank_val else "")

            mdl_name = res.model.get("name", "") if isinstance(res.model, dict) else ""
            ds_name = res.dataset.get("name", "") if isinstance(res.dataset, dict) else ""

            m = res.metrics or {}
            dm = res.delta_metrics or {}
            bm = res.baseline_metrics or {}
            r = res.resources or {}

            if "accuracy" in m:
                pre_m = (
                    f"{bm.get('accuracy', bm.get('pre_acc', 'N/A'))}%"
                    if "accuracy" in bm or "pre_acc" in bm
                    else "N/A"
                )
                post_m = f"{m['accuracy']:.2f}%" if isinstance(m["accuracy"], (int, float)) else str(m["accuracy"])
                delta_m = f"{dm.get('acc_delta', 'N/A')}"
                if isinstance(dm.get("acc_delta"), (int, float)):
                    delta_m = f"{dm['acc_delta']:+.2f}%"
            elif "post_ppl" in m:
                pre_m = f"{bm.get('ppl', 'N/A')}"
                post_m = f"{m['post_ppl']:.2f}" if isinstance(m["post_ppl"], (int, float)) else str(m["post_ppl"])
                delta_m = f"{dm.get('ppl_delta', 'N/A')}"
                if isinstance(dm.get("ppl_delta"), (int, float)):
                    delta_m = f"{dm['ppl_delta']:+.2f}"
            else:
                pre_m, post_m, delta_m = "N/A", "N/A", "N/A"

            ratio_str = (
                f"{m.get('compression_ratio'):.2f}x"
                if isinstance(m.get("compression_ratio"), (int, float))
                else "-"
            )
            vram_val = r.get("peak_vram_mb")
            vram_str = f"{vram_val:.1f}" if isinstance(vram_val, (int, float)) else "-"
            dur_val = r.get("duration_s")
            dur_str = f"{dur_val:.1f}s" if isinstance(dur_val, (int, float)) else "-"

            lines.append(
                f"| {label} | `{mod_type}` | `{mdl_name}` | `{ds_name}` | {pre_m} | {post_m} | {delta_m} | {ratio_str} | {vram_str} | {dur_str} |"
            )
        lines.append("")

    failed_skipped = [r for r in results if r.status != BenchmarkStatus.SUCCESS.value]
    lines.append("## 3. Verification & Anomaly Diagnostics\n")
    if not failed_skipped:
        lines.append("All audited records completed with verified finite metrics and `status=success`.\n")
    else:
        lines.append(f"Recorded **{len(failed_skipped)}** non-success execution(s):\n")
        lines.append("| Zoo Suite | Module | Status | Error Details |")
        lines.append("| :--- | :--- | :---: | :--- |")
        for r in failed_skipped[:25]:
            m_name = r.module.get("name", "") if isinstance(r.module, dict) else str(r.module)
            rk = r.config.get("rank") if isinstance(r.config, dict) else None
            m_label = f"`{m_name}`" + (f" (r={rk})" if rk else "")
            err_text = (r.error or "").replace("\n", " ")[:80]
            lines.append(f"| `{r.benchmark_suite}` | {m_label} | **`{r.status}`** | {err_text or 'None recorded'} |")
        if len(failed_skipped) > 25:
            lines.append(f"| ... | *({len(failed_skipped) - 25} more omitted)* | ... | ... |")
        lines.append("")

    return "\n".join(lines)


def export_dual_zoo_5formats(
    output_dir: Path | str,
    qwen_jsonl: Path | str | None = None,
    bert_jsonl: Path | str | None = None,
    qwen_model: str | None = None,
    qwen_dataset: str | None = None,
    bert_model: str = "bert-base-uncased",
    bert_dataset: str = "sst2",
    default_device: str | None = None,
    require_parquet: bool = True,
    overwrite: bool = False,
    strict: bool = False,
    bert_attn_jsonl: Path | str | None = None,
    qwen_comp_jsonl: Path | str | None = None,
    vlm_jsonl: Path | str | None = None,
    tinyllama_jsonl: Path | str | None = None,
    llama_attn_jsonl: Path | str | None = None,
    *,
    qwen_attn_ledger: Path | str | None = None,
    bert_comp_ledger: Path | str | None = None,
    bert_attn_ledger: Path | str | None = None,
    qwen_comp_ledger: Path | str | None = None,
    vlm_ledger: Path | str | None = None,
    tinyllama_ledger: Path | str | None = None,
    llama_attn_ledger: Path | str | None = None,
    llama_attention_ledger: Path | str | None = None,
) -> dict[str, Any]:
    """Read zoo JSONL ledgers, map to BenchmarkResult, and export all 5 formats + zoo_overview.json."""
    if qwen_jsonl is None and qwen_attn_ledger is not None:
        qwen_jsonl = qwen_attn_ledger
    if bert_jsonl is None and bert_comp_ledger is not None:
        bert_jsonl = bert_comp_ledger
    if bert_attn_jsonl is None and bert_attn_ledger is not None:
        bert_attn_jsonl = bert_attn_ledger
    if qwen_comp_jsonl is None and qwen_comp_ledger is not None:
        qwen_comp_jsonl = qwen_comp_ledger
    if vlm_jsonl is None and vlm_ledger is not None:
        vlm_jsonl = vlm_ledger
    if tinyllama_jsonl is None and tinyllama_ledger is not None:
        tinyllama_jsonl = tinyllama_ledger
    if llama_attn_jsonl is None and llama_attn_ledger is not None:
        llama_attn_jsonl = llama_attn_ledger
    if llama_attn_jsonl is None and llama_attention_ledger is not None:
        llama_attn_jsonl = llama_attention_ledger

    if all(
        p is None
        for p in (
            qwen_jsonl,
            bert_jsonl,
            bert_attn_jsonl,
            qwen_comp_jsonl,
            vlm_jsonl,
            tinyllama_jsonl,
            llama_attn_jsonl,
        )
    ):
        raise ValueError(
            "At least one zoo ledger flag ('--bert-attn-ledger', '--bert-comp-ledger', "
            "'--qwen-attn-ledger', '--qwen-comp-ledger', '--vlm-ledger', '--llama-attention-ledger', "
            "or '--tinyllama-ledger') must be provided."
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    existing_indices = [out_path / name for name in _INDEX_FILENAMES if (out_path / name).exists()]
    if existing_indices:
        if not overwrite:
            existing_str = ", ".join(str(p) for p in existing_indices)
            raise FileExistsError(
                f"Output index files already exist ({existing_str}). Use '--overwrite' or a fresh '--output-dir'."
            )
        for idx_path in existing_indices:
            idx_path.unlink()

    results: list[BenchmarkResult] = []
    source_ledgers: dict[str, Path] = {}
    zoo_counts: dict[str, int] = {}

    if qwen_jsonl is not None:
        qwen_path = Path(qwen_jsonl)
        source_ledgers["qwen_attention_zoo"] = qwen_path
        count = 0
        for line_no, rec in _read_jsonl_records(qwen_path):
            results.append(
                map_qwen_attention_record(
                    rec,
                    line_no=line_no,
                    default_model=qwen_model,
                    default_dataset=qwen_dataset,
                    default_device=default_device,
                )
            )
            count += 1
        zoo_counts["qwen_attention_zoo"] = count

    if bert_jsonl is not None:
        bert_path = Path(bert_jsonl)
        source_ledgers["bert_compressor_zoo"] = bert_path
        count = 0
        for line_no, rec in _read_jsonl_records(bert_path):
            results.append(
                map_bert_compressor_record(
                    rec,
                    line_no=line_no,
                    default_model=bert_model,
                    default_dataset=bert_dataset,
                    default_device=default_device,
                )
            )
            count += 1
        zoo_counts["bert_compressor_zoo"] = count

    optional_specs = (
        ("bert_attention_zoo", bert_attn_jsonl, map_bert_attention_record, None, None),
        ("qwen_compressor_zoo", qwen_comp_jsonl, map_qwen_compressor_record, qwen_model, qwen_dataset),
        ("vlm_llava_scienceqa_zoo", vlm_jsonl, map_vlm_attention_record, None, None),
        ("tinyllama_1.1b_zoo", tinyllama_jsonl, map_tinyllama_attention_record, None, None),
        ("llama_attention_zoo", llama_attn_jsonl, map_llama_attention_record, None, None),
    )
    for suite_name, ledger_arg, mapper_fn, d_model, d_dataset in optional_specs:
        if ledger_arg is None:
            continue
        ledger_path = Path(ledger_arg)
        source_ledgers[suite_name] = ledger_path
        count = 0
        for line_no, rec in _read_jsonl_records(ledger_path):
            results.append(
                mapper_fn(
                    rec,
                    line_no=line_no,
                    default_model=d_model,
                    default_dataset=d_dataset,
                    default_device=default_device,
                )
            )
            count += 1
        zoo_counts[suite_name] = count

    if not results:
        raise ValueError("Input JSONL ledger(s) contained zero records.")

    verification_failures = sum(
        1
        for r in results
        if r.status == BenchmarkStatus.FAIL_NUMERICS.value
        or (
            r.metadata.get("raw_status", "").strip().upper() in {"PASSED", "PASS", "SUCCESS", "OK"}
            and r.status != BenchmarkStatus.SUCCESS.value
        )
    )
    if strict and verification_failures > 0:
        raise RuntimeError(
            f"Strict verification failed: {verification_failures} record(s) had non-finite or missing PASSED metrics."
        )

    json_files: list[str] = []
    markdown_files: list[str] = []
    index_files: dict[str, str] = {}

    for res in results:
        saved = save_all_formats(res, output_dir=out_path, require_parquet=require_parquet)
        if "json" in saved:
            json_files.append(str(saved["json"]))
        if "markdown" in saved:
            markdown_files.append(str(saved["markdown"]))
        for key in ("jsonl", "csv", "parquet"):
            if key in saved:
                index_files[key] = str(saved[key])

    # 1. Complete structured export: results.json
    results_json_path = out_path / "results.json"
    results_payload = [r.to_dict() for r in results]
    _write_json_atomic(results_json_path, results_payload)
    index_files["json"] = str(results_json_path)

    # 2. Cross-zoo overview: zoo_overview.json
    overview_payload = _build_zoo_overview(
        results=results,
        source_ledgers=source_ledgers,
        out_path=out_path,
        index_files=index_files,
        json_files=json_files,
        markdown_files=markdown_files,
        verification_failures=verification_failures,
    )
    overview_path = _write_json_atomic(out_path / "zoo_overview.json", overview_payload)

    # 3. Publication-grade Markdown comparison card: summary.md
    summary_md_path = out_path / "summary.md"
    summary_md_content = _generate_markdown_summary(
        results=results,
        overview=overview_payload,
        out_path=out_path,
    )
    _write_text_atomic(summary_md_path, summary_md_content)
    index_files["summary_md"] = str(summary_md_path)

    status_counts = dict(Counter(r.status for r in results))
    return {
        "output_dir": str(out_path),
        "total_records": len(results),
        "qwen_records": zoo_counts.get("qwen_attention_zoo", 0),
        "bert_records": zoo_counts.get("bert_compressor_zoo", 0),
        "bert_attn_records": zoo_counts.get("bert_attention_zoo", 0),
        "qwen_comp_records": zoo_counts.get("qwen_compressor_zoo", 0),
        "vlm_records": zoo_counts.get("vlm_llava_scienceqa_zoo", 0),
        "tinyllama_records": zoo_counts.get("tinyllama_1.1b_zoo", 0),
        "llama_records": zoo_counts.get("llama_attention_zoo", 0),
        "zoo_counts": zoo_counts,
        "status_counts": status_counts,
        "verification_failures": verification_failures,
        "results_json": str(results_json_path),
        "summary_markdown": str(summary_md_path),
        "zoo_overview": str(overview_path),
        "index_files": index_files,
        "json_files": json_files,
        "markdown_files": markdown_files,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export up to 6 NeuralOps Zoo JSONL ledgers into 5 canonical formats and zoo_overview.json."
    )
    parser.add_argument(
        "--qwen-jsonl",
        "--qwen_jsonl",
        "--qwen-ledger",
        "--qwen_ledger",
        "--qwen-attn-ledger",
        "--qwen_attn_ledger",
        "--qwen-attn-jsonl",
        "--qwen_attn_jsonl",
        dest="qwen_jsonl",
        type=Path,
        default=None,
        help="Path to Qwen Attention Zoo metrics.jsonl (results/qwen_attention_zoo/metrics.jsonl)",
    )
    parser.add_argument(
        "--bert-jsonl",
        "--bert_jsonl",
        "--bert-ledger",
        "--bert_ledger",
        "--bert-comp-ledger",
        "--bert_comp_ledger",
        "--bert-comp-jsonl",
        "--bert_comp_jsonl",
        dest="bert_jsonl",
        type=Path,
        default=None,
        help="Path to BERT Compressor Zoo multirank_lora_ledger.jsonl (results/bert_compressor_zoo/multirank_lora_ledger.jsonl)",
    )
    parser.add_argument(
        "--bert-attn-ledger",
        "--bert_attn_ledger",
        "--bert-attn-jsonl",
        "--bert_attn_jsonl",
        dest="bert_attn_jsonl",
        type=Path,
        default=None,
        help="Path to BERT Attention Zoo metrics.jsonl (results/bert_attention_zoo/metrics.jsonl)",
    )
    parser.add_argument(
        "--qwen-comp-ledger",
        "--qwen_comp_ledger",
        "--qwen-comp-jsonl",
        "--qwen_comp_jsonl",
        dest="qwen_comp_jsonl",
        type=Path,
        default=None,
        help="Path to Qwen Compressor Zoo metrics.jsonl (results/qwen_compressor_zoo/metrics.jsonl)",
    )
    parser.add_argument(
        "--vlm-ledger",
        "--vlm_ledger",
        "--vlm-jsonl",
        "--vlm_jsonl",
        dest="vlm_jsonl",
        type=Path,
        default=None,
        help="Path to VLM LLaVA ScienceQA Zoo metrics.jsonl (results/vlm_llava_scienceqa_zoo/metrics.jsonl)",
    )
    parser.add_argument(
        "--tinyllama-ledger",
        "--tinyllama_ledger",
        "--tinyllama-jsonl",
        "--tinyllama_jsonl",
        dest="tinyllama_jsonl",
        type=Path,
        default=None,
        help="Path to TinyLlama 1.1B Zoo metrics.jsonl (results/tinyllama_1.1b_zoo/metrics.jsonl)",
    )
    parser.add_argument(
        "--llama-attention-ledger",
        "--llama_attention_ledger",
        "--llama-attn-ledger",
        "--llama_attn_ledger",
        "--llama-ledger",
        "--llama_ledger",
        "--llama-attention-jsonl",
        "--llama_attention_jsonl",
        "--llama-attn-jsonl",
        "--llama_attn_jsonl",
        "--llama-jsonl",
        "--llama_jsonl",
        dest="llama_attn_jsonl",
        type=Path,
        default=None,
        help="Path to LLaMA Attention Zoo metrics.jsonl (results/llama_attention_zoo/metrics.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=Path("results/dual_zoo_5formats"),
        help="Output directory for canonical 5-format artifacts.",
    )
    parser.add_argument(
        "--qwen-model",
        default=None,
        help="Fallback model name for Qwen records lacking a 'model' key.",
    )
    parser.add_argument(
        "--qwen-dataset",
        default=None,
        help="Fallback dataset name for Qwen records lacking a 'dataset' key.",
    )
    parser.add_argument(
        "--bert-model",
        default="bert-base-uncased",
        help="Model name for BERT compressor records lacking a 'model' key.",
    )
    parser.add_argument(
        "--bert-dataset",
        default="sst2",
        help="Dataset name for BERT compressor records lacking a 'dataset' key.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Fallback device type when omitted from source records (e.g. 'cuda').",
    )
    parser.add_argument(
        "--require-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail closed if Parquet export fails (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing rolling index files (results.jsonl/csv/parquet) and zoo_overview.json in output_dir.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any PASSED record has missing or non-finite metrics.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable export summary JSON to stdout only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        summary = export_dual_zoo_5formats(
            output_dir=args.output_dir,
            qwen_jsonl=args.qwen_jsonl,
            bert_jsonl=args.bert_jsonl,
            qwen_model=args.qwen_model,
            qwen_dataset=args.qwen_dataset,
            bert_model=args.bert_model,
            bert_dataset=args.bert_dataset,
            default_device=args.device,
            require_parquet=args.require_parquet,
            overwrite=args.overwrite,
            strict=args.strict,
            bert_attn_jsonl=args.bert_attn_jsonl,
            qwen_comp_jsonl=args.qwen_comp_jsonl,
            vlm_jsonl=args.vlm_jsonl,
            tinyllama_jsonl=args.tinyllama_jsonl,
            llama_attn_jsonl=args.llama_attn_jsonl,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        LOG.error("Export configuration/input error: %s", exc)
        return EXIT_USAGE_ERROR
    except RuntimeError as exc:
        msg = str(exc)
        if "Parquet export failed" in msg:
            LOG.error("Output export error: %s", msg)
            return EXIT_OUTPUT_ERROR
        LOG.error("Verification failure: %s", msg)
        return EXIT_NUMERICAL_FAILURE
    except OSError as exc:
        LOG.error("Output I/O error: %s", exc)
        return EXIT_OUTPUT_ERROR

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        LOG.info(
            "Exported %d records (qwen=%d, bert=%d, bert_attn=%d, qwen_comp=%d, vlm=%d, tinyllama=%d, llama=%d) "
            "to %s | statuses=%s | overview=%s",
            summary["total_records"],
            summary["qwen_records"],
            summary["bert_records"],
            summary["bert_attn_records"],
            summary["qwen_comp_records"],
            summary["vlm_records"],
            summary["tinyllama_records"],
            summary["llama_records"],
            summary["output_dir"],
            summary["status_counts"],
            summary["zoo_overview"],
        )
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
