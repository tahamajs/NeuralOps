#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Canonical 5-format benchmark ledger exporter for the Dual Zoo pipelines.

Reads:
  1. Qwen Attention Zoo JSONL (`metrics.jsonl`)
  2. BERT Compressor Zoo JSONL (`multirank_lora_ledger.jsonl`)

Maps every verified entry into canonical `BenchmarkResult` records with exact
recorded hardware, metrics, deltas, timestamps, and fail-closed statuses, then
exports all 5 canonical formats via `save_all_formats` (JSON, JSONL, CSV,
Markdown, and Parquet).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import math
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

_INDEX_FILENAMES = ("results.jsonl", "results.csv", "results.parquet")


def _validate_safe_segment(value: Any, field_name: str, *, allow_slashes: bool = False) -> str:
    """Validate that a path-building identifier is non-empty and free of traversal."""
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Missing required field '{field_name}'.")
    if "\x00" in text or ".." in Path(text).parts:
        raise ValueError(f"Unsafe path segment in field '{field_name}': {text!r}")
    if not allow_slashes and ("/" in text or "\\" in text or Path(text).is_absolute()):
        raise ValueError(f"Unsafe path separator in field '{field_name}': {text!r}")
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
    """Map one BERT Compressor Zoo `multirank_lora_ledger.jsonl` entry to `BenchmarkResult`."""
    comp_key = _validate_safe_segment(record.get("compressor"), "compressor")
    model_name = _validate_safe_segment(record.get("model") or default_model, "model", allow_slashes=True)
    dataset_name = _validate_safe_segment(record.get("dataset") or default_dataset, "dataset")

    numeric_errors: list[str] = []
    status, warnings = _normalize_zoo_status(record.get("status"))

    rank = _extract_int(record, "rank", numeric_errors)
    if rank is None or rank < 0:
        raise ValueError(f"Missing or invalid 'rank' in BERT compressor record at line {line_no}.")

    ratio = _extract_finite_float(record, "ratio", numeric_errors)
    orig_params = _extract_int(record, "orig_params", numeric_errors)
    comp_params = _extract_int(record, "comp_params", numeric_errors)
    zero_shot_acc = _extract_finite_float(record, "zero_shot_acc", numeric_errors)
    zero_shot_loss = _extract_finite_float(record, "zero_shot_loss", numeric_errors)
    delta_zero = _extract_finite_float(record, "delta_zero", numeric_errors)
    healed_acc = _extract_finite_float(record, "healed_acc", numeric_errors)
    healed_loss = _extract_finite_float(record, "healed_loss", numeric_errors)
    delta_healed = _extract_finite_float(record, "delta_healed", numeric_errors)
    duration_s = _extract_finite_float(record, "duration_s", numeric_errors)
    if duration_s is None:
        duration_s = _extract_finite_float(record, "eval_time", numeric_errors)
    if duration_s is None:
        duration_s = _extract_finite_float(record, "train_secs", numeric_errors)

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

    # Populate canonical accuracy/loss fields consumed by saver.py indices.
    if lora_healed and healed_acc is not None:
        metrics["accuracy"] = healed_acc
        if healed_loss is not None:
            metrics["loss"] = healed_loss
    elif healed_acc is not None:
        metrics["accuracy"] = healed_acc
        if zero_shot_loss is not None:
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

    delta_metrics: dict[str, Any] = {}
    if delta_zero is not None:
        delta_metrics["delta_zero"] = delta_zero
    if delta_healed is not None:
        delta_metrics["delta_healed"] = delta_healed

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
        if "accuracy" not in metrics:
            required_error = "Missing required accuracy metric ('zero_shot_acc'/'healed_acc') for PASSED record."
        elif ratio is None or ratio <= 0.0:
            required_error = "Missing or non-positive 'ratio' metric for PASSED compressor record."

    final_status, error_msg = _finalize_status_and_errors(status, numeric_errors, record, required_error)

    config: dict[str, Any] = {"zoo": "bert_compressor_zoo", "rank": rank}
    if lora_healed is not None:
        config["lora_healed"] = lora_healed

    model_dict: dict[str, Any] = {"name": model_name, "family": "bert"}
    if orig_params is not None:
        model_dict["num_params"] = orig_params
    if comp_params is not None:
        model_dict["num_params_after"] = comp_params

    ts_token = str(record.get("timestamp") or record.get("timestamp_end") or f"line_{line_no}")
    identity_seed = f"{comp_key}|{model_name}|{dataset_name}|rank{rank}|{ts_token}|{line_no}"
    run_meta, meta_extra = _build_run_metadata(
        record=record,
        zoo_name="bert_compressor_zoo",
        run_name=f"bert_compressor_{comp_key}_r{rank}",
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
            "rank": rank,
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
) -> dict[str, Any]:
    """Read dual-zoo JSONL ledgers, map to BenchmarkResult, and export all 5 formats."""
    if qwen_jsonl is None and bert_jsonl is None:
        raise ValueError("At least one of '--qwen-jsonl' or '--bert-jsonl' must be provided.")

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
    qwen_count = 0
    bert_count = 0

    if qwen_jsonl is not None:
        for line_no, rec in _read_jsonl_records(Path(qwen_jsonl)):
            results.append(
                map_qwen_attention_record(
                    rec,
                    line_no=line_no,
                    default_model=qwen_model,
                    default_dataset=qwen_dataset,
                    default_device=default_device,
                )
            )
            qwen_count += 1

    if bert_jsonl is not None:
        for line_no, rec in _read_jsonl_records(Path(bert_jsonl)):
            results.append(
                map_bert_compressor_record(
                    rec,
                    line_no=line_no,
                    default_model=bert_model,
                    default_dataset=bert_dataset,
                    default_device=default_device,
                )
            )
            bert_count += 1

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

    status_counts = dict(Counter(r.status for r in results))
    return {
        "output_dir": str(out_path),
        "total_records": len(results),
        "qwen_records": qwen_count,
        "bert_records": bert_count,
        "status_counts": status_counts,
        "verification_failures": verification_failures,
        "index_files": index_files,
        "json_files": json_files,
        "markdown_files": markdown_files,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Qwen Attention Zoo and BERT Compressor Zoo JSONL ledgers into 5 canonical formats."
    )
    parser.add_argument(
        "--qwen-jsonl",
        "--qwen_jsonl",
        "--qwen-ledger",
        "--qwen_ledger",
        dest="qwen_jsonl",
        type=Path,
        default=None,
        help="Path to Qwen Attention Zoo metrics.jsonl",
    )
    parser.add_argument(
        "--bert-jsonl",
        "--bert_jsonl",
        "--bert-ledger",
        "--bert_ledger",
        dest="bert_jsonl",
        type=Path,
        default=None,
        help="Path to BERT Compressor Zoo multirank_lora_ledger.jsonl",
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
        help="Replace existing rolling index files (results.jsonl/csv/parquet) in output_dir.",
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
            "Exported %d records (qwen=%d, bert=%d) to %s | statuses=%s",
            summary["total_records"],
            summary["qwen_records"],
            summary["bert_records"],
            summary["output_dir"],
            summary["status_counts"],
        )
    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
