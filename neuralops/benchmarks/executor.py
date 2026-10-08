"""Single run executor for the neuralops benchmark suite."""

from __future__ import annotations

import logging
import os
import platform
import socket
import sys
import time
import traceback
import uuid
from typing import Any, Dict, Optional

import torch

from neuralops.benchmarks.adapters.classification import ClassificationAdapter
from neuralops.benchmarks.adapters.perplexity import PerplexityAdapter
from neuralops.benchmarks.discovery import discover_modules
from neuralops.benchmarks.result import (
    BenchmarkResult,
    DatasetMetadata,
    EnvironmentInfo,
    ModelMetadata,
    ModuleMetadata,
)

logger = logging.getLogger(__name__)

PERPLEXITY_DATASETS = {"wikitext2", "wikitext103", "ptb", "c4"}


def get_environment_info(device: Optional[str] = None) -> EnvironmentInfo:
    """Capture runtime environment hardware and package versions."""
    cuda_available = torch.cuda.is_available()
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None
    cuda_version = torch.version.cuda if cuda_available else None
    gpu_memory_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if cuda_available
        else None
    )

    return EnvironmentInfo(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=cuda_device_name,
        gpu_count=torch.cuda.device_count() if cuda_available else 0,
        gpu_memory_gb=gpu_memory_gb,
    )


def execute_single(
    module: str,
    model: str = "distilbert-base-uncased",
    dataset: str = "sst2",
    rank: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = "float32",
    batch_size: int = 32,
    max_length: int = 128,
    epochs: int = 3,
    lr: float = 2e-5,
    seed: int = 42,
    eval_only: bool = False,
    timeout_s: Optional[int] = 1800,
    **kwargs: Any,
) -> BenchmarkResult:
    """Execute a benchmark for a single module on a model and dataset.

    Handles discovery, adapter selection, execution, timing, and error trapping
    without aborting the parent process.
    """
    run_id = str(uuid.uuid4())
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_t = time.perf_counter()

    env_info = get_environment_info(device)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Inspect module metadata from discovery
    all_modules = discover_modules()
    mod_info = all_modules.get(module)

    if mod_info is None:
        duration_s = time.perf_counter() - start_t
        end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return BenchmarkResult(
            run_id=run_id,
            status="failed",
            start_time=start_time_iso,
            end_time=end_time_iso,
            duration_s=duration_s,
            module=ModuleMetadata(name=module, type="unknown", rank=rank),
            model=ModelMetadata(name=model, family="unknown"),
            dataset=DatasetMetadata(name=dataset, task_type="unknown"),
            environment=env_info,
            error_type="ModuleNotFoundError",
            error_message=f"Module '{module}' not found in ATTENTION_REGISTRY or COMPRESSOR_REGISTRY.",
            error_traceback="",
            timeout_s=timeout_s,
            timed_out=False,
        )

    module_type = mod_info.type
    module_meta = ModuleMetadata(
        name=module,
        type=module_type,
        rank=rank,
        supports_causal=mod_info.supports_causal,
        supports_gqa=mod_info.supports_gqa,
        supports_rope=mod_info.supports_rope,
        requires_cuda=mod_info.requires_cuda,
        has_custom_injector=mod_info.has_custom_injector,
        target_architecture=mod_info.target_architecture,
    )

    # Check CUDA requirement
    if mod_info.requires_cuda and not torch.cuda.is_available() and target_device == "cpu":
        duration_s = time.perf_counter() - start_t
        end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return BenchmarkResult(
            run_id=run_id,
            status="skipped",
            start_time=start_time_iso,
            end_time=end_time_iso,
            duration_s=duration_s,
            module=module_meta,
            model=ModelMetadata(name=model, family="unknown"),
            dataset=DatasetMetadata(name=dataset, task_type="unknown"),
            environment=env_info,
            error_type="HardwareIncompatible",
            error_message=f"Module '{module}' requires CUDA which is not available in current environment.",
            timeout_s=timeout_s,
            timed_out=False,
        )

    # 2. Select Adapter
    is_lm_task = dataset.lower() in PERPLEXITY_DATASETS
    adapter = None
    try:
        if is_lm_task:
            adapter = PerplexityAdapter(
                module_name=module,
                module_type=module_type,
                model_name=model,
                dataset_name=dataset,
                rank=rank,
                device=target_device,
                dtype=dtype,
                batch_size=batch_size,
                max_length=max_length,
                seed=seed,
                **kwargs,
            )
        else:
            adapter = ClassificationAdapter(
                module_name=module,
                module_type=module_type,
                model_name=model,
                dataset_name=dataset,
                rank=rank,
                device=target_device,
                dtype=dtype,
                batch_size=batch_size,
                max_length=max_length,
                epochs=epochs,
                lr=lr,
                seed=seed,
                eval_only=eval_only,
                **kwargs,
            )

        run_output = adapter.run()
        duration_s = time.perf_counter() - start_t
        end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return BenchmarkResult(
            run_id=run_id,
            status="completed",
            start_time=start_time_iso,
            end_time=end_time_iso,
            duration_s=duration_s,
            module=module_meta,
            model=run_output.get("model_metadata") or ModelMetadata(name=model, family="unknown"),
            dataset=run_output.get("dataset_metadata") or DatasetMetadata(name=dataset, task_type="unknown"),
            environment=env_info,
            train_metrics=run_output.get("train_metrics", {}),
            eval_metrics=run_output.get("eval_metrics", {}),
            latency=run_output.get("latency", {}),
            peak_memory_mb=run_output.get("peak_memory_mb"),
            timeout_s=timeout_s,
            timed_out=False,
        )

    except Exception as e:
        duration_s = time.perf_counter() - start_t
        end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        err_tb = traceback.format_exc()
        logger.error(f"Execution failed for module {module}: {e}\n{err_tb}")

        return BenchmarkResult(
            run_id=run_id,
            status="failed",
            start_time=start_time_iso,
            end_time=end_time_iso,
            duration_s=duration_s,
            module=module_meta,
            model=ModelMetadata(name=model, family="unknown"),
            dataset=DatasetMetadata(name=dataset, task_type="unknown"),
            environment=env_info,
            error_type=type(e).__name__,
            error_message=str(e),
            error_traceback=err_tb,
            timeout_s=timeout_s,
            timed_out=False,
        )
    finally:
        if adapter is not None:
            adapter.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
