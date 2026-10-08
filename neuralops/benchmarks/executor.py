"""Single run executor for the neuralops benchmark suite."""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from typing import Any, Optional

import torch

from neuralops.benchmarks.adapters.classification import ClassificationAdapter
from neuralops.benchmarks.adapters.perplexity import PerplexityAdapter
from neuralops.benchmarks.adapters.ruler import RulerAdapter
from neuralops.benchmarks.adapters.zeroshot import SUPPORTED_ZEROSHOT_DATASETS, ZeroShotAdapter
from neuralops.benchmarks.discovery import discover_modules
from neuralops.benchmarks.result import BenchmarkResult

logger = logging.getLogger(__name__)

PERPLEXITY_DATASETS = {"wikitext2", "wikitext103", "ptb", "c4"}


def _select_adapter(
    module_name: str,
    module_type: str,
    model: str,
    dataset: str,
    rank: int | None,
    device: str,
    dtype: str,
    batch_size: int,
    max_length: int,
    epochs: int,
    lr: float,
    seed: int,
    eval_only: bool,
    **kwargs: Any,
) -> Any:
    """Route benchmark to appropriate task adapter based on dataset task type."""
    ds_lower = dataset.lower()
    if ds_lower in PERPLEXITY_DATASETS:
        return PerplexityAdapter(
            module_name=module_name,
            module_type=module_type,
            model_name=model,
            dataset_name=dataset,
            rank=rank,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_length=max_length,
            seed=seed,
            **kwargs,
        )
    elif "ruler" in ds_lower or "niah" in ds_lower:
        return RulerAdapter(
            module_name=module_name,
            module_type=module_type,
            model_name=model,
            dataset_name=dataset,
            rank=rank,
            device=device,
            dtype=dtype,
            context_length=max_length if max_length > 512 else 4096,
            seed=seed,
            **kwargs,
        )
    elif ds_lower in SUPPORTED_ZEROSHOT_DATASETS or any(k in ds_lower for k in ("arc", "hellaswag", "mmlu", "gsm8k")):
        return ZeroShotAdapter(
            module_name=module_name,
            module_type=module_type,
            model_name=model,
            dataset_name=dataset,
            rank=rank,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_length=max_length,
            seed=seed,
            **kwargs,
        )
    else:
        # Default task is sequence classification (GLUE tasks)
        return ClassificationAdapter(
            module_name=module_name,
            module_type=module_type,
            model_name=model,
            dataset_name=dataset,
            rank=rank,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_length=max_length,
            epochs=epochs,
            lr=lr,
            seed=seed,
            eval_only=eval_only,
            **kwargs,
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

    Produces a canonical BenchmarkResult strictly conforming to schema.
    Captures errors gracefully without raising or terminating execution.
    """
    run_id = str(uuid.uuid4())
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_t = time.perf_counter()

    config_dict = {
        "rank": rank,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
        "eval_only": eval_only,
        **kwargs,
    }

    is_lm_task = dataset.lower() in PERPLEXITY_DATASETS
    if eval_only:
        if not is_lm_task:
            logger.warning(
                "eval_only=True with a freshly-initialized classification head "
                "will produce random accuracy. Set eval_only=False to fine-tune."
            )
        if module and str(module).lower() not in ("none", "baseline", "full", "dense"):
            logger.warning("Compression applied but no recovery training.")

    # 1. Inspect module metadata from discovery (or handle baseline)
    is_baseline = module is None or str(module).lower() in ("none", "baseline", "full", "dense")

    if is_baseline:
        module_name = "baseline"
        module_type = "baseline"
        mod_dict = {
            "name": "baseline",
            "type": "baseline",
            "family": "dense",
            "supports_causal": True,
            "supports_gqa": True,
            "supports_rope": True,
            "requires_cuda": False,
            "rank": rank,
        }
        effective_timeout = timeout_s or 1800
    else:
        module_name = str(module)
        all_modules = discover_modules()
        mod_info = all_modules.get(module_name)

        if mod_info is None:
            duration_s = time.perf_counter() - start_t
            end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return BenchmarkResult(
                run_id=run_id,
                run_name=f"{module}_{model}_{dataset}",
                timestamp_start=start_time_iso,
                timestamp_end=end_time_iso,
                module={"name": module, "type": "unknown", "rank": rank},
                model={"name": model},
                dataset={"name": dataset},
                config=config_dict,
                status="failed",
                error=f"ModuleNotFoundError: Module '{module}' not found in ATTENTION_REGISTRY or COMPRESSOR_REGISTRY.",
                traceback="",
                timeout_s=timeout_s or 1800,
                timed_out=False,
            )

        module_type = mod_info.type
        mod_dict = {
            "name": module_name,
            "type": module_type,
            "family": mod_info.family,
            "supports_causal": mod_info.supports_causal,
            "supports_gqa": mod_info.supports_gqa,
            "supports_rope": mod_info.supports_rope,
            "requires_cuda": mod_info.requires_cuda,
            "rank": rank,
        }
        effective_timeout = timeout_s if timeout_s is not None else mod_info.timeout_s


    # Check CUDA requirement
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if mod_dict.get("requires_cuda", False) and not torch.cuda.is_available() and target_device == "cpu":

        duration_s = time.perf_counter() - start_t
        end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return BenchmarkResult(
            run_id=run_id,
            run_name=f"{module}_{model}_{dataset}",
            timestamp_start=start_time_iso,
            timestamp_end=end_time_iso,
            module=mod_dict,
            model={"name": model},
            dataset={"name": dataset},
            config=config_dict,
            status="skipped",
            error=f"HardwareIncompatible: Module '{module}' requires CUDA which is not available.",
            timeout_s=effective_timeout,
            timed_out=False,
        )

    # 2. Select and execute Adapter
    adapter = None
    try:
        adapter = _select_adapter(
            module_name=module_name,
            module_type=module_type,
            model=model,
            dataset=dataset,
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

        model_meta = run_output.get("model_metadata")
        dataset_meta = run_output.get("dataset_metadata")
        eval_metrics = run_output.get("eval_metrics", {})
        latency_info = run_output.get("latency", {})
        peak_mem = run_output.get("peak_memory_mb", 0.0)

        resources_dict = {
            "peak_vram_mb": peak_mem,
            "latency_ms_per_token": latency_info.get("latency_ms_per_token", 0.0),
            "throughput_tok_per_sec": latency_info.get("throughput_tok_per_sec", 0.0),
            "duration_s": round(duration_s, 2),
            "num_params_after": getattr(model_meta, "num_params", 0) if model_meta else 0,
        }

        return BenchmarkResult(
            run_id=run_id,
            run_name=f"{module}_{model}_{dataset}",
            timestamp_start=start_time_iso,
            timestamp_end=end_time_iso,
            module=mod_dict,
            model=model_meta.__dict__ if model_meta else {"name": model},
            dataset=dataset_meta.__dict__ if dataset_meta else {"name": dataset},
            config=config_dict,
            metrics=eval_metrics,
            resources=resources_dict,
            status="success",
            error=None,
            traceback=None,
            timeout_s=effective_timeout,
            timed_out=False,
        )

    except Exception as e:
        duration_s = time.perf_counter() - start_t
        end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        err_tb = traceback.format_exc()
        logger.error("Execution failed for module %s: %s\n%s", module, e, err_tb)

        return BenchmarkResult(
            run_id=run_id,
            run_name=f"{module}_{model}_{dataset}",
            timestamp_start=start_time_iso,
            timestamp_end=end_time_iso,
            module=mod_dict,
            model={"name": model},
            dataset={"name": dataset},
            config=config_dict,
            status="failed",
            error=f"{type(e).__name__}: {str(e)}",
            traceback=err_tb,
            timeout_s=effective_timeout,
            timed_out=False,
        )
    finally:
        if adapter is not None:
            adapter.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["execute_single"]
