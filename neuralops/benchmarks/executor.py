#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Single run executor for the neuralops benchmark suite

Implements executor functionality for NeuralOps.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

# Python 3.14 standard library introduces the 'compression' package. Ensure the internal
# package directory '.../neuralops' and '.../tests' are never on sys.path directly (only repo root),
# so they do not shadow Python's standard library 'compression' needed by gzip / torch.profiler.
sys.path = [
    p
    for p in sys.path
    if not p.rstrip("/\\").endswith("neuralops") and not p.rstrip("/\\").endswith("tests")
]

import torch

from neuralops.benchmarks.discovery import discover_modules
from neuralops.benchmarks.result import BenchmarkResult
from neuralops.benchmarks.run_logging import setup_run_logger

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
    task: str | None = None,
    **kwargs: Any,
) -> Any:
    """Route benchmark to appropriate task adapter based on task or dataset task type."""
    if task:
        from neuralops.benchmarks.tasks.adapter import select_universal_task_adapter

        adapter_cls = select_universal_task_adapter(task)
        return adapter_cls(
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
            task=task,
            **kwargs,
        )

    ds_lower = dataset.lower()
    if ds_lower in PERPLEXITY_DATASETS:
        from neuralops.benchmarks.adapters.perplexity import PerplexityAdapter

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
        from neuralops.benchmarks.adapters.ruler import RulerAdapter

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
    elif any(k in ds_lower for k in ("arc", "hellaswag", "mmlu", "gsm8k", "truthfulqa", "winogrande")):
        from neuralops.benchmarks.adapters.zeroshot import ZeroShotAdapter

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
        from neuralops.benchmarks.adapters.classification import ClassificationAdapter

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


route_task_adapter = _select_adapter


def _record_event(
    output_dir: Path | str,
    run_id: str,
    module: str,
    status: str,
    error: str = "",
    traceback_str: str = "",
    exitcode: Optional[int] = None,
) -> None:
    """Append structured execution event to events.jsonl."""
    events_path = Path(output_dir) / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": run_id,
        "module": module,
        "status": status,
        "exitcode": exitcode,
        "error": error,
        "traceback": traceback_str,
    }
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event_data) + "\n")


def _isolated_worker_fn(pipe: Any, adapter_kwargs: dict[str, Any]) -> None:
    """Dedicated worker process entrypoint executing benchmark adapter."""
    # Ensure tests/ directory does not shadow standard library modules (e.g. Python 3.14 compression)
    import sys

    sys.path = [
        p
        for p in sys.path
        if not p.rstrip("/").endswith("tests") and not p.rstrip("/").endswith("neuralops")
    ]
    try:
        test_behavior = adapter_kwargs.pop(
            "_test_behavior", os.environ.get("_NEURALOPS_BENCHMARK_TEST_BEHAVIOR")
        )
        if test_behavior == "crash":
            os._exit(134)
        elif test_behavior == "sleep":
            time.sleep(15.0)
            pipe.send(("success", {"eval_metrics": {}}))
            return
        elif test_behavior == "fail":
            raise RuntimeError("Deliberate internal kernel test error")
        elif test_behavior == "success":
            pipe.send(("success", {"eval_metrics": {"accuracy": 0.95}}))
            return

        tracker_type = adapter_kwargs.pop("tracker_type", "none")
        project_name = adapter_kwargs.pop("project_name", "neuralops")
        tracker_instance = None
        if tracker_type and tracker_type != "none":
            try:
                from neuralops.benchmarks.loggers import create_tracker

                tracker_instance = create_tracker(
                    tracker_type=tracker_type,
                    project_name=project_name,
                    run_name=f"{adapter_kwargs.get('module_name')}_{adapter_kwargs.get('model')}_{adapter_kwargs.get('dataset')}",
                    output_dir=adapter_kwargs.get("output_dir", "results"),
                    config=adapter_kwargs,
                )
            except Exception:
                pass

        adapter = _select_adapter(tracker_instance=tracker_instance, **adapter_kwargs)
        output = adapter.run()
        pipe.send(("success", output))
    except BaseException as exc:
        tb = "".join(traceback.format_exception(exc))
        pipe.send(("error", (str(exc), tb)))


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
    task: Optional[str] = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Execute a benchmark for a single module on a model and dataset.

    Produces a canonical BenchmarkResult strictly conforming to schema.
    Captures errors gracefully without raising or terminating execution.
    """
    run_id = str(uuid.uuid4())
    start_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_t = time.perf_counter()

    out_dir = kwargs.get("output_dir", "results")
    run_logger, log_file, jsonl_file = setup_run_logger(out_dir, run_id, str(module))
    run_logger.info(
        "Beginning execution of run %s: module=%s, model=%s, dataset=%s, rank=%s, epochs=%s, batch_size=%s, task=%s",
        run_id,
        module,
        model,
        dataset,
        rank,
        epochs,
        batch_size,
        task,
    )

    config_dict = {
        "rank": rank,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "max_length": max_length,
        "seed": seed,
        "eval_only": eval_only,
        "task": task,
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

    if not is_baseline:
        all_modules = discover_modules()
        if module_name not in all_modules:
            duration_s = time.perf_counter() - start_t
            end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            err_msg = f"ModuleNotFoundError: Module '{module}' not found in registry."
            run_logger.error(err_msg)
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
                error=err_msg,
                traceback="",
                timeout_s=timeout_s or 1800,
                timed_out=False,
            )

    dist_backend = (kwargs.get("distributed_backend") or kwargs.get("distributed") or "none").lower()
    if dist_backend in ("accelerate", "ddp", "fsdp", "deepspeed"):
        try:
            import accelerate

            run_logger.info(
                "Distributed acceleration backend '%s' detected (accelerate %s).",
                dist_backend,
                getattr(accelerate, "__version__", ""),
            )
        except ImportError:
            run_logger.warning(
                "Distributed backend '%s' requested but 'accelerate' package is not installed. Falling back to single-device execution.",
                dist_backend,
            )
            kwargs["distributed_backend"] = "none"

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if mod_dict.get("requires_cuda", False) and not torch.cuda.is_available() and target_device == "cpu":
        duration_s = time.perf_counter() - start_t
        end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        skip_msg = f"HardwareIncompatible: Module '{module}' requires CUDA which is not available."
        run_logger.warning(skip_msg)
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
            error=skip_msg,
            timeout_s=effective_timeout,
            timed_out=False,
        )

    if target_device.startswith("cuda") and torch.cuda.is_available():
        try:
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mb = free_bytes / (1024 * 1024)
            if free_mb < 500:
                run_logger.warning(
                    "Low VRAM warning before run %s: only %.1f MB free out of %.1f MB",
                    run_id,
                    free_mb,
                    total_bytes / (1024 * 1024),
                )
        except Exception as e:
            run_logger.debug("Could not query GPU memory: %s", e)

    adapter = None
    tracker_instance = None
    tracker_type = kwargs.get("tracker", "none")
    if tracker_type and tracker_type != "none":
        try:
            from neuralops.benchmarks.loggers import create_tracker

            tracker_instance = create_tracker(
                tracker_type=tracker_type,
                project_name=kwargs.get("project_name", "neuralops"),
                run_name=f"{module}_{model}_{dataset}_{run_id[:8]}",
                output_dir=out_dir,
                config=config_dict,
            )
            tracker_instance.log_params(config_dict)
        except Exception as tracker_init_err:
            run_logger.debug("MLOps tracker initialization failed: %s", tracker_init_err)

    try:
        mp_ctx = mp.get_context("spawn")
        parent_conn, child_conn = mp_ctx.Pipe(duplex=False)

        extra_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in (
                "output_dir",
                "project_name",
                "module_name",
                "module_type",
                "model",
                "dataset",
                "rank",
                "device",
                "dtype",
                "batch_size",
                "max_length",
                "epochs",
                "lr",
                "seed",
                "eval_only",
                "task",
                "tracker_type",
            )
        }
        worker_kwargs = dict(
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
            task=task,
            tracker_type=tracker_type,
            project_name=kwargs.get("project_name", "neuralops"),
            output_dir=out_dir,
            **extra_kwargs,
        )

        worker_proc = mp_ctx.Process(
            target=_isolated_worker_fn,
            args=(child_conn, worker_kwargs),
            name=f"benchmark-worker-{run_id[:8]}",
        )
        worker_proc.start()
        child_conn.close()

        worker_proc.join(timeout=effective_timeout)

        # 1. Timeout Check
        if worker_proc.is_alive():
            worker_proc.terminate()
            worker_proc.join(timeout=1.0)
            if worker_proc.is_alive():
                worker_proc.kill()
                worker_proc.join(timeout=1.0)

            duration_s = time.perf_counter() - start_t
            end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            timeout_msg = (
                f"TimeoutError: Execution timed out after {effective_timeout}s for module '{module}' "
                f"on {model}/{dataset}."
            )
            run_logger.error(timeout_msg)
            _record_event(
                output_dir=out_dir,
                run_id=run_id,
                module=module_name,
                status="timeout",
                error=timeout_msg,
                traceback_str=f"TimeoutError: Execution exceeded {effective_timeout} seconds limit.",
                exitcode=-15,
            )
            return BenchmarkResult(
                run_id=run_id,
                run_name=f"{module}_{model}_{dataset}",
                timestamp_start=start_time_iso,
                timestamp_end=end_time_iso,
                module=mod_dict,
                model={"name": model},
                dataset={"name": dataset},
                config=config_dict,
                status="timeout",
                error=timeout_msg,
                traceback=f"TimeoutError: Execution exceeded {effective_timeout} seconds limit.",
                timeout_s=effective_timeout,
                timed_out=True,
            )

        # 2. Crash / Non-zero exit check
        if worker_proc.exitcode != 0:
            duration_s = time.perf_counter() - start_t
            end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            exitcode = worker_proc.exitcode
            is_hardware_or_signal = (exitcode < 0) or (exitcode in (134, 139, 135, 136, 137))
            status_val = "hardware_error" if is_hardware_or_signal else "failed"

            err_msg = f"Process terminated abnormally with exit code {exitcode} ({status_val})"
            tb_str = ""
            if parent_conn.poll(0.1):
                try:
                    msg_type, payload = parent_conn.recv()
                    if msg_type == "error":
                        err_msg, tb_str = payload
                except Exception:
                    pass

            run_logger.error("Run %s failed with %s: %s", run_id, status_val, err_msg)
            _record_event(
                output_dir=out_dir,
                run_id=run_id,
                module=module_name,
                status=status_val,
                error=err_msg,
                traceback_str=tb_str,
                exitcode=exitcode,
            )
            return BenchmarkResult(
                run_id=run_id,
                run_name=f"{module}_{model}_{dataset}",
                timestamp_start=start_time_iso,
                timestamp_end=end_time_iso,
                module=mod_dict,
                model={"name": model},
                dataset={"name": dataset},
                config=config_dict,
                status=status_val,
                error=err_msg,
                traceback=tb_str,
                timeout_s=effective_timeout,
                timed_out=False,
            )

        # 3. Clean exit (exitcode == 0)
        try:
            msg_type, payload = parent_conn.recv()
        except EOFError:
            msg_type, payload = "error", ("Worker process closed without sending output", "")

        if msg_type == "error":
            duration_s = time.perf_counter() - start_t
            end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            err_msg, tb_str = payload
            status_val = (
                "DATASET_ERROR"
                if any(t in err_msg.lower() for t in ("dataset", "tokenizer", "wikitext", "glue"))
                else "failed"
            )
            _record_event(
                output_dir=out_dir,
                run_id=run_id,
                module=module_name,
                status=status_val,
                error=err_msg,
                traceback_str=tb_str,
                exitcode=0,
            )
            return BenchmarkResult(
                run_id=run_id,
                run_name=f"{module}_{model}_{dataset}",
                timestamp_start=start_time_iso,
                timestamp_end=end_time_iso,
                module=mod_dict,
                model={"name": model},
                dataset={"name": dataset},
                config=config_dict,
                status=status_val,
                error=err_msg,
                traceback=tb_str,
                timeout_s=effective_timeout,
                timed_out=False,
            )

        run_output = payload
        _record_event(
            output_dir=out_dir,
            run_id=run_id,
            module=module_name,
            status="success",
            error="",
            traceback_str="",
            exitcode=0,
        )

        duration_s = time.perf_counter() - start_t
        end_time_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        model_meta = run_output.get("model_metadata")
        dataset_meta = run_output.get("dataset_metadata")
        eval_metrics = run_output.get("eval_metrics", {})
        latency_info = run_output.get("latency", {})
        peak_mem = run_output.get("peak_memory_mb", 0.0)
        train_duration_s = run_output.get("train_duration_s", 0.0)

        resources_dict = {
            "peak_vram_mb": peak_mem,
            "latency_ms_per_token": latency_info.get("latency_ms_per_token", 0.0),
            "throughput_tok_per_sec": latency_info.get("throughput_tok_per_sec", 0.0),
            "duration_s": round(duration_s, 2),
            "training_duration_s": train_duration_s,
            "num_params_after": getattr(model_meta, "num_params", 0) if model_meta else 0,
        }

        run_logger.info(
            "Run %s completed successfully in %.2fs (training: %.2fs). Metrics: %s",
            run_id,
            duration_s,
            train_duration_s,
            eval_metrics,
        )

        if tracker_instance is not None:
            try:
                log_payload = {**eval_metrics, **resources_dict}
                tracker_instance.log_metrics(
                    {k: float(v) for k, v in log_payload.items() if isinstance(v, (int, float))}
                )
                tracker_instance.finish()
            except Exception as tracker_err:
                run_logger.debug("MLOps tracker logging failed: %s", tracker_err)

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
        run_logger.error("Execution failed for module %s: %s\n%s", module, e, err_tb)

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
            try:
                adapter.cleanup()
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def execute_single_eval_bridge(
    module: str,
    module_type: str = "attention",
    model: str = "gpt2",
    dataset: str = "wikitext",
    rank: int | None = 64,
    device: str = "cuda",
    dtype: str = "float16",
    batch_size: int = 16,
    max_length: int = 512,
    **kwargs: Any,
) -> Any:
    """Execute single benchmark run directly via the unified BenchmarkEvaluator bridge."""
    from neuralops.benchmarks.eval_bridge import BenchmarkEvaluator

    evaluator = BenchmarkEvaluator(
        model=model,
        attention_type=module if module_type == "attention" else None,
        compression_type=module if module_type == "compressor" else None,
        rank=rank,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        seq_len=max_length,
    )
    return evaluator.run_evaluation(dataset=dataset, **kwargs)


__all__ = ["execute_single", "execute_single_eval_bridge"]
