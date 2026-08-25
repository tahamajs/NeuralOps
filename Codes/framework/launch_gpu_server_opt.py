"""
Optimized GPU Server Universal Execution & Benchmark Engine (Multi-Worker Enabled).
Features:
- Multi-Worker Parallel Queue (--workers N) overlapping CPU setup with GPU kernel execution.
- In-memory dataset and base model caching.
- Hard per-module timeout wrapper (120s).
- Isolated execution without mutating baseline framework state.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Ensure framework is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

from framework.eval_all_models_datasets import (
    BenchmarkResult,
    TRACK_MODULES,
)
from framework.eval_all_models_datasets_opt import evaluate_single_combination_opt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GPULauncherOpt")


def _eval_worker_task(task_args: Dict[str, Any]) -> BenchmarkResult:
    """Executes a single combination with a hard timeout."""
    timeout = task_args.get("timeout", 120)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            evaluate_single_combination_opt,
            track=task_args["track"],
            module_name=task_args["module_name"],
            module_type=task_args["module_type"],
            model_family=task_args["model_family"],
            dataset=task_args["dataset"],
            rank=task_args.get("rank", 16),
            num_steps=task_args.get("max_steps", 5),
            batch_size=task_args.get("batch_size", 32),
            seq_len=task_args.get("seq_len", 128),
            device=task_args["device"],
            results_dir=task_args.get("output_dir", "./results/gpu_server"),
            allow_synthetic=task_args.get("allow_synthetic", False),
        )
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.error("TIMEOUT: %s on %s/%s exceeded %ds.", task_args["module_name"], task_args["model_family"], task_args["dataset"], timeout)
            return BenchmarkResult(
                track=task_args["track"],
                module_name=task_args["module_name"],
                module_type=task_args["module_type"],
                model_family=task_args["model_family"],
                dataset=task_args["dataset"],
                status="TIMEOUT",
            )


def run_gpu_benchmark_sweep_opt(
    models: List[str],
    datasets: List[str],
    precision: str = "bf16",
    batch_size: int = 32,
    seq_len: int = 128,
    max_steps: int = 5,
    timeout: int = 120,
    workers: int = 1,
    tracks: Optional[List[str]] = None,
    output_dir: str = "./results/gpu_server",
    log_name: str = "GPU_BENCHMARK_REPORT_OPT.md",
) -> List[BenchmarkResult]:
    """
    Executes benchmark sweep using multi-worker execution and cached tensors.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info("Initializing Optimized Benchmark on %s with %d worker(s)...", device, workers)

    active_tracks = tracks or list(TRACK_MODULES.keys())
    task_list: List[Dict[str, Any]] = []

    for t_key in active_tracks:
        t_data = TRACK_MODULES.get(t_key, {})
        attns = t_data.get("attentions", [])
        comps = t_data.get("compressors", [])

        for attn in attns:
            for m in models:
                for d in datasets:
                    task_list.append({
                        "track": t_key,
                        "module_name": attn,
                        "module_type": "attention",
                        "model_family": m,
                        "dataset": d,
                        "max_steps": max_steps,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "timeout": timeout,
                        "device": device,
                        "output_dir": output_dir,
                    })

        for comp in comps:
            for m in models:
                for d in datasets:
                    task_list.append({
                        "track": t_key,
                        "module_name": comp,
                        "module_type": "compressor",
                        "model_family": m,
                        "dataset": d,
                        "max_steps": max_steps,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "timeout": timeout,
                        "device": device,
                        "output_dir": output_dir,
                    })

    results: List[BenchmarkResult] = []
    start_time = time.time()

    if workers <= 1:
        for task in task_list:
            res = _eval_worker_task(task)
            results.append(res)
            logger.info("[%s] %s | %s | %s -> %s (fwd: %.2fms, bwd: %.2fms)",
                        res.track, res.module_name, res.model_family, res.dataset, res.status, res.forward_latency_ms, res.backward_latency_ms)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_worker_task, t): t for t in task_list}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                results.append(res)
                logger.info("[%s] %s | %s | %s -> %s (fwd: %.2fms, bwd: %.2fms)",
                            res.track, res.module_name, res.model_family, res.dataset, res.status, res.forward_latency_ms, res.backward_latency_ms)

    elapsed = time.time() - start_time
    total_evals = len(results)
    passed_evals = sum(1 for r in results if r.status == "PASS")
    logger.info("Sweep Finished: %d/%d PASS in %.2f seconds (%.3f s/combo).", passed_evals, total_evals, elapsed, elapsed / max(1, total_evals))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized NeuralOps GPU Benchmark Launcher")
    parser.add_argument("--models", type=str, default="bert,roberta,t5")
    parser.add_argument("--datasets", type=str, default="sst2,mrpc")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent worker threads")
    parser.add_argument("--tracks", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/gpu_server")
    args = parser.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    track_list = [t.strip().lower() for t in args.tracks.split(",")] if args.tracks else None

    run_gpu_benchmark_sweep_opt(
        models=models,
        datasets=datasets,
        precision=args.precision,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        timeout=args.timeout,
        workers=args.workers,
        tracks=track_list,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
