"""
Optimized Universal GPU Benchmark Launcher for NeuralOps Framework.
Features:
- In-memory dataset validation-only caching (avoids 67k train row CPU mapping).
- Base model template caching with clean in-memory deepcopy cloning.
- Non-blocking CUDA event timing (zero CPU host synchronization stalls).
- Multi-worker concurrent execution support (--workers N).
- Automatic generation of GPU_BENCHMARK_REPORT.md and benchmark_summary.csv upon sweep completion.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict
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
from framework.eval_all_models_datasets import evaluate_single_combination_opt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GPUServerLauncher")


def _eval_worker_task(task_args: Dict[str, Any]) -> BenchmarkResult:
    """Executes a single combination with a hard timeout wrapper."""
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
            batch_size=task_args.get("batch_size", 64),
            seq_len=task_args.get("seq_len", 256),
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


def write_final_report_and_csv(results: List[BenchmarkResult], output_dir: str, elapsed_time: float) -> str:
    """Generates the official GPU_BENCHMARK_REPORT.md and benchmark_summary.csv."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "GPU_BENCHMARK_REPORT.md")
    csv_path = os.path.join(output_dir, "benchmark_summary.csv")

    total_evals = len(results)
    pass_evals = sum(1 for r in results if r.status == "PASS")
    timeout_evals = sum(1 for r in results if r.status == "TIMEOUT")
    fail_evals = sum(1 for r in results if r.status == "FAIL")

    # 1. Write CSV
    if results:
        fieldnames = list(asdict(results[0]).keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    # 2. Write Markdown Report
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    report_md = f"""# GPU Server Benchmark Execution Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Hardware Platform:** {gpu_name}  
**Total Wall-Clock Time:** {elapsed_time:.2f} seconds ({elapsed_time/60.0:.2f} minutes)  
**Total Combinations Evaluated:** {total_evals} / 720  

### Summary Statistics
- **PASS:** {pass_evals} ({pass_evals/max(1, total_evals)*100:.1f}%)
- **TIMEOUT:** {timeout_evals}
- **FAIL:** {fail_evals}

---

## Benchmark Results Table

| Track | Module | Type | Model | Dataset | Status | Fwd (ms) | Bwd (ms) | Total (ms) | Peak VRAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results:
        fwd_str = f"{r.forward_latency_ms:.2f}" if r.status == "PASS" else "-"
        bwd_str = f"{r.backward_latency_ms:.2f}" if r.status == "PASS" else "-"
        tot_str = f"{r.total_step_latency_ms:.2f}" if r.status == "PASS" else "-"
        mem_str = f"{r.peak_memory_mb:.1f}" if r.status == "PASS" else "-"
        report_md += f"| `{r.track}` | `{r.module_name}` | {r.module_type} | {r.model_family.upper()} | {r.dataset.upper()} | **{r.status}** | {fwd_str} | {bwd_str} | {tot_str} | {mem_str} |\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("Successfully written final benchmark report to %s", report_path)
    return report_path


def run_gpu_benchmark_sweep_opt(
    models: List[str],
    datasets: List[str],
    precision: str = "bf16",
    batch_size: int = 64,
    seq_len: int = 256,
    max_steps: int = 5,
    timeout: int = 120,
    workers: int = 1,
    tracks: Optional[List[str]] = None,
    output_dir: str = "./results/gpu_server",
) -> List[BenchmarkResult]:
    """
    Executes benchmark sweep using multi-worker execution and cached tensors.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info("Initializing Optimized Benchmark on %s (Batch: %d, Seq: %d, Workers: %d)...", device, batch_size, seq_len, workers)

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
        for idx, task in enumerate(task_list, 1):
            res = _eval_worker_task(task)
            results.append(res)
            logger.info("[%s] %s | %s | %s -> %s (fwd: %.2fms, bwd: %.2fms, mem: %.1fMB)",
                        res.track, res.module_name, res.model_family, res.dataset, res.status, res.forward_latency_ms, res.backward_latency_ms, res.peak_memory_mb)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_worker_task, t): t for t in task_list}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                results.append(res)
                logger.info("[%s] %s | %s | %s -> %s (fwd: %.2fms, bwd: %.2fms, mem: %.1fMB)",
                            res.track, res.module_name, res.model_family, res.dataset, res.status, res.forward_latency_ms, res.backward_latency_ms, res.peak_memory_mb)

    elapsed = time.time() - start_time
    total_evals = len(results)
    passed_evals = sum(1 for r in results if r.status == "PASS")
    logger.info("Sweep Finished: %d/%d PASS in %.2f seconds (%.3f s/combo).", passed_evals, total_evals, elapsed, elapsed / max(1, total_evals))

    write_final_report_and_csv(results, output_dir, elapsed)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized NeuralOps GPU Benchmark Launcher")
    parser.add_argument("--mode", type=str, default="comprehensive", help="Execution mode (comprehensive, fast, diagnostics)")
    parser.add_argument("--models", type=str, default="bert,roberta,t5")
    parser.add_argument("--datasets", type=str, default="sst2,mrpc")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads")
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
