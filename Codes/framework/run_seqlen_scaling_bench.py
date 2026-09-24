#!/usr/bin/env python3
"""
Sequence Length Scaling Benchmark Harness.

Measures:
  1. Throughput (tokens/sec)
  2. Latency (ms/token)
  3. Peak VRAM (MB)
across sequence lengths [128, 256, 512, 1024] at batch_size=1 on GPT-2.
Runs each measurement 3 times and reports the median.
Handles OOM gracefully.
Outputs:
  - results/seqlen_scaling/<MODULE>/summary.csv
  - results/SEQLEN_SCALING_MASTER.csv
"""

import os
import sys
import argparse
import time
import csv
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
CODES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

from framework.core.snapshot_restore import ModelStateSnapshot
from framework.eval_all_models_datasets import apply_module_to_model
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SeqLenScaling")


def measure_single_length(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    num_trials: int = 3,
    num_iters: int = 25,
    warmup_iters: int = 5,
) -> Dict[str, Any]:
    """
    Measures throughput, latency, and peak VRAM for a single sequence length.
    Runs `num_trials` times and returns the median results.
    """
    throughputs = []
    latencies = []
    peak_vrams = []
    forward_times = []

    for trial in range(num_trials):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        # Synthesize input_ids
        input_ids = torch.randint(
            low=10,
            high=50000,
            size=(batch_size, seq_len),
            dtype=torch.long,
            device=device,
        )

        # Warmup passes
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(input_ids)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize(device)
        start_event.record()

        with torch.no_grad():
            for _ in range(num_iters):
                _ = model(input_ids)

        end_event.record()
        torch.cuda.synchronize(device)

        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_sec = elapsed_ms / 1000.0

        total_tokens = batch_size * seq_len * num_iters
        throughput_tok_sec = total_tokens / elapsed_sec
        latency_ms_per_tok = elapsed_ms / total_tokens
        forward_time_ms = elapsed_ms / num_iters
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

        throughputs.append(throughput_tok_sec)
        latencies.append(latency_ms_per_tok)
        peak_vrams.append(peak_vram_mb)
        forward_times.append(forward_time_ms)

    med_throughput = float(np.median(throughputs))
    std_throughput = float(np.std(throughputs))
    med_latency = float(np.median(latencies))
    med_vram = float(np.median(peak_vrams))
    med_forward_time = float(np.median(forward_times))
    mean_forward_time = float(np.mean(forward_times))
    std_forward_time = float(np.std(forward_times))

    return {
        "status": "PASS",
        "throughput_tok_sec": round(med_throughput, 1),
        "throughput_std": round(std_throughput, 1),
        "latency_ms_per_token": round(med_latency, 4),
        "forward_time_ms": round(med_forward_time, 2),
        "forward_time_mean_ms": round(mean_forward_time, 2),
        "forward_time_std_ms": round(std_forward_time, 2),
        "peak_vram_mb": round(med_vram, 2),
    }


def run_benchmark_for_module(
    base_model: nn.Module,
    base_snapshot: ModelStateSnapshot,
    module_name: str,
    group_name: str,
    seq_lens: List[int],
    batch_size: int,
    device: torch.device,
    output_dir: str,
    master_csv: str,
    num_trials: int = 3,
    num_iters: int = 25,
    warmup_iters: int = 5,
) -> List[Dict[str, Any]]:
    """
    Evaluates one module across all specified sequence lengths.
    Restores baseline state before adapting.
    """
    logger.info("=" * 70)
    logger.info("Evaluating module '%s' (Group: %s) across seq_lens: %s", module_name, group_name, seq_lens)
    logger.info("=" * 70)

    # 1. Restore clean base model
    logger.info("Restoring clean base model snapshot...")
    base_snapshot.restore()
    base_model.eval()

    # 2. Apply module
    adapted_model = base_model
    base_dtype = next(base_model.parameters()).dtype
    if module_name.lower() not in ("full_attention", "dense", "baseline", "full"):
        try:
            logger.info("Injecting module '%s' into base model...", module_name)
            adapted_model, _ = apply_module_to_model(
                base_model,
                module_name=module_name,
                module_type="auto",
                rank=16,
                device=device,
            )
            # Ensure all newly injected parameters match device and precision
            adapted_model = adapted_model.to(device=device, dtype=base_dtype)
            adapted_model.eval()
        except Exception as e:
            logger.error("Failed to inject module '%s': %s", module_name, e)
            results = []
            for sl in seq_lens:
                results.append({
                    "seq_len": sl,
                    "throughput_tok_sec": 0.0,
                    "latency_ms_per_token": 0.0,
                    "peak_vram_mb": 0.0,
                    "status": f"INJECT_FAIL: {type(e).__name__}",
                })
            save_results(module_name, group_name, results, output_dir, master_csv)
            return results

    results = []
    for sl in seq_lens:
        logger.info("--- Testing '%s' at seq_len=%d, batch_size=%d (3 trials) ---", module_name, sl, batch_size)
        try:
            metrics = measure_single_length(
                model=adapted_model,
                seq_len=sl,
                batch_size=batch_size,
                device=device,
                num_trials=num_trials,
                num_iters=num_iters,
                warmup_iters=warmup_iters,
            )
            row = {
                "seq_len": sl,
                "throughput_tok_sec": metrics["throughput_tok_sec"],
                "throughput_std": metrics["throughput_std"],
                "latency_ms_per_token": metrics["latency_ms_per_token"],
                "forward_time_ms": metrics["forward_time_ms"],
                "forward_time_mean_ms": metrics["forward_time_mean_ms"],
                "forward_time_std_ms": metrics["forward_time_std_ms"],
                "peak_vram_mb": metrics["peak_vram_mb"],
                "status": metrics["status"],
            }
            logger.info("Result: seq_len=%d -> Forward: %.2f ± %.2f ms (med: %.2f ms), Throughput: %.1f ± %.1f tok/s, Latency: %.4f ms/tok, Peak VRAM: %.2f MB [%s]",
                        sl, row["forward_time_mean_ms"], row["forward_time_std_ms"], row["forward_time_ms"],
                        row["throughput_tok_sec"], row["throughput_std"], row["latency_ms_per_token"], row["peak_vram_mb"], row["status"])
            results.append(row)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_msg = str(e).lower()
            if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in err_msg:
                logger.warning("CUDA Out of Memory for module '%s' at seq_len=%d: %s", module_name, sl, e)
                torch.cuda.empty_cache()
                row = {
                    "seq_len": sl,
                    "throughput_tok_sec": 0.0,
                    "latency_ms_per_token": 0.0,
                    "forward_time_ms": 0.0,
                    "peak_vram_mb": 0.0,
                    "status": "OOM",
                }
                results.append(row)
            else:
                logger.error("Runtime error for module '%s' at seq_len=%d: %s", module_name, sl, e)
                row = {
                    "seq_len": sl,
                    "throughput_tok_sec": 0.0,
                    "latency_ms_per_token": 0.0,
                    "forward_time_ms": 0.0,
                    "peak_vram_mb": 0.0,
                    "status": f"FAIL: {type(e).__name__}",
                }
                results.append(row)

    save_results(module_name, group_name, results, output_dir, master_csv)
    return results


def save_results(
    module_name: str,
    group_name: str,
    results: List[Dict[str, Any]],
    output_dir: str,
    master_csv: str,
):
    """
    Saves individual module summary.csv and appends to master CSV.
    """
    mod_dir = os.path.join(output_dir, module_name)
    os.makedirs(mod_dir, exist_ok=True)
    summary_path = os.path.join(mod_dir, "summary.csv")

    # 1. Write module summary
    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "forward_time_ms", "forward_time_mean_ms", "forward_time_std_ms", "throughput_tok_sec", "throughput_std", "latency_ms_per_token", "peak_vram_mb", "status"])
        for r in results:
            writer.writerow([r["seq_len"], r.get("forward_time_ms", 0.0), r.get("forward_time_mean_ms", 0.0), r.get("forward_time_std_ms", 0.0), r["throughput_tok_sec"], r.get("throughput_std", 0.0), r["latency_ms_per_token"], r["peak_vram_mb"], r["status"]])
    logger.info("Saved module summary to: %s", summary_path)

    # 2. Update master CSV (replace old rows for module if present)
    os.makedirs(os.path.dirname(os.path.abspath(master_csv)), exist_ok=True)
    existing_rows = []
    if os.path.exists(master_csv) and os.path.getsize(master_csv) > 0:
        with open(master_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if row and row[0] != module_name:
                        existing_rows.append(row)

    with open(master_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["module", "group", "seq_len", "forward_time_ms", "forward_time_mean_ms", "forward_time_std_ms", "throughput_tok_sec", "throughput_std", "latency_ms_per_token", "peak_vram_mb", "status"])
        for r in existing_rows:
            writer.writerow(r)
        for r in results:
            writer.writerow([module_name, group_name, r["seq_len"], r.get("forward_time_ms", 0.0), r.get("forward_time_mean_ms", 0.0), r.get("forward_time_std_ms", 0.0), r["throughput_tok_sec"], r.get("throughput_std", 0.0), r["latency_ms_per_token"], r["peak_vram_mb"], r["status"]])
    logger.info("Updated master CSV with %d rows: %s", len(results), master_csv)


def main():
    parser = argparse.ArgumentParser(description="Sequence Length Scaling Benchmark")
    parser.add_argument("--modules", type=str, required=True, help="Comma-separated module names to test")
    parser.add_argument("--group", type=str, default="auto", help="Module group (e.g. Group A, Group B, Group C)")
    parser.add_argument("--model", type=str, default="models/gpt2", help="Path or HF ID of base model")
    parser.add_argument("--seq_lens", type=str, default="128,256,512,1024", help="Comma-separated sequence lengths")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (strict 1 for fair comparison)")
    parser.add_argument("--num_trials", type=int, default=3, help="Number of trials for median computation")
    parser.add_argument("--num_iters", type=int, default=25, help="Number of timed forward passes per trial")
    parser.add_argument("--warmup_iters", type=int, default=5, help="Number of warmup forward passes")
    parser.add_argument("--output_dir", type=str, default="results/seqlen_scaling", help="Output directory")
    parser.add_argument("--master_csv", type=str, default="results/SEQLEN_SCALING_MASTER.csv", help="Master CSV path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    seq_lens = [int(s.strip()) for s in args.seq_lens.split(",") if s.strip()]
    module_list = [m.strip() for m in args.modules.split(",") if m.strip()]

    # Load base model once
    logger.info("Loading base model '%s' in fp16...", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    logger.info("Taking base model snapshot for in-memory module swapping...")
    base_snapshot = ModelStateSnapshot(model)

    # Process modules
    for module_name in module_list:
        run_benchmark_for_module(
            base_model=model,
            base_snapshot=base_snapshot,
            module_name=module_name,
            group_name=args.group,
            seq_lens=seq_lens,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
            master_csv=args.master_csv,
            num_trials=args.num_trials,
            num_iters=args.num_iters,
            warmup_iters=args.warmup_iters,
        )

    logger.info("All requested modules completed successfully.")


if __name__ == "__main__":
    main()
