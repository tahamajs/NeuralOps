#!/usr/bin/env python3
"""
Sequence-Length Scaling Runner: Evaluates VRAM, latency, and throughput scaling across context lengths.

Evaluates scaling across sequence lengths: [128, 512, 1024, 2048, 4096]
Modules: ["dense", "a3", "kmeans", "svd-mamba", "primal"]

Measures:
  - Peak VRAM footprint via torch.cuda.max_memory_allocated()
  - Forward pass latency (ms) and throughput (tokens/sec)
  - Graceful OOM handling: marks "OOM" in results without crashing sweep

Usage:
    python run_seqlen_scaling.py --dry_run
    python run_seqlen_scaling.py --batch_size 2
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
sys.path[:0] = [str(CODES), str(ROOT), str(ROOT / "Codes")]

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
from framework.eval_all_models_datasets import apply_module_to_model

LOG = logging.getLogger("SeqlenScaling")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SEQLENS = [128, 512, 1024, 2048, 4096]
MODULES = ["dense", "a3", "kmeans", "svd-mamba", "primal"]

CSV_FIELDS = [
    "module",
    "seq_len",
    "status",
    "peak_vram_mb",
    "latency_ms",
    "throughput_tok_per_sec",
    "error",
    "timestamp",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sequence length computational complexity scaling evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=str, default="models/distilbert_sst2",
                        help="Base model path")
    parser.add_argument("--output_dir", type=str, default="results/seqlen_scaling",
                        help="Directory to save scaling outputs")
    parser.add_argument("--seq_lens", type=int, nargs="*", default=SEQLENS,
                        help="Sequence lengths to evaluate")
    parser.add_argument("--modules", type=str, nargs="*", default=MODULES,
                        help="Modules to evaluate across sequence lengths")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (small batch prevents early OOM)")
    parser.add_argument("--warmup_steps", type=int, default=3,
                        help="Warmup forward passes before timing")
    parser.add_argument("--eval_steps", type=int, default=10,
                        help="Timed forward passes for benchmark averaging")
    parser.add_argument("--rank", type=int, default=64,
                        help="Compression rank")
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Print scaling matrix without running")
    return parser.parse_args()


def benchmark_forward_scaling(model, batch_size: int, seq_len: int, warmup: int = 3,
                              steps: int = 10, device: str = "cuda"):
    """
    Executes warmup and timed forward passes with synthetic sequence inputs.
    Returns peak VRAM (MB), latency (ms), and throughput (tok/s).
    """
    model.eval()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

    # Timing
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    latency_ms = (elapsed / steps) * 1000.0
    total_tokens = batch_size * seq_len * steps
    throughput = total_tokens / elapsed

    if device == "cuda" and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_vram_mb = 0.0

    return {
        "status": "PASS",
        "peak_vram_mb": round(peak_vram_mb, 2),
        "latency_ms": round(latency_ms, 2),
        "throughput_tok_per_sec": round(throughput, 1),
        "error": "",
    }


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("SEQUENCE-LENGTH SCALING BENCHMARK SUITE")
    print(f"Sequence Lengths: {args.seq_lens}")
    print(f"Modules:          {args.modules}")
    print(f"Batch Size:       {args.batch_size}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Planned sequence length scaling matrix:")
        for mod in args.modules:
            for L in args.seq_lens:
                print(f"  Module: {mod:<15} | SeqLen: {L:4d} | Out: {args.output_dir}/{mod}/L{L}.json")
        print("\n[DRY RUN COMPLETE] No hardware runs executed.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    all_rows = []

    for mod in args.modules:
        mod_dir = out_dir / mod
        mod_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("=== Benchmarking Module Scaling: %s ===", mod)

        for L in args.seq_lens:
            res_file = mod_dir / f"L{L}.json"
            LOG.info("  SeqLen: %d ...", L)

            try:
                model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
                if device == "cuda":
                    model.half()
                model.to(device)

                if mod != "dense":
                    typ = "attention" if mod in ATTENTION_REGISTRY else "compressor"
                    model, _ = apply_module_to_model(model, mod, typ, rank=args.rank, device=device)

                metrics = benchmark_forward_scaling(
                    model, batch_size=args.batch_size, seq_len=L,
                    warmup=args.warmup_steps, steps=args.eval_steps, device=device
                )

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as oom:
                LOG.warning("CUDA OOM for module %s at seq_len %d", mod, L)
                metrics = {
                    "status": "OOM",
                    "peak_vram_mb": 0.0,
                    "latency_ms": 0.0,
                    "throughput_tok_per_sec": 0.0,
                    "error": "CUDA Out Of Memory",
                }
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                LOG.exception("Error evaluating %s at seq_len %d: %s", mod, L, e)
                metrics = {
                    "status": "FAIL",
                    "peak_vram_mb": 0.0,
                    "latency_ms": 0.0,
                    "throughput_tok_per_sec": 0.0,
                    "error": str(e),
                }

            row = {
                "module": mod,
                "seq_len": L,
                "status": metrics["status"],
                "peak_vram_mb": metrics["peak_vram_mb"],
                "latency_ms": metrics["latency_ms"],
                "throughput_tok_per_sec": metrics["throughput_tok_per_sec"],
                "error": metrics["error"],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            all_rows.append(row)
            res_file.write_text(json.dumps(row, indent=2), encoding="utf-8")

            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for r in all_rows:
                    writer.writerow(r)

    print(f"\n[DONE] Sequence length scaling benchmark completed. Results: {summary_csv}")


if __name__ == "__main__":
    main()
