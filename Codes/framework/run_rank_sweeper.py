#!/usr/bin/env python3
"""
Automated rank sweep runner for low-rank compressor modules.
Runs accuracy sweeps across ranks [16, 32, 64, 128, 256, 384],
extracts classification metrics (NO PPL), and writes rank_curve.csv.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_RANKS = [16, 32, 64, 128, 256, 384]


def parse_args():
    parser = argparse.ArgumentParser(description="Run rank sweep for compressor modules.")
    parser.add_argument("--modules", required=True, help="Comma-separated list of modules to sweep")
    parser.add_argument("--ranks", default="16,32,64,128,256,384", help="Comma-separated ranks")
    parser.add_argument("--model", default="models/distilbert_sst2", help="Model path")
    parser.add_argument("--model_family", default="bert", help="Model family")
    parser.add_argument("--dataset", default="sst2", help="Dataset name")
    parser.add_argument("--recovery_steps", type=int, default=50, help="Recovery steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_base", default="results/rank_sweep_sst2", help="Base output directory")
    parser.add_argument("--parent_checkpoint", default="results/bert_sst2_baseline/finetuned_model.pt", help="Baseline checkpoint")
    parser.add_argument("--cached_batches_dir", default="results/bert_sst2_baseline/batch_cache_sst2", help="Batch cache dir")
    return parser.parse_args()


def read_summary_row(summary_csv_path: Path):
    if not summary_csv_path.exists():
        return None
    try:
        with open(summary_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row
    except Exception as e:
        print(f"Error reading {summary_csv_path}: {e}")
    return None


def run_single_rank(module: str, rank: int, args) -> dict:
    rank_dir = Path(args.output_base) / module / f"rank_{rank}"
    summary_csv = rank_dir / "summary.csv"

    # If already successfully evaluated, return existing data
    if summary_csv.exists():
        row = read_summary_row(summary_csv)
        if row and row.get("status") in ("PASS", "FAIL", "FAIL_BELOW_CHANCE"):
            print(f"[{module} | rank {rank}] Found existing valid summary. Skipping run.")
            return row

    rank_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "Codes/framework/run_quality_sweep.py",
        "--modules", module,
        "--model", args.model,
        "--model_family", args.model_family,
        "--dataset", args.dataset,
        "--rank", str(rank),
        "--recovery_steps", str(args.recovery_steps),
        "--learning_rate", str(args.learning_rate),
        "--seed", str(args.seed),
        "--output_dir", str(rank_dir) + "/",
    ]
    if Path(args.parent_checkpoint).exists():
        cmd.extend(["--parent_checkpoint", args.parent_checkpoint])
    if Path(args.cached_batches_dir).exists():
        cmd.extend(["--cached_batches_dir", args.cached_batches_dir])

    print(f"\n========================================================")
    print(f"Executing: {' '.join(cmd)}")
    print(f"========================================================")

    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"Command failed with code {res.returncode}")

    row = read_summary_row(summary_csv)
    if row:
        return row

    return {
        "module_name": module,
        "status": f"ERROR_CODE_{res.returncode}",
        "accuracy_pct": 0.0,
        "f1_score": 0.0,
        "val_loss": 99.0,
        "peak_vram_mb": 0.0,
        "compression_rate": 1.0,
        "num_compressed_layers": 0,
        "num_fallback_layers": 0,
        "elapsed_s": elapsed,
    }


def write_rank_curve(module: str, rows: list, output_base: str):
    module_dir = Path(output_base) / module
    module_dir.mkdir(parents=True, exist_ok=True)
    curve_csv = module_dir / "rank_curve.csv"

    fields = [
        "rank", "status", "accuracy_pct", "f1_score", "val_loss",
        "peak_vram_mb", "compression_rate", "compression_ratio_pct",
        "num_compressed_layers", "num_fallback_layers", "elapsed_s"
    ]

    with open(curve_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "rank": r.get("rank"),
                "status": r.get("status", "N/A"),
                "accuracy_pct": r.get("accuracy_pct", 0.0),
                "f1_score": r.get("f1_score", 0.0),
                "val_loss": r.get("val_loss", 0.0),
                "peak_vram_mb": r.get("peak_vram_mb", 0.0),
                "compression_rate": r.get("compression_rate", 1.0),
                "compression_ratio_pct": r.get("compression_ratio_pct", 0.0),
                "num_compressed_layers": r.get("num_compressed_layers", 0),
                "num_fallback_layers": r.get("num_fallback_layers", 0),
                "elapsed_s": r.get("elapsed_s", 0.0),
            })
    print(f"\nSaved rank curve to: {curve_csv}")


def print_rank_table(module: str, rows: list):
    print(f"\n### Rank Sweep Results: `{module}` (DistilBERT SST-2)")
    print("| Rank | Status | Accuracy (%) | F1 Score | Val Loss | Peak VRAM (MB) | Compression Rate | Layers (Comp/Fall) | Elapsed (s) |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in rows:
        rank = r.get("rank")
        status = r.get("status", "N/A")
        acc = float(r.get("accuracy_pct", 0.0))
        f1 = float(r.get("f1_score", 0.0))
        loss = float(r.get("val_loss", 0.0)) if r.get("val_loss") else 0.0
        vram = float(r.get("peak_vram_mb", 0.0)) if r.get("peak_vram_mb") else 0.0
        comp = float(r.get("compression_rate", 1.0)) if r.get("compression_rate") else 1.0
        n_comp = r.get("num_compressed_layers", "0")
        n_fall = r.get("num_fallback_layers", "0")
        elapsed = float(r.get("elapsed_s", 0.0)) if r.get("elapsed_s") else 0.0
        print(f"| {rank} | {status} | {acc:.2f}% | {f1:.4f} | {loss:.4f} | {vram:.1f} | {comp:.2f}x | {n_comp}/{n_fall} | {elapsed:.1f}s |")


def main():
    args = parse_args()
    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]

    for mod in modules:
        print(f"\n========================================================")
        print(f"Starting Rank Sweep for Module: {mod}")
        print(f"Ranks to evaluate: {ranks}")
        print(f"========================================================")

        rows = []
        for rank in ranks:
            row = run_single_rank(mod, rank, args)
            row["rank"] = rank
            rows.append(row)

        write_rank_curve(mod, rows, args.output_base)
        print_rank_table(mod, rows)


if __name__ == "__main__":
    main()
