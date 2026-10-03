#!/usr/bin/env python3
"""
Baseline Comparison Runner: Benchmarks key literature baselines against Proposed method.

Baselines evaluated:
  1. Dense (Original uncompressed baseline model)
  2. Naive SVD (Truncated SVD without data calibration or whitening)
  3. ASVD (Activation-aware SVD scaling, Yuan et al., 2023)
  4. FWSVD / GFWS (Fisher-weighted SVD, Wang et al., 2022)
  5. SVD-LLM (Cholesky activation-whitened SVD, Wang et al., 2024)
  6. Ours (A3 / Adaptive Activation-Aware Compression)

Datasets evaluated:
  - SST-2 (GLUE binary classification)
  - MNLI (GLUE 3-way natural language inference)
  - AG News (4-class topic categorization)

Usage:
    python run_baseline_comparison.py --dry_run
    python run_baseline_comparison.py --rank 64
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).resolve().parent
CODES = HERE.parent
ROOT = CODES.parent
sys.path[:0] = [str(CODES), str(ROOT)]

from datasets import load_dataset, load_from_disk
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.eval_all_models_datasets import apply_module_to_model
from neuralops.benchmarks.run_quality_sweep import evaluate_comprehensive, prepare_eval_batches

LOG = logging.getLogger("BaselineComparison")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

BASELINES = [
    ("dense", "Original uncompressed full-rank baseline"),
    ("naive_svd", "Uncalibrated plain truncated SVD"),
    ("asvd", "Activation-Aware SVD (Yuan et al., 2023)"),
    ("gfws_d", "Fisher-Weighted / Covariance SVD (Wang et al., 2022)"),
    ("svd_llm_v2", "SVD-LLM Cholesky Whitened SVD (Wang et al., 2024)"),
    ("a3", "Ours: Adaptive Activation-Aware Decomposition"),
]

DATASETS = ["sst2", "mnli", "ag_news"]

DATASET_CONFIGS = {
    "sst2": {
        "model": "models/distilbert_sst2",
        "dataset_dir": "neuralops/data/glue_data/sst2",
        "num_labels": 2,
    },
    "mnli": {
        "model": "models/distilbert_mnli_full",
        "dataset_dir": "neuralops/data/glue_data/mnli",
        "num_labels": 3,
    },
    "ag_news": {
        "model": "models/distilbert_agnews_full",
        "dataset_dir": "neuralops/data/ag_news",
        "num_labels": 4,
    },
}

CSV_FIELDS = [
    "dataset",
    "method",
    "status",
    "accuracy_pct",
    "f1_score",
    "compression_rate",
    "compression_ratio_pct",
    "peak_vram_mb",
    "latency_ms_per_token",
    "throughput_tok_per_sec",
    "elapsed_s",
    "total_params_before",
    "total_params_after",
    "timestamp",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run comparative evaluation of literature baselines against proposed method",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--rank", type=int, default=64, help="Decomposition rank for low-rank methods"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/01_baselines/baseline_comparison",
        help="Destination directory for outputs",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=DATASETS, help="List of datasets to evaluate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Evaluation batch size"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Evaluate in FP16 precision"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (kept 1 for single GPU)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Print planned execution table without running evaluations",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("ICLR BASELINE COMPARISON BENCHMARK SUITE")
    print(f"Target Rank:   {args.rank}")
    print(f"Datasets:      {args.datasets}")
    print(f"Methods:       {[b[0] for b in BASELINES]}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Planned evaluation matrix:")
        for ds in args.datasets:
            for b_name, b_desc in BASELINES:
                print(f"  Dataset: {ds:<10} | Baseline: {b_name:<15} ({b_desc})")
        print("\n[DRY RUN COMPLETE] No hardware sweeps were executed.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    all_rows = []

    for ds_name in args.datasets:
        cfg = DATASET_CONFIGS.get(ds_name, {})
        model_path = cfg.get("model", f"models/distilbert_{ds_name}")
        ds_dir = out_dir / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        LOG.info("=== Starting Baseline Suite for Dataset: %s ===", ds_name)

        for b_name, b_desc in BASELINES:
            LOG.info("Evaluating [%s] on %s...", b_name, ds_name)
            t0 = time.time()
            res_file = ds_dir / f"{b_name}.json"

            try:
                # Load fresh model
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                if args.fp16 and device.type == "cuda":
                    model.half()
                model.to(device)
                p_before = sum(p.numel() for p in model.parameters())

                if b_name != "dense":
                    typ = "attention" if b_name in ATTENTION_REGISTRY else "compressor"
                    model, comp_stats = apply_module_to_model(
                        model, b_name, typ, rank=args.rank, device=device
                    )

                p_after = sum(p.numel() for p in model.parameters())
                comp_rate = round(p_before / max(1, p_after), 4)
                comp_pct = round(100.0 * (1.0 - p_after / max(1, p_before)), 2)

                # Real evaluation via evaluate_comprehensive
                tok = AutoTokenizer.from_pretrained(model_path)
                ds_path = cfg.get("dataset_dir")
                if ds_path and Path(ds_path).exists():
                    val_ds = load_from_disk(ds_path)
                elif Path(f"neuralops/data/glue_data/{ds_name}").exists():
                    val_ds = load_from_disk(f"neuralops/data/glue_data/{ds_name}")
                else:
                    val_ds = load_dataset(
                        "glue" if ds_name != "ag_news" else "ag_news",
                        ds_name if ds_name != "ag_news" else None,
                    )

                val_split = val_ds.get(
                    "validation_matched",
                    val_ds.get("validation", val_ds.get("test", None)),
                )
                eval_batches = prepare_eval_batches(
                    tok, val_split, ds_name, 128, args.batch_size, device
                )
                eval_metrics = evaluate_comprehensive(
                    model,
                    tok,
                    val_split,
                    ds_name,
                    128,
                    args.batch_size,
                    device,
                    fp16=args.fp16,
                    precomputed_batches=eval_batches,
                )

                elapsed = round(time.time() - t0, 2)
                row = {
                    "dataset": ds_name,
                    "method": b_name,
                    "status": "PASS",
                    "accuracy_pct": round(eval_metrics.get("accuracy_pct", 0.0), 2),
                    "f1_score": round(eval_metrics.get("f1_score", 0.0), 4),
                    "compression_rate": comp_rate,
                    "compression_ratio_pct": comp_pct,
                    "peak_vram_mb": eval_metrics.get("peak_vram_mb", 0.0),
                    "latency_ms_per_token": eval_metrics.get("latency_ms", 0.0),
                    "throughput_tok_per_sec": eval_metrics.get("tokens_per_sec", 0.0),
                    "elapsed_s": elapsed,
                    "total_params_before": p_before,
                    "total_params_after": p_after,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                LOG.exception(
                    "Baseline %s on %s encountered error: %s", b_name, ds_name, e
                )
                row = {
                    "dataset": ds_name,
                    "method": b_name,
                    "status": "FAIL",
                    "accuracy_pct": 0.0,
                    "f1_score": 0.0,
                    "compression_rate": 1.0,
                    "compression_ratio_pct": 0.0,
                    "peak_vram_mb": 0.0,
                    "latency_ms_per_token": 0.0,
                    "throughput_tok_per_sec": 0.0,
                    "elapsed_s": round(time.time() - t0, 2),
                    "total_params_before": 0,
                    "total_params_after": 0,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }

            all_rows.append(row)
            res_file.write_text(json.dumps(row, indent=2), encoding="utf-8")

            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                for r in all_rows:
                    writer.writerow(r)

    print(f"\n[DONE] Baseline comparison completed. Summary: {summary_csv}")


if __name__ == "__main__":
    main()
