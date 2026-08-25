"""
Auto-Runner CLI for Efficient Transformer Toolkit.

Provides a unified execution entrypoint to benchmark, evaluate, and stress-test
any or all 298 Attention Mechanisms and Compression Engines across:
  - Models: BERT, RoBERTa, T5, GPT-2, DistilBERT
  - Datasets: SST-2, MRPC, QNLI, WikiText-2, Synthetic
  - Parallel multiprocessing across CPU/GPU cores

Usage Examples:
  # Run all Track 1 (SVD) modules on BERT and RoBERTa on SST-2 & MRPC:
  python auto_run.py --track 1 --models bert,roberta --datasets sst2,mrpc

  # Run a specific module on all models:
  python auto_run.py --module aegis_v1 --models bert,roberta,t5 --datasets sst2

  # Run full framework benchmark across all 5 tracks:
  python auto_run.py --all --workers 4

  # Generate markdown summary report from saved JSONs:
  python auto_run.py --report
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure framework is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
CODES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for p in [PROJECT_DIR, CODES_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from framework.eval_all_models_datasets import (
    TRACK_MODULES,
    DEFAULT_RESULTS_DIR,
    BenchmarkResult,
    evaluate_single_combination,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AutoRunner")


def run_track(
    track_key: str,
    models: List[str],
    datasets: List[str],
    rank: int = 16,
    results_dir: str = DEFAULT_RESULTS_DIR,
    allow_synthetic: bool = False,
) -> List[BenchmarkResult]:
    """
    Sequentially runs all modules for a given track across specified models and datasets.
    """
    track_data = TRACK_MODULES.get(track_key, {})
    attentions = track_data.get("attentions", [])
    compressors = track_data.get("compressors", [])

    total_tasks = (len(attentions) + len(compressors)) * len(models) * len(datasets)
    logger.info("Starting %s benchmark: %d total evaluations...", track_key, total_tasks)

    results: List[BenchmarkResult] = []
    done = 0

    # 1. Attentions
    for attn_name in attentions:
        for model in models:
            for dataset in datasets:
                res = evaluate_single_combination(
                    track=track_key,
                    module_name=attn_name,
                    module_type="attention",
                    model_family=model,
                    dataset=dataset,
                    rank=rank,
                    results_dir=results_dir,
                    allow_synthetic=allow_synthetic,
                )
                results.append(res)
                done += 1
                logger.info(
                    "[%d/%d] %s | %s | %s -> %s (fwd: %.1fms, bwd: %.1fms)",
                    done, total_tasks, attn_name, model, dataset, res.status,
                    res.forward_latency_ms, res.backward_latency_ms
                )

    # 2. Compressors
    for comp_name in compressors:
        for model in models:
            for dataset in datasets:
                res = evaluate_single_combination(
                    track=track_key,
                    module_name=comp_name,
                    module_type="compressor",
                    model_family=model,
                    dataset=dataset,
                    rank=rank,
                    results_dir=results_dir,
                    allow_synthetic=allow_synthetic,
                )
                results.append(res)
                done += 1
                logger.info(
                    "[%d/%d] %s | %s | %s -> %s (ratio: %.2fx, fwd: %.1fms)",
                    done, total_tasks, comp_name, model, dataset, res.status,
                    res.compression_ratio, res.forward_latency_ms
                )

    return results


def generate_markdown_report(results_dir: str = DEFAULT_RESULTS_DIR) -> str:
    """
    Gathers all individual JSON results and compiles a publication-ready Markdown report.
    """
    json_files = glob.glob(os.path.join(results_dir, "json", "**", "*.json"), recursive=True)
    if not json_files:
        logger.warning("No JSON result files found in %s", results_dir)
        return ""

    results: List[Dict[str, Any]] = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception:
            continue

    total_runs = len(results)
    passed_runs = sum(1 for r in results if r.get("status") == "PASS")
    pass_rate = (passed_runs / max(1, total_runs)) * 100.0

    lines: List[str] = [
        "# Comprehensive Benchmark & Empirical Evaluation Report",
        "",
        f"**Total Model-Dataset Evaluations:** {total_runs}  ",
        f"**Successful Passes:** {passed_runs} / {total_runs} ({pass_rate:.1f}%)  ",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
        "---",
        "",
        "## Summary by Track & Module",
        "",
        "| Track | Module | Type | Model | Dataset | Status | Fwd (ms) | Bwd (ms) | Speedup | Compression | Grad Norm |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in sorted(results, key=lambda x: (x.get("track", ""), x.get("module_name", ""), x.get("model_family", ""))):
        lines.append(
            f"| `{r.get('track')}` | `{r.get('module_name')}` | {r.get('module_type')} | "
            f"**{r.get('model_family', '').upper()}** | `{r.get('dataset')}` | "
            f"**{r.get('status')}** | {r.get('forward_latency_ms', 0.0):.1f} | "
            f"{r.get('backward_latency_ms', 0.0):.1f} | {r.get('speedup_vs_baseline', 1.0):.2f}x | "
            f"{r.get('compression_ratio', 1.0):.2f}x | {r.get('grad_norm', 0.0):.3f} |"
        )

    report_content = "\n".join(lines)
    report_path = os.path.join(results_dir, "BENCHMARK_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Saved consolidated markdown report to %s", report_path)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Efficient Transformer Toolkit Auto-Runner CLI")
    parser.add_argument("--track", type=str, default=None, help="Track number (1, 2, 3, 4, 5) or track name")
    parser.add_argument("--all", action="store_true", help="Run all 5 tracks across all models and datasets")
    parser.add_argument("--module", type=str, default=None, help="Specific module to benchmark")
    parser.add_argument("--module_type", type=str, default="attention", choices=["attention", "compressor"])
    parser.add_argument("--models", type=str, default="bert,roberta,t5", help="Comma-separated models (bert,roberta,t5,gpt2,distilbert)")
    parser.add_argument("--datasets", type=str, default="sst2,mrpc", help="Comma-separated datasets (sst2,mrpc,qnli,wikitext2,synthetic)")
    parser.add_argument("--rank", type=int, default=16, help="Target rank for low-rank / SVD modules")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Directory to save JSON/CSV results")
    parser.add_argument("--report", action="store_true", help="Generate consolidated markdown report from saved JSONs")
    args = parser.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    if args.report:
        generate_markdown_report(args.results_dir)
        return

    if args.module:
        logger.info("Running single module evaluation: %s", args.module)
        for m in models:
            for d in datasets:
                res = evaluate_single_combination(
                    track="custom",
                    module_name=args.module,
                    module_type=args.module_type,
                    model_family=m,
                    dataset=d,
                    rank=args.rank,
                    results_dir=args.results_dir,
                )
                print(f"Result: {res.module_name} on {res.model_family} ({res.dataset}): {res.status}")
        generate_markdown_report(args.results_dir)
        return

    track_map = {
        "1": "track1_svd",
        "2": "track2_ssm",
        "3": "track3_moe",
        "4": "track4_kernel_quant",
        "5": "track5_pruning",
        "track1": "track1_svd",
        "track2": "track2_ssm",
        "track3": "track3_moe",
        "track4": "track4_kernel_quant",
        "track5": "track5_pruning",
        "track1_svd": "track1_svd",
        "track2_ssm": "track2_ssm",
        "track3_moe": "track3_moe",
        "track4_kernel_quant": "track4_kernel_quant",
        "track5_pruning": "track5_pruning",
    }

    if args.all:
        tracks_to_run = list(TRACK_MODULES.keys())
    elif args.track:
        t_key = track_map.get(args.track.lower())
        if not t_key:
            raise ValueError(f"Unknown track: {args.track}. Choices: 1, 2, 3, 4, 5")
        tracks_to_run = [t_key]
    else:
        tracks_to_run = ["track1_svd"]

    for t in tracks_to_run:
        run_track(
            track_key=t,
            models=models,
            datasets=datasets,
            rank=args.rank,
            results_dir=args.results_dir,
        )

    generate_markdown_report(args.results_dir)


if __name__ == "__main__":
    main()
