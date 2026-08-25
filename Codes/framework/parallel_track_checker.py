"""
Parallel Multi-Track Framework Validator.

Executes all 5 tracks in parallel using ProcessPoolExecutor:
- Track 1: SVD & Low-Rank Factorization
- Track 2: Spectral, SSM & Recurrent
- Track 3: MoE, Hybrid & Routing
- Track 4: Kernel Approximations & Quantization
- Track 5: Structural Pruning & Sparsity

Verifies:
1. Dynamic layer replacement into BERT, RoBERTa, and T5.
2. Forward pass and shape consistency across hidden_size=128 and hidden_size=768.
3. Backward pass gradient propagation.
4. Compression factor computation and statistics logging.
5. Saves individual JSON results to disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure framework is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(CODES_DIR, ".."))
for p in [PROJECT_DIR, CODES_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from framework.eval_all_models_datasets import (
    DEFAULT_RESULTS_DIR,
    TRACK_MODULES,
    BenchmarkResult,
    evaluate_single_combination,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ParallelChecker")


def validate_single_track(
    track_key: str,
    models: List[str],
    datasets: List[str],
    rank: int = 16,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> Tuple[str, int, int, List[BenchmarkResult]]:
    """Validates all modules within a single track."""
    track_data = TRACK_MODULES.get(track_key, {})
    attentions = track_data.get("attentions", [])
    compressors = track_data.get("compressors", [])

    results: List[BenchmarkResult] = []
    logger.info("⚡ [Worker %s] Starting verification of %d attentions, %d compressors...",
                track_key, len(attentions), len(compressors))

    # Attentions
    for attn in attentions:
        for m in models:
            for d in datasets:
                res = evaluate_single_combination(
                    track=track_key,
                    module_name=attn,
                    module_type="attention",
                    model_family=m,
                    dataset=d,
                    rank=rank,
                    results_dir=results_dir,
                )
                results.append(res)

    # Compressors
    for comp in compressors:
        for m in models:
            for d in datasets:
                res = evaluate_single_combination(
                    track=track_key,
                    module_name=comp,
                    module_type="compressor",
                    model_family=m,
                    dataset=d,
                    rank=rank,
                    results_dir=results_dir,
                )
                results.append(res)

    passed = sum(1 for r in results if r.status == "PASS")
    total = len(results)
    logger.info("✅ [Worker %s] Completed: %d/%d PASSED (%.1f%%)",
                track_key, passed, total, (passed / max(1, total)) * 100.0)

    return track_key, passed, total, results


def run_parallel_verification(
    models: List[str],
    datasets: List[str],
    max_workers: int = 5,
    rank: int = 16,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    """Runs all 5 tracks in parallel using concurrent processes."""
    print("=" * 80)
    print("      NEURALOPS 5-TRACK PARALLEL FRAMEWORK VERIFICATION ENGINE")
    print(f"      Parallel Workers: {max_workers} | Models: {models} | Datasets: {datasets}")
    print("=" * 80)

    start_time = time.time()
    tracks = list(TRACK_MODULES.keys())
    all_results: Dict[str, Tuple[int, int]] = {}
    master_results: List[BenchmarkResult] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_track = {
            executor.submit(
                validate_single_track,
                t,
                models,
                datasets,
                rank,
                results_dir,
            ): t
            for t in tracks
        }

        for future in concurrent.futures.as_completed(future_to_track):
            track_name = future_to_track[future]
            try:
                t_key, passed, total, res_list = future.result()
                all_results[t_key] = (passed, total)
                master_results.extend(res_list)
            except Exception as e:
                logger.error("Track '%s' raised an unhandled exception: %s", track_name, e)
                all_results[track_name] = (0, 1)

    elapsed = time.time() - start_time
    total_evals = sum(tot for _, tot in all_results.values())
    total_passed = sum(p for p, _ in all_results.values())
    pass_pct = (total_passed / max(1, total_evals)) * 100.0

    print("\n" + "=" * 80)
    print("                     PARALLEL VERIFICATION SUMMARY")
    print("=" * 80)
    for t_key in tracks:
        passed, total = all_results.get(t_key, (0, 0))
        pct = (passed / max(1, total)) * 100.0 if total > 0 else 0.0
        status_str = "✅ PASS" if passed == total and total > 0 else "⚠️ INCOMPLETE"
        print(f"  {status_str} | {t_key:<22} : {passed:>3}/{total:<3} ({pct:>5.1f}%)")

    print("-" * 80)
    print(f"  Total Evaluations : {total_evals}")
    print(f"  Successful Passes : {total_passed} / {total_evals} ({pass_pct:.1f}%)")
    print(f"  Total Time        : {elapsed:.2f} seconds")
    print("=" * 80)

    return {
        "total_evals": total_evals,
        "total_passed": total_passed,
        "pass_pct": pass_pct,
        "elapsed_seconds": elapsed,
        "track_summaries": all_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel Multi-Track Framework Validator")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent worker processes")
    parser.add_argument("--models", type=str, default="bert,roberta,t5", help="Comma-separated model architectures")
    parser.add_argument("--datasets", type=str, default="sst2,mrpc", help="Comma-separated evaluation datasets")
    parser.add_argument("--rank", type=int, default=16, help="Target rank for low-rank modules")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Results output directory")
    args = parser.parse_args()

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]

    run_parallel_verification(
        models=models,
        datasets=datasets,
        max_workers=args.workers,
        rank=args.rank,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
