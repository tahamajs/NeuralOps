"""Evaluation artifact generation and summary reporting.

Handles persistence of benchmark run artifacts (result.json, run directory metadata),
filelock-synchronized appending to consolidated summary.csv and summary.md, and structured logging.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from neuralops.evaluation.eval_types import BenchmarkResult, LM_DATASETS

logger = logging.getLogger("EvalEngine")


def format_summary(result: BenchmarkResult) -> str:
    """Format a human-readable one-line string summarizing benchmark results.

    Args:
        result: BenchmarkResult instance.

    Returns:
        Formatted summary string.
    """
    is_lm = result.dataset.lower() in LM_DATASETS
    metric_str = f"ppl={result.perplexity:.2f}" if is_lm else f"acc={result.accuracy_pct:.2f}%"
    return (
        f"[{result.status.upper()}] {result.module_name} on {result.model_family}/{result.dataset}: "
        f"{metric_str}, comp={result.compression_ratio:.3f}x, "
        f"latency={result.forward_latency_ms:.2f}ms, mem={result.peak_memory_mb:.1f}MB"
    )


def log_result_summary(result: BenchmarkResult, custom_logger: logging.Logger | None = None) -> None:
    """Log formatted summary of the evaluation result at INFO level.

    Args:
        result: BenchmarkResult instance.
        custom_logger: Optional logger; defaults to EvalEngine logger.
    """
    log = custom_logger or logger
    log.info(format_summary(result))


def save_benchmark_artifacts(
    result: BenchmarkResult,
    results_dir: str = "./results",
    comp_stats: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    elapsed_seconds: float = 0.0,
) -> None:
    """Persist structured benchmark artifacts to disk (JSON, CSV, Markdown).

    Creates run-specific directories with full JSON results and updates master summary files
    under a concurrency-safe file lock.

    Args:
        result: Evaluated BenchmarkResult object.
        results_dir: Base directory path to write artifacts to.
        comp_stats: Optional dictionary containing layer compression statistics.
        extra_metadata: Optional dictionary with error diagnostics and environmental info.
        elapsed_seconds: Total duration of the combination evaluation in seconds.
    """
    if comp_stats is None:
        comp_stats = {}
    if extra_metadata is None:
        extra_metadata = {}

    try:
        base_dir = Path(results_dir)
        run_name = f"{result.module_name}_{result.model_family}_{result.dataset}"
        run_dir = base_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        full_result_json = {
            "metrics": result.to_dict(),
            "compression_stats": comp_stats,
            "error_type": extra_metadata.get("error_type"),
            "traceback": extra_metadata.get("traceback"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": extra_metadata.get("git_sha", "clean"),
            "git_dirty": extra_metadata.get("git_dirty", False),
        }
        (run_dir / "result.json").write_text(
            json.dumps(full_result_json, indent=2), encoding="utf-8"
        )

        # Append to consolidated summary.csv and summary.md
        summary_dir = base_dir
        summary_dir.mkdir(parents=True, exist_ok=True)
        csv_file = summary_dir / "summary.csv"
        csv_header = (
            "module,model_family,dataset,status,"
            "accuracy,ppl,f1,"
            "num_compressed,num_fallback,num_skipped,"
            "params_before,params_after,compression_rate,"
            "avg_recon_error,max_error_threshold,"
            "calibration_used,calibration_source,"
            "peak_vram_mb,elapsed_s,"
            "forward_latency_ms,throughput_tok_per_sec,"
            "fidelity_score,error_message\n"
        )
        is_lm = result.dataset.lower() in LM_DATASETS
        num_compressed = result.num_compressed_layers
        num_fallback = result.num_fallback_layers
        row = (
            f"{result.module_name},{result.model_family},{result.dataset},{result.status},"
            f"{result.accuracy_pct if not is_lm else ''},"
            f"{result.perplexity if is_lm else ''},"
            f"{result.f1_score if not is_lm else ''},"
            f"{num_compressed},{num_fallback},0,"
            f"{result.params_before},{result.params_after},{round(result.compression_ratio, 4)},"
            f"{round(comp_stats.get('avg_reconstruction_error', 0.0), 4)},0.60,"
            f"{result.calibration_used},{result.calibration_source or 'none'},"
            f"{result.peak_memory_mb},{round(elapsed_seconds, 2)},"
            f"{result.forward_latency_ms},{result.throughput_tokens_sec},"
            f"{result.fidelity_score},{result.error_message or ''}\n"
        )

        try:
            import filelock

            lock_p = summary_dir / "summary.lock"
            with filelock.FileLock(str(lock_p), timeout=30):
                if not csv_file.exists():
                    csv_file.write_text(csv_header + row, encoding="utf-8")
                else:
                    with open(csv_file, "a", encoding="utf-8") as f:
                        f.write(row)

                md_file = summary_dir / "summary.md"
                md_metric = result.perplexity if is_lm else result.accuracy_pct
                md_line = (
                    f"| {result.module_name} | {result.model_family} | {result.dataset} | "
                    f"{result.status} | {md_metric} | {result.f1_score} | {num_compressed} | "
                    f"{round(result.compression_ratio, 3)}x | {result.calibration_used} | "
                    f"{round(elapsed_seconds, 1)}s |\n"
                )
                if not md_file.exists():
                    md_header = (
                        "# Evaluation Summary\n\n"
                        "| Module | Model | Dataset | Status | Acc/PPL | F1 | Compressed Layers | Comp Rate | Calibrated | Elapsed |\n"
                        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
                    )
                    md_file.write_text(md_header + md_line, encoding="utf-8")
                else:
                    with open(md_file, "a", encoding="utf-8") as f:
                        f.write(md_line)
        except ImportError:
            # Fallback if filelock is absent
            if not csv_file.exists():
                csv_file.write_text(csv_header + row, encoding="utf-8")
            else:
                with open(csv_file, "a", encoding="utf-8") as f:
                    f.write(row)

    except Exception as save_e:
        logger.error("Failed to write comprehensive run artifacts: %s", save_e)


__all__ = [
    "format_summary",
    "log_result_summary",
    "save_benchmark_artifacts",
]
