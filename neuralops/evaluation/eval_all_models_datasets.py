"""Evaluation harness and unified entry point for NeuralOps benchmark suite.

This module is a clean re-export hub providing backward compatibility across all existing
evaluation scripts and CLI entry points.
Underlying implementations are factored into:
- eval_types.py: BenchmarkResult schema and dataset/track constants
- eval_utils.py: Batch filtering and logit/loss extraction
- eval_data.py: Dataset loaders and calibration pipeline
- eval_patcher.py: Model factory and module injection engine
- eval_reporter.py: Result formatting and disk artifact persistence
- eval_pipeline.py: Adaptive batch probing and single combination orchestrator
"""

from __future__ import annotations

import logging
import torch

# Re-export data models and constants
from neuralops.evaluation.eval_types import (
    ACTIVATION_AWARE_MODULES,
    BenchmarkResult,
    COMPUTE_OPTIMIZERS,
    DEPRECATED_MODULES,
    GLUE_DATASETS,
    KV_CACHE_OPTIMIZERS,
    LM_DATASETS,
    MODEL_TOKENIZER_MAPPING,
    TRACK_MODULES,
)

# Re-export evaluation utilities
from neuralops.evaluation.eval_utils import (
    DEFAULT_MAX_CACHE_SIZE,
    _evict_if_needed,
    _extract_logits_and_loss,
    _filter_batch,
)

# Re-export data helpers
from neuralops.evaluation.eval_data import (
    _generate_synthetic_batch,
    build_calibration_loader,
    get_dataset_batch,
)

# Re-export patcher functions
from neuralops.evaluation.eval_patcher import (
    apply_module_to_model,
    build_model_instance,
    get_base_model_cached,
)

# Re-export reporter functions
from neuralops.evaluation.eval_reporter import (
    format_summary,
    log_result_summary,
    save_benchmark_artifacts,
)

# Re-export pipeline orchestrator functions
from neuralops.evaluation.eval_pipeline import (
    evaluate_single_combination,
    evaluate_single_combination_opt,
    probe_optimal_batch_size,
)

logger = logging.getLogger("EvalEngine")


def run_track(
    track: str,
    model_family: str,
    dataset: str,
    rank: int = 64,
    seq_len: int = 128,
    batch_size: int = 8,
    device: torch.device | None = None,
    results_dir: str = "./results",
    allow_synthetic: bool = False,
) -> list[BenchmarkResult]:
    """Run all attention and compressor modules within a given track against target model and dataset.

    Args:
        track: Track name (e.g., 'track1_svd', 'track2_ssm', etc.).
        model_family: Architecture family name.
        dataset: Target dataset name.
        rank: Rank parameter.
        seq_len: Max sequence length.
        batch_size: Batch size.
        device: Execution device.
        results_dir: Base directory for output results.
        allow_synthetic: Whether to fall back to synthetic data if datasets cannot be fetched.

    Returns:
        List of completed BenchmarkResult objects.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modules_cfg = TRACK_MODULES.get(track)
    if modules_cfg is None:
        raise ValueError(f"Unknown track: {track}")

    results: list[BenchmarkResult] = []

    for mtype_key in ("attentions", "compressors"):
        for module_name in modules_cfg.get(mtype_key, []):
            module_type = "attention" if mtype_key == "attentions" else "compressor"
            logger.info("=== %s :: %s :: %s :: r=%d ===", track, module_name, model_family, rank)
            try:
                r = evaluate_single_combination_opt(
                    track=track,
                    module_name=module_name,
                    module_type=module_type,
                    model_family=model_family,
                    dataset=dataset,
                    rank=rank,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    device=device,
                    results_dir=results_dir,
                    allow_synthetic=allow_synthetic,
                )
                results.append(r)
                logger.info(
                    "--> %s: status=%s  acc=%.2f  ppl=%.2f  comp=%.3fx",
                    module_name,
                    getattr(r, "status", "unknown"),
                    getattr(r, "accuracy_pct", 0.0),
                    getattr(r, "perplexity", 0.0),
                    getattr(r, "compression_ratio", 1.0),
                )
            except Exception as e:
                logger.exception("Run crashed for %s: %s", module_name, e)
                # ساخت یک آبجکت نتیجه شکست‌خورده استاندارد برای جلوگیری از گم شدن ماژول در گزارش نهایی
                try:
                    crashed_r = BenchmarkResult(
                        run_id=f"{track}_{module_name}_{model_family}",
                        run_name=f"{module_name}_{model_family}",
                        module={"name": module_name, "type": module_type, "rank": rank},
                        model={"name": model_family},
                        dataset={"name": dataset},
                        config={"track": track, "rank": rank, "seq_len": seq_len},
                        metrics={},
                        resources={},
                        status="crashed",
                        error_message=str(e),
                    )
                    results.append(crashed_r)
                except Exception:
                    pass

    return results


def main() -> None:
    """CLI entry point for running evaluation tracks."""
    import argparse

    parser = argparse.ArgumentParser(description="NeuralOps Universal Evaluation Engine")
    parser.add_argument("--track", type=str, default="track1_svd", choices=list(TRACK_MODULES.keys()))
    parser.add_argument("--model", type=str, default="bert")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--allow_synthetic", action="store_true")

    args = parser.parse_args()
    run_track(
        track=args.track,
        model_family=args.model,
        dataset=args.dataset,
        rank=args.rank,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
        allow_synthetic=args.allow_synthetic,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BenchmarkResult",
    "TRACK_MODULES",
    "MODEL_TOKENIZER_MAPPING",
    "LM_DATASETS",
    "GLUE_DATASETS",
    "ACTIVATION_AWARE_MODULES",
    "COMPUTE_OPTIMIZERS",
    "KV_CACHE_OPTIMIZERS",
    "DEPRECATED_MODULES",
    "build_model_instance",
    "apply_module_to_model",
    "get_dataset_batch",
    "build_calibration_loader",
    "get_base_model_cached",
    "probe_optimal_batch_size",
    "evaluate_single_combination_opt",
    "evaluate_single_combination",
    "save_benchmark_artifacts",
    "format_summary",
    "log_result_summary",
    "run_track",
    "main",
    "_filter_batch",
    "_extract_logits_and_loss",
    "_evict_if_needed",
    "_generate_synthetic_batch",
    "DEFAULT_MAX_CACHE_SIZE",
]