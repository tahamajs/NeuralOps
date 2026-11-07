#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Evaluation APIs loaded on demand to keep package imports lightweight."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

_EXPORTS = {
    "BenchmarkEvaluator": ("neuralops.benchmarks.eval_bridge", "BenchmarkEvaluator"),
    "BenchmarkEvalResult": ("neuralops.benchmarks.eval_bridge", "BenchmarkEvalResult"),
    "evaluate_model": ("neuralops.benchmarks.eval_bridge", "evaluate_model"),
    "get_eval_bridge": ("neuralops.benchmarks.eval_bridge", "get_eval_bridge"),
    "run_comprehensive_benchmark": ("neuralops.benchmarks.eval_bridge", "run_comprehensive_benchmark"),
    "metrics": ("neuralops.evaluation.metrics", None),
    "get_task_metrics": ("neuralops.trainer.metrics", "get_task_metrics"),
    "probe_optimal_batch_size": ("neuralops.evaluation.eval_pipeline", "probe_optimal_batch_size"),
    "ABTester": ("neuralops.evaluation.ab_testing", "ABTester"),
    "ABTestResult": ("neuralops.evaluation.ab_testing", "ABTestResult"),
    "FairnessEvaluator": ("neuralops.evaluation.fairness", "FairnessEvaluator"),
    "FairnessReport": ("neuralops.evaluation.fairness", "FairnessReport"),
    "LabelQualityDetector": ("neuralops.evaluation.label_quality", "LabelQualityDetector"),
    "LabelQualityReport": ("neuralops.evaluation.label_quality", "LabelQualityReport"),
    "RetrainDecision": ("neuralops.evaluation.retraining", "RetrainDecision"),
    "RetrainPolicy": ("neuralops.evaluation.retraining", "RetrainPolicy"),
    "RetrainingOrchestrator": ("neuralops.evaluation.retraining", "RetrainingOrchestrator"),
    "TriggerType": ("neuralops.evaluation.retraining", "TriggerType"),
    "NeedleInAHaystackEvaluator": ("neuralops.evaluation.niah", "NeedleInAHaystackEvaluator"),
    "plot_niah_heatmap": ("neuralops.evaluation.niah", "plot_niah_heatmap"),
    "NeuralOpsLMEvalAdapter": ("neuralops.evaluation.lm_eval_adapter", "NeuralOpsLMEvalAdapter"),
    "evaluate_zero_shot_tasks": ("neuralops.evaluation.lm_eval_adapter", "evaluate_zero_shot_tasks"),
    "compute_perplexity_from_loss": ("neuralops.evaluation.metrics", "compute_perplexity_from_loss"),
    "compute_token_perplexity": ("neuralops.evaluation.metrics", "compute_token_perplexity"),
    "compute_top_k_accuracy": ("neuralops.evaluation.metrics", "compute_top_k_accuracy"),
    "compute_linear_cka": ("neuralops.evaluation.metrics", "compute_linear_cka"),
    "compute_representation_similarity": (
        "neuralops.evaluation.metrics",
        "compute_representation_similarity",
    ),
    "compute_numerical_deviation": ("neuralops.evaluation.metrics", "compute_numerical_deviation"),
    "measure_peak_vram_mb": ("neuralops.evaluation.metrics", "measure_peak_vram_mb"),
    "measure_latency_and_throughput": (
        "neuralops.evaluation.metrics",
        "measure_latency_and_throughput",
    ),
    "EvaluationReport": ("neuralops.evaluation.metrics", "EvaluationReport"),
}

# Scan evaluation directory to dynamically export all evaluation submodules
_eval_dir = Path(__file__).resolve().parent
_MODULE_EXPORTS = {
    f.stem: f"neuralops.evaluation.{f.stem}"
    for f in _eval_dir.glob("*.py")
    if f.is_file() and f.name not in ("__init__.py", "__main__.py")
}

__all__ = sorted([*_EXPORTS, *_MODULE_EXPORTS])


def __getattr__(name: str) -> Any:
    """Load a public evaluation symbol only when a caller requests it."""
    if name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        module = import_module(module_name)
        value = module if attribute is None else getattr(module, attribute)
    elif name in _MODULE_EXPORTS:
        value = import_module(_MODULE_EXPORTS[name])
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
