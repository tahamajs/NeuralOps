#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Evaluation module for model performance, fairness, robustness, label quality, and A/B testing.
"""

from neuralops.trainer import metrics
from neuralops.trainer.metrics import get_task_metrics
from .eval_pipeline import probe_optimal_batch_size

from .ab_testing import ABTester, ABTestResult
from .fairness import FairnessEvaluator, FairnessReport
from .label_quality import LabelQualityDetector, LabelQualityReport
from .retraining import (
    RetrainDecision,
    RetrainingOrchestrator,
    RetrainPolicy,
    TriggerType,
)

try:
    from .robustness import RobustnessReport, TextAdversarialTester
except ImportError:
    RobustnessReport = None  # type: ignore
    TextAdversarialTester = None  # type: ignore

from .niah import NeedleInAHaystackEvaluator, plot_niah_heatmap
from .lm_eval_adapter import NeuralOpsLMEvalAdapter, evaluate_zero_shot_tasks

from .metrics import (
    compute_perplexity_from_loss,
    compute_token_perplexity,
    compute_top_k_accuracy,
    compute_linear_cka,
    compute_representation_similarity,
    compute_numerical_deviation,
    measure_peak_vram_mb,
    measure_latency_and_throughput,
    EvaluationReport,
)

from neuralops.benchmarks.eval_bridge import (
    BenchmarkEvalResult,
    BenchmarkEvaluator,
    evaluate_model,
    get_eval_bridge,
    run_comprehensive_benchmark,
)

__all__ = [
    "BenchmarkEvaluator",
    "BenchmarkEvalResult",
    "evaluate_model",
    "get_eval_bridge",
    "run_comprehensive_benchmark",
    "metrics",
    "get_task_metrics",
    "probe_optimal_batch_size",
    "ABTester",
    "ABTestResult",
    "FairnessEvaluator",
    "FairnessReport",
    "LabelQualityDetector",
    "LabelQualityReport",
    "RetrainDecision",
    "RetrainPolicy",
    "RetrainingOrchestrator",
    "TriggerType",
    "RobustnessReport",
    "TextAdversarialTester",
    "NeedleInAHaystackEvaluator",
    "plot_niah_heatmap",
    "NeuralOpsLMEvalAdapter",
    "evaluate_zero_shot_tasks",
    "compute_perplexity_from_loss",
    "compute_token_perplexity",
    "compute_top_k_accuracy",
    "compute_linear_cka",
    "compute_representation_similarity",
    "compute_numerical_deviation",
    "measure_peak_vram_mb",
    "measure_latency_and_throughput",
    "EvaluationReport",
]

from neuralops.evaluation import ab_testing
from neuralops.evaluation import eval_all_models_datasets
from neuralops.evaluation import eval_data
from neuralops.evaluation import eval_patcher
from neuralops.evaluation import eval_perplexity
from neuralops.evaluation import eval_pipeline
from neuralops.evaluation import eval_reporter
from neuralops.evaluation import eval_types
from neuralops.evaluation import eval_utils
from neuralops.evaluation import fairness
from neuralops.evaluation import full_mnli_validation
from neuralops.evaluation import label_quality
from neuralops.evaluation import metrics
from neuralops.evaluation import retraining
from neuralops.evaluation import run_baseline_comparison
from neuralops.evaluation import run_lm_eval_harness
