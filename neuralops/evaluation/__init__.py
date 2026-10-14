"""Evaluation module for model performance, fairness, robustness, label quality, and A/B testing."""

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

from .niah_passkey import NeedleInAHaystackEvaluator, plot_niah_heatmap
from .lm_eval_adapter import NeuralOpsLMEvalAdapter, evaluate_zero_shot_tasks

__all__ = [
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
]

# Auto-wired package members
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
