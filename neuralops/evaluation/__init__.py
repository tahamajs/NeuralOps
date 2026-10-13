"""Evaluation module for model performance, fairness, robustness, label quality, and A/B testing."""

from neuralops.trainer import metrics
from neuralops.trainer.metrics import get_task_metrics

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

__all__ = [
    "metrics",
    "get_task_metrics",
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
