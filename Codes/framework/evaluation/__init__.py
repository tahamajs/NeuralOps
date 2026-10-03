"""Evaluation module for model performance, fairness, robustness, label quality, and A/B testing."""

from framework.trainer import metrics
from framework.trainer.metrics import get_task_metrics

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
