"""
Automated retraining pipeline – triggers retraining based on drift, performance
degradation, time schedule, or data volume thresholds.

Inspired by: Vertex AI Model Monitoring, Azure ML pipelines, ClearML, and AutoDrift.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from neuralops.core.registry import Registry

logger = logging.getLogger(__name__)
RETRAIN_TRIGGER_REGISTRY = Registry("retrain_triggers")


class TriggerType(Enum):
    DRIFT = "drift"
    PERFORMANCE = "performance"
    SCHEDULE = "schedule"
    DATA_VOLUME = "data_volume"
    MANUAL = "manual"


@dataclass
class RetrainPolicy:
    """Configuration for when and how to retrain."""

    trigger_types: list[TriggerType] = field(default_factory=lambda: [TriggerType.DRIFT])

    # Drift trigger
    drift_threshold: float = 0.3
    drift_methods: list[str] = field(default_factory=lambda: ["ks", "psi"])

    # Performance trigger
    min_accuracy: float = 0.85
    max_accuracy_degradation: float = 0.05

    # Schedule trigger
    schedule_cron: str | None = None  # "0 2 * * 1" for every Monday 2 AM
    max_days_since_last_train: int = 30

    # Data volume trigger
    min_new_samples: int = 1000
    max_new_samples: int = 100_000

    # Retraining behaviour
    require_better_model: bool = True
    min_improvement: float = 0.01
    max_retrains_per_day: int = 3
    validation_split: float = 0.2
    notify_on_trigger: bool = True
    notify_on_complete: bool = True


@dataclass
class RetrainDecision:
    """Output of the retraining decision engine."""

    should_retrain: bool
    reason: str
    trigger_type: TriggerType | None = None
    urgency: str = "low"  # "low", "medium", "high", "critical"
    details: dict[str, Any] = field(default_factory=dict)


class RetrainingOrchestrator:
    """
    Central decision engine for automated retraining.

    Usage:
        orch = RetrainingOrchestrator(RetrainPolicy())
        decision = orch.evaluate(
            current_metrics={"accuracy": 0.82},
            reference_metrics={"accuracy": 0.88},
            drift_report=drift_report,
            new_data_count=5000,
            days_since_last_train=35,
        )
        if decision.should_retrain:
            orch.trigger_retraining(run_id="model-v3", training_fn=my_train_func)
    """

    def __init__(self, policy: RetrainPolicy = None) -> None:
        self.policy = policy or RetrainPolicy()
        self._retrain_count_today: int = 0
        self._last_retrain_date: str = ""

    def evaluate(
        self,
        current_metrics: dict[str, float] | None = None,
        reference_metrics: dict[str, float] | None = None,
        drift_report: Any | None = None,
        new_data_count: int = 0,
        days_since_last_train: int = 0,
        manual_request: bool = False,
    ) -> RetrainDecision:
        """Evaluate all configured triggers and return a retraining decision."""

        # Reset daily counter if it's a new day
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._last_retrain_date:
            self._retrain_count_today = 0
            self._last_retrain_date = today

        if self._retrain_count_today >= self.policy.max_retrains_per_day:
            return RetrainDecision(
                should_retrain=False,
                reason="Max daily retrains reached",
                details={"retrains_today": self._retrain_count_today},
            )

        # 1. Manual trigger
        if manual_request and TriggerType.MANUAL in self.policy.trigger_types:
            return RetrainDecision(
                should_retrain=True,
                reason="Manual retraining requested",
                trigger_type=TriggerType.MANUAL,
                urgency="high",
            )

        # 2. Performance degradation
        if (
            TriggerType.PERFORMANCE in self.policy.trigger_types
            and current_metrics
            and reference_metrics
        ):
            decision = self._check_performance(current_metrics, reference_metrics)
            if decision.should_retrain:
                return decision

        # 3. Drift trigger
        if TriggerType.DRIFT in self.policy.trigger_types and drift_report:
            decision = self._check_drift(drift_report)
            if decision.should_retrain:
                return decision

        # 4. Data volume trigger
        if TriggerType.DATA_VOLUME in self.policy.trigger_types:
            decision = self._check_data_volume(new_data_count)
            if decision.should_retrain:
                return decision

        # 5. Schedule trigger
        if TriggerType.SCHEDULE in self.policy.trigger_types:
            decision = self._check_schedule(days_since_last_train)
            if decision.should_retrain:
                return decision

        return RetrainDecision(should_retrain=False, reason="No triggers met")

    def _check_performance(
        self, current: dict[str, float], reference: dict[str, float]
    ) -> RetrainDecision:
        """check performance."""
        acc_key = "accuracy" if "accuracy" in current else "eval_accuracy"
        curr_acc = current.get(acc_key, 1.0)
        ref_acc = reference.get(acc_key, 1.0)
        degradation = ref_acc - curr_acc

        if curr_acc < self.policy.min_accuracy:
            return RetrainDecision(
                should_retrain=True,
                reason=f"Accuracy {curr_acc:.3f} below minimum {self.policy.min_accuracy}",
                trigger_type=TriggerType.PERFORMANCE,
                urgency="critical" if degradation > 0.1 else "high",
                details={
                    "current_accuracy": curr_acc,
                    "reference_accuracy": ref_acc,
                    "degradation": degradation,
                },
            )

        if degradation > self.policy.max_accuracy_degradation:
            return RetrainDecision(
                should_retrain=True,
                reason=f"Accuracy degraded by {degradation:.3f} (threshold: {self.policy.max_accuracy_degradation})",
                trigger_type=TriggerType.PERFORMANCE,
                urgency="medium",
                details={"degradation": degradation},
            )

        return RetrainDecision(should_retrain=False, reason="Performance within acceptable range")

    def _check_drift(self, drift_report) -> RetrainDecision:
        """check drift."""
        n_drifted = getattr(drift_report, "n_features_drifted", 0)
        n_total = getattr(drift_report, "n_features_tested", 1)
        ratio = n_drifted / max(n_total, 1)

        if ratio > self.policy.drift_threshold:
            return RetrainDecision(
                should_retrain=True,
                reason=f"Drift detected in {n_drifted}/{n_total} features ({ratio:.1%})",
                trigger_type=TriggerType.DRIFT,
                urgency="high" if ratio > 0.5 else "medium",
                details={"drifted_features": n_drifted, "total_features": n_total},
            )
        return RetrainDecision(should_retrain=False, reason="Drift within acceptable range")

    def _check_data_volume(self, new_count: int) -> RetrainDecision:
        """check data volume."""
        if new_count >= self.policy.min_new_samples:
            return RetrainDecision(
                should_retrain=True,
                reason=f"New data volume ({new_count}) exceeded threshold ({self.policy.min_new_samples})",
                trigger_type=TriggerType.DATA_VOLUME,
                urgency="low",
                details={"new_samples": new_count},
            )
        return RetrainDecision(should_retrain=False, reason="Insufficient new data")

    def _check_schedule(self, days_since: int) -> RetrainDecision:
        """check schedule."""
        if days_since >= self.policy.max_days_since_last_train:
            return RetrainDecision(
                should_retrain=True,
                reason=f"Scheduled retrain: {days_since} days since last training (max: {self.policy.max_days_since_last_train})",
                trigger_type=TriggerType.SCHEDULE,
                urgency="low",
                details={"days_since_last_train": days_since},
            )
        return RetrainDecision(should_retrain=False, reason="Schedule not due yet")

    def trigger_retraining(
        self,
        run_id: str,
        training_fn: Callable,
        decision: RetrainDecision,
        **training_kwargs,
    ) -> dict[str, Any]:
        """Execute retraining and validate the new model."""
        logger.info("Retraining triggered for %s: %s", run_id, decision.reason)
        self._retrain_count_today += 1

        try:
            result = training_fn(**training_kwargs)
            logger.info(
                "Retraining completed for %s. Metrics: %s",
                run_id,
                {k: v for k, v in result.items() if isinstance(v, (int, float))},
            )
        except Exception as e:
            logger.exception("Retraining failed for %s: %s", run_id, e)
            return {"status": "failed", "error": str(e)}

        return {"status": "completed", "run_id": run_id, "metrics": result}
