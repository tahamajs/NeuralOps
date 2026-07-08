"""Celery tasks for anomaly detection and HPO."""

import logging

from celery import shared_task
from django.utils import timezone

logger = logging.getLogger(__name__)


@shared_task
def analyze_logs_task(run_id: str):
    """Analyze logs for a specific training run using the LLM."""
    from experiment.models import TrainingRun

    from .llm_log_analyzer import LLMLogAnalyzer
    from .models import ExperimentLogAnalysis

    try:
        run = TrainingRun.objects.get(id=run_id)
    except TrainingRun.DoesNotExist:
        return "Run not found"

    logs = run.logs or ""
    if not logs.strip():
        return "No logs to analyze"

    analyzer = LLMLogAnalyzer()
    anomalies = analyzer.analyze_logs(logs, run.final_metrics or {})

    created = 0
    for anomaly in anomalies:
        ExperimentLogAnalysis.objects.create(
            run=run,
            anomaly_type=anomaly.get("type", "loss_spike"),
            severity=anomaly.get("severity", 0.5),
            detected_at_step=anomaly.get("step"),
            description=anomaly.get("description", ""),
            suggested_fix=anomaly.get("suggested_fix", ""),
        )
        created += 1

    return f"Found {created} anomalies for run {run_id}"


@shared_task
def run_hpo_task(hpo_run_id: str):
    """Execute a full HPO run using the SMAC engine."""
    from .hpo_engine import SMACEngine
    from .models import HPORun, HPOTrial

    try:
        hpo_run = HPORun.objects.get(id=hpo_run_id)
    except HPORun.DoesNotExist:
        return "HPO run not found"

    if hpo_run.status != "queued":
        return f"HPO run is already {hpo_run.status}"

    hpo_run.status = "running"
    hpo_run.started_at = timezone.now()
    hpo_run.save()

    engine = SMACEngine(
        search_space=hpo_run.search_space,
        objective_direction=hpo_run.objective_direction,
    )

    for trial_num in range(1, hpo_run.max_trials + 1):
        # Check if cancelled
        hpo_run.refresh_from_db()
        if hpo_run.status == "canceled":
            return "HPO run cancelled"

        # Suggest next params
        params = engine.suggest_next()

        # Create trial
        trial = HPOTrial.objects.create(
            hpo_run=hpo_run,
            trial_number=trial_num,
            params=params,
            objective_metric=hpo_run.objective_metric,
            direction=hpo_run.objective_direction,
            status="pending",
        )

        # In production: launch actual training run with these params
        # For now, simulate an objective value
        trial.status = "completed"
        trial.objective_value = _simulate_objective(params, hpo_run.objective_metric)
        trial.completed_at = timezone.now()
        trial.save()

        # Observe the result
        engine.observe(params, trial.objective_value)

        # Update best
        if hpo_run.best_objective_value is None:
            hpo_run.best_objective_value = trial.objective_value
            hpo_run.best_trial = trial
            hpo_run.best_params = params
            hpo_run.save()
        elif (
            hpo_run.objective_direction == "maximize"
            and trial.objective_value > hpo_run.best_objective_value
        ):
            hpo_run.best_objective_value = trial.objective_value
            hpo_run.best_trial = trial
            hpo_run.best_params = params
            hpo_run.save()
        elif (
            hpo_run.objective_direction == "minimize"
            and trial.objective_value < hpo_run.best_objective_value
        ):
            hpo_run.best_objective_value = trial.objective_value
            hpo_run.best_trial = trial
            hpo_run.best_params = params
            hpo_run.save()

    hpo_run.status = "completed"
    hpo_run.completed_at = timezone.now()
    hpo_run.save()
    return f"HPO run completed. Best value: {hpo_run.best_objective_value}"


def _simulate_objective(params: dict, metric: str) -> float:
    """Simulate an objective value for testing."""
    import random

    lr = float(params.get("learning_rate", 2e-5))
    bs = int(params.get("batch_size", 16))
    # Simple heuristic: better values for moderate LR and larger batch
    base = 0.8
    if 1e-5 <= lr <= 5e-4:
        base += 0.05
    if bs >= 16:
        base += 0.05
    base += random.uniform(-0.02, 0.02)
    return round(min(0.99, max(0.5, base)), 4)
