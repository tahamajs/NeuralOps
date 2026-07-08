"""
Celery tasks for insight generation and recommendation creation.
"""

import logging
from datetime import timedelta

from celery import shared_task
from django.utils import timezone

from .insight_engine import analyze_user_experiments
from .models import (ExperimentInsight, ExperimentRecommendation,
                     ExperimentSummary)

logger = logging.getLogger(__name__)


@shared_task
def generate_insights_for_user(user_id):
    """Generate insights for a single user."""
    from django.contrib.auth import get_user_model

    User = get_user_model()
    user = User.objects.get(id=user_id)

    insights_data = analyze_user_experiments(user)
    created = 0
    for data in insights_data:
        # Check if a similar insight already exists (avoid duplicates)
        if not ExperimentInsight.objects.filter(
            user=user,
            insight_type=data["insight_type"],
            title=data["title"],
            created_at__gte=timezone.now() - timedelta(hours=24),
        ).exists():
            ExperimentInsight.objects.create(
                user=user,
                insight_type=data["insight_type"],
                title=data["title"],
                description=data["description"],
                confidence=data.get("confidence", 0.0),
                metadata=data.get("metadata", {}),
                related_runs=data.get("related_runs", []),
            )
            created += 1

    return f"Generated {created} insights for {user.username}"


@shared_task
def generate_recommendations_for_user(user_id):
    """Generate next-experiment recommendations for a single user."""
    from django.contrib.auth import get_user_model
    from experiment.models import TrainingRun

    User = get_user_model()
    user = User.objects.get(id=user_id)

    runs = TrainingRun.objects.filter(user=user, status="completed").order_by(
        "-created_at"
    )

    if not runs.exists():
        return

    # Find the best run so far
    best_run = max(
        (r for r in runs if r.final_metrics and r.final_metrics.get("eval_accuracy")),
        key=lambda r: r.final_metrics["eval_accuracy"],
        default=None,
    )
    if not best_run:
        return

    config = best_run.config or {}
    best_accuracy = best_run.final_metrics["eval_accuracy"]

    # Generate a recommendation: vary the attention type or method
    recommendations = []
    tried_attentions = set(
        r.config.get("attention_type")
        for r in runs
        if r.config and r.config.get("attention_type")
    )
    all_attentions = ["full", "galerkin", "linformer", "spectraformer", "padre"]
    for attn in all_attentions:
        if attn not in tried_attentions:
            new_config = {**config, "attention_type": attn}
            recommendations.append(
                {
                    "title": f"Try {attn} attention on {config.get('task_name', 'your task')}",
                    "description": f"Experiment with the '{attn}' attention mechanism. Your best run used '{config.get('attention_type', 'full')}' with accuracy {best_accuracy:.4f}.",
                    "predicted_improvement": 2.0
                    + (5.0 if attn in ["galerkin", "spectraformer"] else 0),
                    "confidence": 0.5,
                    "configuration": new_config,
                    "rationale": f"Other researchers have seen improvements when switching from '{config.get('attention_type', 'full')}' to '{attn}'.",
                    "based_on_runs": [str(best_run.id)],
                }
            )
            break  # Only suggest one new attention type

    created = 0
    for rec_data in recommendations:
        _, is_new = ExperimentRecommendation.objects.get_or_create(
            user=user,
            title=rec_data["title"],
            defaults=rec_data,
        )
        if is_new:
            created += 1

    return f"Generated {created} recommendations for {user.username}"


@shared_task
def generate_weekly_summary(user_id):
    """Generate a weekly experiment summary for a user."""
    from django.contrib.auth import get_user_model
    from experiment.models import TrainingRun

    User = get_user_model()
    user = User.objects.get(id=user_id)

    end = timezone.now().date()
    start = end - timedelta(days=7)

    runs = TrainingRun.objects.filter(
        user=user,
        created_at__date__gte=start,
        created_at__date__lte=end,
    )

    if not runs.exists():
        return

    completed = runs.filter(status="completed")
    best = max(
        (
            r
            for r in completed
            if r.final_metrics and r.final_metrics.get("eval_accuracy")
        ),
        key=lambda r: r.final_metrics["eval_accuracy"],
        default=None,
    )

    # Get insights from this period
    insights = ExperimentInsight.objects.filter(
        user=user,
        created_at__date__gte=start,
    )

    summary = ExperimentSummary.objects.create(
        user=user,
        period_start=start,
        period_end=end,
        total_experiments=runs.count(),
        successful_experiments=completed.count(),
        best_accuracy=best.final_metrics.get("eval_accuracy") if best else None,
        best_loss=best.final_metrics.get("eval_loss") if best else None,
        insights_text="\n".join(f"- {i.title}" for i in insights[:5]),
    )
    return f"Generated summary for {user.username}"
