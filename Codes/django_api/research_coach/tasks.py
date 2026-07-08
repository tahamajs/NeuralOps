"""
Celery tasks for background analysis and recommendation generation.
"""

import logging

from celery import shared_task
from django.contrib.auth import get_user_model
from django.db.models import Avg

from .chroma_store import add_experiment_memory
from .models import AnalysisResult, ExperimentMemory, Recommendation

User = get_user_model()
logger = logging.getLogger(__name__)


@shared_task
def run_periodic_analysis(user_id=None):
    """Run analysis on experiments and generate insights."""
    if user_id:
        users = User.objects.filter(id=user_id)
    else:
        users = User.objects.filter(is_active=True)

    from experiment.models import TrainingRun

    for user in users:
        runs = TrainingRun.objects.filter(user=user).order_by("-created_at")
        if not runs.exists():
            continue

        # ── Cluster analysis ──────────────────────────────────
        completed = runs.filter(status="completed")
        if completed.count() >= 3:
            avg_accuracy = completed.aggregate(a=Avg("final_metrics__eval_accuracy"))[
                "a"
            ]
            best_run = completed.order_by("-final_metrics__eval_accuracy").first()
            worst_run = completed.order_by("final_metrics__eval_accuracy").first()

            content = f"""You have completed {completed.count()} experiments.

Best run: {best_run.name or best_run.id} with accuracy {best_run.final_metrics.get('eval_accuracy', 'N/A')}
Worst run: {worst_run.name or worst_run.id} with accuracy {worst_run.final_metrics.get('eval_accuracy', 'N/A')}

Average accuracy across all completed runs: {avg_accuracy:.4f}
"""
            AnalysisResult.objects.create(
                user=user,
                analysis_type="cluster",
                title=f"Experiment Cluster Analysis ({completed.count()} runs)",
                content=content,
            )

        # ── Trend detection ───────────────────────────────────
        if completed.count() >= 5:
            recent = completed[:5]
            accuracies = [r.final_metrics.get("eval_accuracy", 0) or 0 for r in recent]
            if len(accuracies) >= 3 and accuracies[0] > accuracies[-1]:
                AnalysisResult.objects.create(
                    user=user,
                    analysis_type="trend",
                    title="📈 Positive Trend Detected!",
                    content="Your last 5 experiments show an improving accuracy trend. Whatever you're doing, keep going! Consider pushing further with harder tasks or more aggressive compression.",
                )

        # ── Motivation ────────────────────────────────────────
        if completed.count() > 0:
            AnalysisResult.objects.create(
                user=user,
                analysis_type="motivation",
                title="💪 Keep Up the Great Work!",
                content=f"You've completed {completed.count()} experiments so far. Every run teaches you something new. The best researchers iterate fast — and you're doing exactly that!",
            )

        # ── Recommendations ───────────────────────────────────
        if best_run and best_run.final_metrics:
            acc = best_run.final_metrics.get("eval_accuracy", 0) or 0
            config = best_run.config or {}
            if acc < 0.7:
                Recommendation.objects.create(
                    user=user,
                    recommendation_type="architecture",
                    title="Try a more powerful model",
                    description=f"Your best accuracy is {acc:.4f}. Consider upgrading from {config.get('model_name', 'current')} to a larger variant.",
                    confidence=0.8,
                )
            elif acc < 0.85:
                Recommendation.objects.create(
                    user=user,
                    recommendation_type="hyperparameter",
                    title="Fine-tune your hyperparameters",
                    description="Your accuracy is decent but could be better. Try adjusting learning rate, batch size, or epochs.",
                    confidence=0.6,
                )
            else:
                Recommendation.objects.create(
                    user=user,
                    recommendation_type="compression",
                    title="Try compressing your model",
                    description="Your accuracy is excellent! Now try applying model compression to reduce size while maintaining performance.",
                    confidence=0.7,
                    action_data={"compressor": "erc", "allowed_error": 0.02},
                )

    return f"Analysis completed for {users.count()} users"


@shared_task
def index_experiment_in_memory(run_id):
    """Index a single experiment in ChromaDB."""
    from experiment.models import TrainingRun

    try:
        run = TrainingRun.objects.get(id=run_id)
    except TrainingRun.DoesNotExist:
        return "Run not found"

    # Build summary text
    config = run.config or {}
    metrics = run.final_metrics or {}
    summary = f"Experiment: {run.name or run.id}. "
    summary += f"Model: {config.get('model_name', 'unknown')}. "
    summary += f"Task: {config.get('task_name', 'unknown')}. "
    summary += f"Attention: {config.get('attention_type', 'full')}. "
    summary += f"Method: {config.get('method', 'baseline')}. "
    summary += f"Accuracy: {metrics.get('eval_accuracy', 'N/A')}. "
    summary += f"Loss: {metrics.get('eval_loss', 'N/A')}."

    add_experiment_memory(
        str(run.id),
        summary,
        {
            "run_id": str(run.id),
            "user_id": run.user_id,
            "status": run.status,
        },
    )

    ExperimentMemory.objects.update_or_create(
        run=run,
        defaults={
            "chroma_id": str(run.id),
            "summary_text": summary,
        },
    )

    return f"Indexed experiment {run.id}"
