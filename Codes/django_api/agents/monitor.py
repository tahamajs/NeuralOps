"""
Autonomous Training Monitor (Celery tasks).
"""

import matplotlib

matplotlib.use("Agg")
from agents.paper_recommender import get_paper_recommendations_for_run
from agents.tools import analyze_training
from celery import shared_task
from experiment.models import TrainingRun
from notes.models import ResearcherNote


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def monitor_training_run(
    self,
    run_id: str,
    user_id: str,
    check_interval_ratio: float = 0.25,
    last_checked_step: int = 0,
    auto_stop_enabled: bool = True,
    note_generation_enabled: bool = True,
):
    try:
        run = TrainingRun.objects.get(id=run_id)
    except TrainingRun.DoesNotExist:
        return {"status": "error", "error": "Run not found"}

    if run.status not in ("running", "queued"):
        return {"status": "skipped", "reason": f"Run status: {run.status}"}

    total_steps = run.total_steps or 100
    current_step = run.current_step or 0

    checkpoints = [int(total_steps * r) for r in [0.25, 0.50, 0.75, 1.0]]
    next_checkpoint = None
    for cp in checkpoints:
        if cp > last_checked_step:
            next_checkpoint = cp
            break
    if next_checkpoint is None:
        return {"status": "done", "message": "All checkpoints completed"}

    if current_step < next_checkpoint - 1:
        monitor_training_run.apply_async(
            args=[
                run_id,
                user_id,
                check_interval_ratio,
                last_checked_step,
                auto_stop_enabled,
                note_generation_enabled,
            ],
            countdown=30,
        )
        return {"status": "rescheduled", "next_checkpoint": next_checkpoint}

    # Analyse
    result = analyze_training(run_id=str(run.id))
    analysis = result.get("analysis", result)

    # Generate note
    if note_generation_enabled:
        note_content = _build_report(run, analysis, next_checkpoint)
        ResearcherNote.objects.create(
            author_id=user_id,
            title=f"📊 Training Checkpoint — Step {next_checkpoint}/{total_steps}",
            content=note_content,
            tags=[
                "auto-generated",
                "training-report",
                analysis.get("status", "unknown"),
            ],
            is_public=False,
            pinned=False,
            metadata={"run_id": run_id, "checkpoint": next_checkpoint},
        )

    # Auto‑stop
    if auto_stop_enabled and analysis.get("status") == "diverging":
        from jobscheduler.scheduler import scheduler

        scheduler.stop_job(run_id)
        run.status = "stopped"
        run.save(update_fields=["status"])
        ResearcherNote.objects.create(
            author_id=user_id,
            title="🛑 Auto‑Stopped — Divergence Detected",
            content=f"Run `{run_id}` was auto‑stopped due to divergence.",
            tags=["auto-generated", "alert", "diverging"],
            is_public=False,
            pinned=True,
            metadata={"run_id": run_id},
        )

    # Paper recommendations
    if note_generation_enabled:
        recommendations = get_paper_recommendations_for_run(run, limit=3)
        if recommendations:
            _append_recommendations(user_id, run_id, recommendations)

    # Schedule next
    if next_checkpoint < total_steps:
        monitor_training_run.apply_async(
            args=[
                run_id,
                user_id,
                check_interval_ratio,
                next_checkpoint,
                auto_stop_enabled,
                note_generation_enabled,
            ],
            countdown=30,
        )

    return {"status": "checkpoint", "step": next_checkpoint}


def _build_report(run, analysis, checkpoint):
    ms = analysis.get("metrics_summary", {})
    config = run.config or {}
    return f"""## 📈 Training Checkpoint Report

**Run ID:** `{run.id}`
**Checkpoint:** Step {checkpoint} / {run.total_steps or '?'}
**Status:** {analysis.get('status', 'unknown')}

### Config
- Model: `{config.get('model_name', '?')}`
- Task: `{config.get('task_name', '?')}`
- Attention: `{config.get('attention_type', 'full')}`

### Issues
{', '.join(analysis.get('issues', ['none']))}

### Suggestions
{chr(10).join('- ' + s for s in analysis.get('suggestions', []))}

### Metrics
| Metric | Value |
|--------|-------|
| Current Loss | {ms.get('current_loss', '—')} |
| Best Loss | {ms.get('best_loss', '—')} |
| Best Accuracy | {ms.get('best_accuracy', '—')} |
| Trend | {ms.get('trend', '—')} |
"""


def _append_recommendations(user_id, run_id, recommendations):
    note = (
        ResearcherNote.objects.filter(
            author_id=user_id,
            metadata__run_id=run_id,
            tags__contains=["auto-generated"],
        )
        .order_by("-created_at")
        .first()
    )
    if note:
        rec_text = "\n\n### 📚 Related Papers\n"
        for i, p in enumerate(recommendations, 1):
            rec_text += f"{i}. **[{p.get('title','Untitled')}]({p.get('url','#')})**\n"
            rec_text += f"   {p.get('abstract','')[:200]}...\n\n"
        note.content += rec_text
        note.save(update_fields=["content"])
