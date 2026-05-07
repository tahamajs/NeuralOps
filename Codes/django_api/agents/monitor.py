# agents/monitor.py
"""
Autonomous Training Monitor – watches runs at 1/4 intervals,
detects issues (overfitting, plateau, divergence), generates notes,
and can auto‑stop failed runs.
"""
import io
import base64
import json
import time
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from celery import shared_task
from django.utils import timezone
from django.conf import settings

from experiment.models import TrainingRun, TrainingSnapshot
from notes.models import ResearcherNote
from agents.tools import analyze_training, suggest_next_experiment
from agents.paper_recommender import get_paper_recommendations_for_run


@shared_task(bind=True)
def monitor_training_run(self, run_id: str, user_id: str, check_interval_ratio: float = 0.25,
                         last_checked_step: int = 0, auto_stop_enabled: bool = True,
                         note_generation_enabled: bool = True):
    """
    Monitors a training run at every 'check_interval_ratio' (default 1/4) of total steps.

    At each checkpoint:
      1. Analyzes training (overfitting, plateau, divergence, underfitting).
      2. Generates a note with plots and metrics (if note_generation_enabled).
      3. Optionally auto‑stops if a critical issue is detected.
      4. Searches related papers and adds recommendations.
      5. Schedules the next checkpoint.
    """
    try:
        run = TrainingRun.objects.get(id=run_id)
    except TrainingRun.DoesNotExist:
        return

    if run.status not in ('running', 'queued'):
        return

    total_steps = run.total_steps or 100
    current_step = run.current_step or 0

    # Calculate next checkpoint step
    checkpoints = [int(total_steps * ratio) for ratio in [0.25, 0.50, 0.75, 1.0]]
    next_checkpoint = None
    for cp in checkpoints:
        if cp > last_checked_step:
            # Only trigger if we're near or past the checkpoint
            next_checkpoint = cp
            break

    if next_checkpoint is None:
        return  # all checkpoints done

    # If current step hasn't reached checkpoint yet, reschedule
    if current_step < next_checkpoint - 1:
        # Re‑queue with a delay
        monitor_training_run.apply_async(
            args=[run_id, user_id, check_interval_ratio, last_checked_step,
                  auto_stop_enabled, note_generation_enabled],
            countdown=30,
        )
        return

    # ── 1. Analyze Training ──
    analysis = analyze_training({"run_id": run_id})

    # ── 2. Generate Note (if enabled) ──
    if note_generation_enabled:
        note_content = _build_training_note(run, analysis, next_checkpoint)
        ResearcherNote.objects.create(
            run_id=None,  # or link to the Job/TrainingRun model
            author_id=user_id,
            title=f"📊 Training Checkpoint – Step {next_checkpoint}/{total_steps}",
            content=note_content,
            tags=["auto-generated", "training-report", run.status],
            is_public=False,
            pinned=False,
            metadata={
                "run_id": run_id,
                "checkpoint": next_checkpoint,
                "analysis": analysis,
            },
        )

    # ── 3. Auto‑Stop Logic ──
    critical_statuses = ('diverging',)
    if auto_stop_enabled and analysis['status'] in critical_statuses:
        from jobscheduler.scheduler import scheduler
        scheduler.stop_job(str(run_id))
        run.status = 'stopped'
        run.save(update_fields=['status'])

        # Create an alert note
        ResearcherNote.objects.create(
            author_id=user_id,
            title=f"🛑 Auto‑Stopped – {analysis['status']} detected",
            content=_build_alert_note(run, analysis),
            tags=["auto-generated", "alert", analysis['status']],
            is_public=False,
            pinned=True,
            metadata={"run_id": run_id, "analysis": analysis},
        )

    # ── 4. Paper Recommendations ──
    if note_generation_enabled:
        recommendations = get_paper_recommendations_for_run(run, limit=3)
        if recommendations:
            _append_recommendations_to_latest_note(user_id, run_id, recommendations)

    # ── 5. Schedule next checkpoint ──
    if next_checkpoint < total_steps:
        monitor_training_run.apply_async(
            args=[run_id, user_id, check_interval_ratio, next_checkpoint,
                  auto_stop_enabled, note_generation_enabled],
            countdown=30,
        )

    return {
        "run_id": run_id,
        "checkpoint": next_checkpoint,
        "analysis": analysis,
        "auto_stopped": auto_stop_enabled and analysis['status'] in critical_statuses,
    }


def _build_training_note(run, analysis, checkpoint) -> str:
    """Build a rich Markdown note with charts."""
    snapshots = TrainingSnapshot.objects.filter(run=run).order_by('step')
    steps = [s.step for s in snapshots]
    losses = [s.train_loss for s in snapshots if s.train_loss is not None]
    eval_losses = [s.eval_loss for s in snapshots if s.eval_loss is not None]
    eval_accs = [s.eval_accuracy for s in snapshots if s.eval_accuracy is not None]

    config = run.config or {}
    metrics = run.final_metrics or {}

    # Generate loss chart as base64
    chart_b64 = _generate_loss_chart_b64(steps, losses, eval_losses, eval_accs)

    note = f"""## 📈 Training Checkpoint Report

**Run ID:** `{run.id}`
**Checkpoint:** Step {checkpoint} / {run.total_steps or '?'}
**Status:** {analysis['status']}
**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

### ⚙️ Configuration
- Model: `{config.get('model_name', 'N/A')}`
- Task: `{config.get('task_name', 'N/A')}`
- Attention: `{config.get('attention_type', 'full')}`
- Method: `{config.get('method', 'baseline')}`
- Epochs: `{config.get('epochs', 'N/A')}`
- Batch Size: `{config.get('batch_size', 'N/A')}`
- Learning Rate: `{config.get('learning_rate', 'N/A')}`

### 🔬 Analysis
"""
    if analysis.get('issues'):
        note += f"\n**Issues Detected:** {', '.join(analysis['issues'])}\n"
    else:
        note += "\n✅ No issues detected – training is healthy.\n"

    if analysis.get('suggestions'):
        note += "\n**Suggestions:**\n"
        for s in analysis['suggestions']:
            note += f"- {s}\n"

    note += f"""
### 📊 Metrics Summary
| Metric | Value |
|--------|-------|
| Current Loss | {analysis['metrics_summary'].get('current_loss', '—')} |
| Best Loss | {analysis['metrics_summary'].get('best_loss', '—')} |
| Best Accuracy | {analysis['metrics_summary'].get('best_accuracy', '—')} |
| Total Steps | {analysis['metrics_summary'].get('total_steps', '—')} |
| Trend | {analysis['metrics_summary'].get('trend', '—')} |

### 📉 Training Curves
![Loss Chart](data:image/png;base64,{chart_b64})
"""
    return note


def _build_alert_note(run, analysis) -> str:
    """Build an alert note when auto‑stop triggers."""
    return f"""## 🛑 AUTO‑STOP TRIGGERED

**Run ID:** `{run.id}`
**Status:** {analysis['status']}
**Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

### Issues
{', '.join(analysis.get('issues', ['unknown']))}

### Suggestions
""" + '\n'.join(f"- {s}" for s in analysis.get('suggestions', []))


def _generate_loss_chart_b64(steps, losses, eval_losses, eval_accs):
    """Generate a loss chart and return as base64."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color='tab:orange')
    if steps and losses:
        ax1.plot(steps[:len(losses)], losses, color='tab:orange', label='Train Loss', linewidth=1.5)
    if eval_losses:
        eval_steps = steps[:len(eval_losses)] if len(steps) >= len(eval_losses) else list(range(1, len(eval_losses)+1))
        ax1.plot(eval_steps, eval_losses, color='tab:red', label='Eval Loss', linestyle='--', marker='o', markersize=2)
    ax1.tick_params(axis='y', labelcolor='tab:orange')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)

    if eval_accs:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:green')
        eval_steps = steps[:len(eval_accs)] if len(steps) >= len(eval_accs) else list(range(1, len(eval_accs)+1))
        ax2.plot(eval_steps, eval_accs, color='tab:green', label='Eval Accuracy', linestyle=':', marker='x', markersize=2)
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.legend(loc='upper right')

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _append_recommendations_to_latest_note(user_id, run_id, recommendations):
    """Append paper recommendations to the latest auto‑generated note."""
    note = ResearcherNote.objects.filter(
        author_id=user_id,
        metadata__run_id=run_id,
        tags__contains=["auto-generated"],
    ).order_by('-created_at').first()

    if not note:
        return

    rec_text = "\n\n### 📚 Related Papers\n"
    for i, paper in enumerate(recommendations, 1):
        rec_text += f"{i}. **[{paper.get('title', 'Untitled')}]({paper.get('url', '#')})**\n"
        rec_text += f"   - {paper.get('abstract', 'No abstract')[:200]}...\n"
        if paper.get('authors'):
            rec_text += f"   - Authors: {', '.join(paper['authors'][:3])}\n"
        rec_text += "\n"

    note.content += rec_text
    note.save(update_fields=['content'])