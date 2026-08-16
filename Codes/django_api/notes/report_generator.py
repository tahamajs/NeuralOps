"""
Auto-generate rich Markdown research notes from training run data.
Inspired by W&B Reports and tfruns auto-reporting.
"""

import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from django.conf import settings
from experiment.models import TrainingRun, TrainingSnapshot
from jobscheduler.models import Job
from notes.models import ResearcherNote


class ExperimentReportGenerator:
    """
    Generates comprehensive Markdown experiment reports from TrainingRun data.

    Usage:
        generator = ExperimentReportGenerator(run)
        note = generator.generate(author=user, save=True)
    """

    def __init__(self, run: TrainingRun):
        self.run = run
        self.config = run.config or {}
        self.metrics = run.final_metrics or {}
        self.history = run.training_history or {}
        self.snapshots = list(TrainingSnapshot.objects.filter(run=run).order_by("step"))

    def generate(
        self, author, save: bool = True, template_name: Optional[str] = None
    ) -> ResearcherNote:
        """Generate and optionally save the report."""
        content = self._build_content()

        job_instance = None
        if isinstance(self.run, Job):
            job_instance = self.run
        else:
            try:
                job_instance = Job.objects.filter(id=self.run.id).first()
            except Exception:
                job_instance = None

        note = ResearcherNote(
            author=author,
            run=job_instance,
            note_type="training_report",
            title=f"Training Report: {self.config.get('model_name', 'Model')} → {self.config.get('task_name', 'Task')}",
            content=content,
            tags=["auto-generated", "training-report", self.run.status],
            is_public=False,
            is_pinned=False,
            metadata={
                "hyperparameters": {
                    k: v
                    for k, v in self.config.items()
                    if k not in ("output_dir", "name")
                },
                "metrics_snapshot": self.metrics,
                "model_version": str(self.run.id),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "charts": self._generate_charts_data(),
            },
        )

        if save:
            note.save()
        return note

    def _build_content(self) -> str:
        """Build complete Markdown report."""
        sections = [
            self._header(),
            self._configuration(),
            self._metrics_summary(),
            self._training_curves(),
            self._analysis(),
            self._artifacts(),
            self._next_steps(),
        ]
        return "\n\n".join(sections)

    def _header(self) -> str:
        status_icon = {
            "completed": "✅",
            "running": "🔄",
            "failed": "❌",
            "stopped": "⏹️",
        }.get(self.run.status, "❓")
        return f"""# {status_icon} Experiment Report: {self.config.get('model_name', 'Unknown Model')}

**Task:** {self.config.get('task_name', 'N/A')}  
**Status:** {self.run.status}  
**Run ID:** `{self.run.id}`  
**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  
**Duration:** {self._compute_duration()}
"""

    def _configuration(self) -> str:
        cfg = self.config
        rows = [
            f"| **Base Model** | `{cfg.get('model_name', 'N/A')}` |",
            f"| **Task** | `{cfg.get('task_name', 'N/A')}` |",
            f"| **Attention Type** | `{cfg.get('attention_type', 'full')}` |",
            f"| **Training Method** | `{cfg.get('method', 'baseline')}` |",
            f"| **Compressor** | `{cfg.get('compressor', 'none')}` |",
            f"| **Epochs** | {cfg.get('epochs', 'N/A')} |",
            f"| **Batch Size** | {cfg.get('batch_size', 'N/A')} |",
            f"| **Learning Rate** | {cfg.get('learning_rate', 'N/A')} |",
            f"| **Max Sequence Length** | {cfg.get('max_seq_length', 'N/A')} |",
            f"| **Compression Rank** | {cfg.get('compression_rank', 'N/A')} |",
            f"| **Git Commit** | `{cfg.get('git_commit', 'N/A')}` |",
        ]
        return f"""## ⚙️ Configuration

| Parameter | Value |
|-----------|-------|
{chr(10).join(rows)}
"""

    def _metrics_summary(self) -> str:
        m = self.metrics
        best_loss = (
            min(self.history.get("losses", [0]))
            if self.history.get("losses")
            else "N/A"
        )
        best_acc = (
            max(self.history.get("eval_accuracies", [0]))
            if self.history.get("eval_accuracies")
            else "N/A"
        )

        # Compute trend
        losses = self.history.get("losses", [])
        trend = "Unknown"
        if len(losses) >= 5:
            if losses[-1] < losses[0] * 0.9:
                trend = "📉 Improving"
            elif losses[-1] > losses[0] * 1.1:
                trend = "📈 Worsening"
            else:
                trend = "➡️ Stable"

        return f"""## 📊 Results Summary

| Metric | Value |
|--------|-------|
| **Final Accuracy** | {m.get('eval_accuracy', 'N/A')} |
| **Final F1** | {m.get('eval_f1', 'N/A')} |
| **Final Loss** | {m.get('eval_loss', 'N/A')} |
| **Best Accuracy** | {best_acc} |
| **Best Loss** | {best_loss} |
| **Training Trend** | {trend} |
| **Parameters Saved** | {m.get('saved_parameters', 'N/A')} |
| **Compression Ratio** | {m.get('compression_ratio', 'N/A')} |
"""

    def _training_curves(self) -> str:
        """Generate training curve chart and embed as base64."""
        chart_b64 = self._generate_loss_chart_b64()
        if chart_b64:
            return f"""## 📈 Training Curves

![Loss & Accuracy](data:image/png;base64,{chart_b64})
"""
        return "## 📈 Training Curves\n\n*(No training history data available)*"

    def _analysis(self) -> str:
        """Add analysis section if available."""
        try:
            analysis = self.run.analysis
            if analysis:
                issues = (
                    ", ".join(analysis.suggestions[:3])
                    if analysis.suggestions
                    else "None"
                )
                return f"""## 🔍 Training Analysis

**Status:** {analysis.status}  
**Issues Detected:** {issues}

**Suggestions:**
{chr(10).join(f'- {s}' for s in analysis.suggestions[:5])}
"""
        except Exception:
            pass
        return ""

    def _artifacts(self) -> str:
        """List linked artifacts."""
        lines = []
        model_dir = Path(settings.MEDIA_ROOT) / str(self.run.id) / "model"
        if model_dir.exists():
            files = list(model_dir.glob("*"))
            if files:
                lines.append("**Model Artifacts:**")
                for f in files[:10]:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    lines.append(f"- `{f.name}` ({size_mb:.1f} MB)")
        if lines:
            return "## 📁 Artifacts\n\n" + "\n".join(lines)
        return ""

    def _next_steps(self) -> str:
        return """## 🚀 Next Steps

- [ ] Review metrics and identify improvement areas
- [ ] Compare with previous runs
- [ ] Consider hyperparameter tuning: `learning_rate`, `batch_size`
- [ ] Try different attention types or compression methods
- [ ] Document findings in a follow-up note

---
*Report auto-generated by NeuralOps Experiment Tracker*
"""

    def _compute_duration(self) -> str:
        if self.run.started_at and self.run.completed_at:
            delta = self.run.completed_at - self.run.started_at
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        return "N/A"

    def _generate_loss_chart_b64(self) -> Optional[str]:
        """Generate a dual-axis loss/accuracy chart as base64 PNG."""
        snapshots = self.snapshots
        if not snapshots:
            return None

        steps = [s.step for s in snapshots]
        losses = [s.train_loss for s in snapshots if s.train_loss is not None]
        eval_losses = [s.eval_loss for s in snapshots if s.eval_loss is not None]
        eval_accs = [s.eval_accuracy for s in snapshots if s.eval_accuracy is not None]

        if not losses and not eval_losses:
            return None

        fig, ax1 = plt.subplots(figsize=(10, 5))

        if losses:
            ax1.plot(
                steps[: len(losses)],
                losses,
                "tab:orange",
                linewidth=1.5,
                label="Train Loss",
            )
        if eval_losses:
            eval_steps = steps[: len(eval_losses)]
            ax1.plot(
                eval_steps,
                eval_losses,
                "tab:red",
                linestyle="--",
                marker="o",
                markersize=2,
                label="Eval Loss",
            )
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        if eval_accs:
            ax2 = ax1.twinx()
            eval_steps = steps[: len(eval_accs)]
            ax2.plot(
                eval_steps,
                eval_accs,
                "tab:green",
                linestyle=":",
                marker="x",
                markersize=2,
                label="Eval Accuracy",
            )
            ax2.set_ylabel("Accuracy")
            ax2.legend(loc="upper right")

        plt.title(
            f'Training Curves — {self.config.get("model_name", "Model")} / {self.config.get("task_name", "Task")}'
        )
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _generate_charts_data(self) -> list:
        """Return chart data for frontend rendering."""
        snapshots = self.snapshots
        if not snapshots:
            return []

        return [
            {
                "name": "loss",
                "type": "line",
                "data": [
                    {"step": s.step, "value": s.train_loss}
                    for s in snapshots
                    if s.train_loss is not None
                ],
            },
            {
                "name": "eval_accuracy",
                "type": "line",
                "data": [
                    {"step": s.step, "value": s.eval_accuracy}
                    for s in snapshots
                    if s.eval_accuracy is not None
                ],
            },
        ]


def generate_experiment_report(
    run_id: str, author, save: bool = True
) -> Optional[ResearcherNote]:
    """Convenience function to generate a report for a training run."""
    try:
        run = TrainingRun.objects.get(id=run_id)
    except TrainingRun.DoesNotExist:
        return None

    generator = ExperimentReportGenerator(run)
    return generator.generate(author=author, save=save)
