"""
Celery task definitions for asynchronous training, evaluation, and compression.
Replaces raw threading with production‑grade task queue.
"""

import os
import traceback
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from .models import TrainingRun


@shared_task(bind=True, max_retries=2, default_retry_delay=60)
def run_training_task(self, run_id: str):
    """Celery task for training a model."""
    from .runners import run_training

    run = TrainingRun.objects.get(id=run_id)
    run.status = "running"
    run.started_at = timezone.now()
    run.save(update_fields=["status", "started_at"])

    try:
        run_training(run)
        run.status = "completed"
        run.completed_at = timezone.now()
        run.save(update_fields=["status", "completed_at"])
    except Exception as e:
        run.status = "failed"
        run.logs = (run.logs or "") + f"\nFAILED: {e}\n{traceback.format_exc()}"
        run.save(update_fields=["status", "logs"])
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=1, default_retry_delay=30)
def run_evaluation_task(self, run_id: str):
    """Celery task for evaluating a trained model."""
    from .runners import run_evaluation

    run = TrainingRun.objects.get(id=run_id)
    run.status = "running"
    run.save(update_fields=["status"])

    try:
        run_evaluation(run)
        run.status = "completed"
    except Exception as e:
        run.status = "failed"
        run.logs = (run.logs or "") + f"\nEVAL FAILED: {e}"
        run.save(update_fields=["status", "logs"])
        raise self.retry(exc=e)
    finally:
        run.completed_at = timezone.now()
        run.save(update_fields=["completed_at", "status"])


@shared_task(bind=True, max_retries=1)
def run_compression_task(self, run_id: str):
    """Celery task for compressing a model."""
    from .runners import run_compression

    run = TrainingRun.objects.get(id=run_id)
    run.status = "running"
    run.save(update_fields=["status"])

    try:
        run_compression(run)
        run.status = "completed"
    except Exception as e:
        run.status = "failed"
        run.logs = (run.logs or "") + f"\nCOMPRESSION FAILED: {e}"
        run.save(update_fields=["status", "logs"])
    finally:
        run.completed_at = timezone.now()
        run.save(update_fields=["completed_at", "status"])


@shared_task(bind=True)
def auto_create_model_version(self, run_id: str):
    """Auto‑create a model version record after successful training."""
    from model_versioning.models import ModelVersion

    run = TrainingRun.objects.get(id=run_id)
    model_dir = Path(settings.MEDIA_ROOT) / str(run.id) / "model"

    try:
        size_mb = 0.0
        pth_files = list(model_dir.glob("*.pth"))
        if pth_files:
            size_mb = os.path.getsize(pth_files[0]) / 1e6

        latest = (
            ModelVersion.objects.filter(run=run).order_by("-version_number").first()
        )
        new_num = (latest.version_number + 1) if latest else 1
        ModelVersion.objects.create(
            run=run,
            version=f"v{new_num}",
            version_number=new_num,
            artifact_path=str(model_dir),
            artifact_size_mb=round(size_mb, 2),
            metrics=run.final_metrics or {},
        )
        return {"status": "created", "version": f"v{new_num}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


from agents.models import AgentSession
from celery import shared_task


@shared_task(bind=True, max_retries=0)
def run_autonomous_agent(self, session_id: str, user_id: int):
    session = AgentSession.objects.get(id=session_id)
    session.status = "active"
    session.save()
    while session.cycles_completed < session.max_cycles and session.status == "active":
        from agents.agent_runner import run_agent_cycle

        run_agent_cycle(session, user_id)
        session.refresh_from_db()
        if session.total_llm_cost_cents >= session.max_llm_cost_cents:
            session.status = "review"
            session.save()
            break
    session.status = "completed" if session.status == "active" else session.status
    session.save()


# agents/tasks.py — add this


@shared_task(bind=True, max_retries=2)
def execute_research_plan(self, plan_id: str):
    """Execute a research plan step by step."""
    plan = ResearchPlan.objects.get(id=plan_id)
    try:
        for step in plan.steps:
            # Search for papers
            results = search_papers(query=step["task"], limit=5)
            for paper in results:
                ResearchFinding.objects.create(
                    plan=plan,
                    source_url=paper["url"],
                    source_title=paper["title"],
                    source_type=paper.get("source", "web"),
                    credibility_score=_evaluate_source(paper),
                    summary=paper.get("abstract", "")[:500],
                    relevance_score=0.7,
                )
            step["status"] = "completed"
            plan.steps = plan.steps  # trigger save
            plan.save()

        # Synthesize report
        findings = plan.findings.all()
        report_md = _synthesize_report(plan.objective, findings)
        ResearchReport.objects.create(plan=plan, content_markdown=report_md)
        plan.status = "completed"
        plan.save()
    except Exception as e:
        plan.status = "failed"
        plan.save()
        raise self.retry(exc=e)


def _evaluate_source(paper: dict) -> float:
    """Score source credibility (citation count, venue, etc.)."""
    citations = paper.get("citation_count", 0)
    venue = paper.get("venue", "")
    score = min(1.0, citations / 1000.0)
    if "nature" in venue.lower() or "neurips" in venue.lower():
        score += 0.2
    return min(1.0, score)


def _synthesize_report(objective: str, findings) -> str:
    """Use LLM to synthesize findings into a structured report."""
    # In production: call OpenAI / Anthropic API
    report = f"# Research Report: {objective}\n\n"
    for f in findings:
        report += f"## {f.source_title}\n- Credibility: {f.credibility_score:.2f}\n- Summary: {f.summary}\n\n"
    return report
