"""
Celery task definitions for asynchronous training, evaluation, and compression.
Replaces raw threading with production‑grade task queue.
"""

import json, traceback, os
from pathlib import Path

import psutil
from celery import shared_task
from django.utils import timezone
from django.conf import settings

from .models import TrainingRun, TrainingSnapshot


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

        latest = ModelVersion.objects.filter(run=run).order_by('-version_number').first()
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
    
from celery import shared_task
from agents.models import AgentSession

@shared_task(bind=True, max_retries=0)
def run_autonomous_agent(self, session_id: str, user_id: int):
    session = AgentSession.objects.get(id=session_id)
    session.status = 'active'
    session.save()
    while session.cycles_completed < session.max_cycles and session.status == 'active':
        from agents.agent_runner import run_agent_cycle
        run_agent_cycle(session, user_id)
        session.refresh_from_db()
        if session.total_llm_cost_cents >= session.max_llm_cost_cents:
            session.status = 'review'
            session.save()
            break
    session.status = 'completed' if session.status == 'active' else session.status
    session.save()