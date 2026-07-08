"""Celery tasks for federation — coordinate distributed training."""

import logging
import random
import time

from celery import shared_task
from django.utils import timezone

from .models import FederationTask

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2, default_retry_delay=60)
def run_federation_task(self, task_id: str):
    """
    Simulated distributed training run across nodes.
    In production, this would orchestrate actual training across provider agents.
    """
    try:
        task = FederationTask.objects.get(id=task_id)
    except FederationTask.DoesNotExist:
        return

    if task.status != "running":
        return

    shards = task.shards.filter(status="pending")
    for shard in shards:
        shard.status = "running"
        shard.started_at = timezone.now()
        shard.save(update_fields=["status", "started_at"])

    # Simulate training steps
    total_steps = task.total_steps or 1000
    for step in range(1, total_steps + 1, 10):
        task.refresh_from_db()
        if task.status in ("canceled", "failed"):
            break

        # Simulate progress on a random shard
        active_shards = task.shards.filter(status="running")
        if active_shards.exists():
            shard = random.choice(active_shards)
            shard.steps_completed += 10
            shard.samples_processed += task.batch_size_per_node * 10
            shard.avg_loss = round(random.uniform(0.1, 1.0), 4)
            shard.save(
                update_fields=["steps_completed", "samples_processed", "avg_loss"]
            )

        # Update task progress
        task.current_step = step
        task.progress_percent = min(100.0, (step / total_steps) * 100)
        task.total_loss = round(random.uniform(0.1, 1.0), 4)
        task.save(update_fields=["current_step", "progress_percent", "total_loss"])

        time.sleep(0.1)  # Simulate work

    # Mark shards as completed
    for shard in task.shards.filter(status="running"):
        shard.status = "completed"
        shard.completed_at = timezone.now()
        shard.save(update_fields=["status", "completed_at"])

    # Complete the task
    task.status = "completed"
    task.completed_at = timezone.now()
    task.save(update_fields=["status", "completed_at"])

    from .models import FederationResult

    FederationResult.objects.create(
        task=task,
        final_loss=task.total_loss,
        total_time_seconds=(
            (task.completed_at - task.started_at).total_seconds()
            if task.started_at
            else 0
        ),
        metrics={"simulated": True},
    )
