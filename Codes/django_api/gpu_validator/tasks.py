"""Celery tasks for the validator system."""

import logging

from celery import shared_task

from .models import ValidatorNode, Verification, VerificationTask

logger = logging.getLogger(__name__)


@shared_task
def auto_assign_verification_tasks():
    """
    Automatically assign pending verification tasks to the most reliable
    available validators. Called periodically.
    """
    pending_tasks = VerificationTask.objects.filter(
        status__in=["pending", "in_progress"]
    )

    for task in pending_tasks:
        # How many more validators do we need?
        current_count = task.verifications.count()
        needed = task.required_validators - current_count
        if needed <= 0:
            continue

        # Get active validators not already assigned to this task
        assigned_ids = task.verifications.values_list("validator_id", flat=True)
        available = (
            ValidatorNode.objects.filter(status="active")
            .exclude(id__in=assigned_ids)
            .order_by("-accuracy_score")[:needed]
        )

        for validator in available:
            Verification.objects.get_or_create(
                task=task,
                validator=validator,
                defaults={"result": "valid", "confidence": 0.0},  # placeholder
            )

        # Update task status
        if task.status == "pending" and task.verifications.count() > 0:
            task.status = "in_progress"
            task.save(update_fields=["status"])

    return pending_tasks.count()


@shared_task
def update_validator_accuracy_scores():
    """Daily task to recalculate all validator accuracy scores."""
    for node in ValidatorNode.objects.filter(status="active"):
        node.update_accuracy()
