"""
Celery tasks for the GPU Marketplace.
"""

import logging

from celery import shared_task
from django.utils import timezone

from .matching_engine import match_all_orders
from .models import ComputeJob, GPUListing, JobEvent

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def run_matching_cycle(self):
    """
    Periodic task: run the matching engine.
    Called by Celery beat every 30 seconds.
    """
    try:
        return match_all_orders()
    except Exception as exc:
        logger.exception("Matching cycle failed")
        raise self.retry(exc=exc)


@shared_task
def update_dynamic_pricing():
    """Periodic task: update dynamic prices based on demand/supply."""
    for listing in GPUListing.objects.filter(status="online"):
        listing.update_dynamic_price()
    return GPUListing.objects.filter(status="online").count()


@shared_task
def expire_stale_orders():
    """Cancel orders that have passed their deadline."""
    count = ComputeOrder.objects.filter(
        status="open", deadline__lt=timezone.now()
    ).update(status="expired")
    return count


@shared_task
def update_reliability_scores():
    """Update reliability scores for all listings."""
    for listing in GPUListing.objects.filter(status__in=["online", "busy"]):
        total = listing.completed_jobs + listing.failed_jobs
        if total > 0:
            success_rate = (listing.completed_jobs / total) * 100
            listing.reliability_score = round(
                listing.reliability_score * 0.9 + success_rate * 0.1, 2
            )
            listing.uptime_percentage = success_rate
            listing.save(update_fields=["reliability_score", "uptime_percentage"])


@shared_task
def provision_job(job_id: str):
    """
    Provision a matched job on the provider's machine.
    In production, this would:
    1. Call the provider's agent via WebSocket/HTTP
    2. Pull the Docker image
    3. Start the container
    4. Stream logs back
    """
    try:
        job = ComputeJob.objects.get(id=job_id)
    except ComputeJob.DoesNotExist:
        return

    job.status = "provisioning"
    job.save(update_fields=["status"])
    JobEvent.objects.create(job=job, event_type="provisioning")

    # Dispatch to provider agent via Channels WebSocket group
    from channels.layers import get_channel_layer
    from asgiref.sync import async_to_sync

    channel_layer = get_channel_layer()
    if channel_layer:
        try:
            async_to_sync(channel_layer.group_send)(
                f"provider_listing_{job.listing.id}",
                {
                    "type": "send_job",
                    "job": {
                        "id": str(job.id),
                        "docker_image": job.docker_image,
                        "command": job.command,
                    }
                }
            )
            logger.info(f"Dispatched job {job.id} to provider listing {job.listing.id}")
        except Exception as e:
            logger.error(f"Failed to dispatch job to provider agent: {e}")

    return {"job_id": job_id, "status": "provisioning"}
