"""Auto-create usage records when jobs complete, and update indices."""

import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

logger = logging.getLogger(__name__)


@receiver(post_save, sender="gpu_marketplace.ComputeJob")
def create_usage_record_on_completion(sender, instance, created, **kwargs):
    """When a compute job completes, create a UsageRecord for billing."""
    if instance.status == "completed" and instance.hours_consumed:
        from .models import UsageRecord

        UsageRecord.objects.create(
            user=instance.consumer,
            job=instance,
            gpu_model=instance.listing.gpu_model,
            hours_consumed=float(instance.hours_consumed),
            cost_usd=instance.total_cost_usd or 0,
            platform_fee_usd=0,  # placeholder
        )
