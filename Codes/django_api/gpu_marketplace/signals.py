from django.db.models.signals import post_save
from django.dispatch import receiver
from django.db import transaction
from .models import ComputeOrder, ComputeJob, GPUListing, JobEvent
from .tasks import provision_job


@receiver(post_save, sender=ComputeOrder)
def auto_match_on_order(sender, instance, created, **kwargs):
    """When a new market buy order is placed, try to match immediately."""
    if created and instance.order_type == 'buy' and instance.order_mode == 'market':
        transaction.on_commit(lambda: _try_immediate_match(instance))


@receiver(post_save, sender=ComputeJob)
def auto_provision_on_match(sender, instance, created, **kwargs):
    """When a new job is created (matched), trigger provisioning."""
    if created and instance.status == 'queued':
        transaction.on_commit(lambda: provision_job.delay(str(instance.id)))


@receiver(post_save, sender=ComputeJob)
def update_listing_on_job_complete(sender, instance, **kwargs):
    """When a job completes, update listing stats and free it."""
    if instance.status == 'completed':
        listing = instance.listing
        listing.status = 'online'
        listing.completed_jobs += 1
        listing.total_hours_rented += float(instance.hours_consumed or 0)
        listing.total_earnings_usd += instance.total_cost_usd or 0
        listing.save()
    elif instance.status == 'failed':
        listing = instance.listing
        listing.status = 'online'
        listing.failed_jobs += 1
        listing.save()


def _try_immediate_match(order):
    """Attempt to immediately match a market buy order."""
    from .matching_engine import MatchingEngine
    MatchingEngine().run()