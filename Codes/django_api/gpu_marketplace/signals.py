# gpu_marketplace/signals.py — EXTENDED VERSION
from django.db import transaction
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import ComputeJob, ComputeOrder, GPUListing
from .tasks import provision_job

# ── Existing imports ──────────────────────────────────────────
try:
    from experiment.models import Notification  # your existing notifications
except ImportError:
    Notification = None

try:
    from experiment.activity import log_activity  # your existing activity logger
except ImportError:
    log_activity = None

try:
    from gpu_escrow.models import EscrowTransaction  # escrow app
except ImportError:
    EscrowTransaction = None


# ═══════════════════════════════════════════════════════════════
# Notifications (real‑time)
# ═══════════════════════════════════════════════════════════════
def _notify(user, type, content, related_obj=None):
    """Send a notification if the notification app is installed."""
    if Notification:
        Notification.objects.create(
            recipient=user,
            sender=None,
            type=type,
            content=content,
            note=None,
        )


# ═══════════════════════════════════════════════════════════════
# Activity Logging
# ═══════════════════════════════════════════════════════════════
def _log(user, action_type, description, content_object=None, metadata=None):
    """Log an activity event if the activity app is installed."""
    if log_activity:
        log_activity(
            user=user,
            action_type=action_type,
            description=description,
            content_object=content_object,
            metadata=metadata or {},
        )


# ═══════════════════════════════════════════════════════════════
# Escrow
# ═══════════════════════════════════════════════════════════════
def _lock_escrow(job):
    """Lock consumer's funds in escrow when a job is matched."""
    if EscrowTransaction:
        EscrowTransaction.objects.create(
            job=job,
            tx_type="lock",
            amount_usd=job.price_per_hour * job.order.required_hours,
            from_user=job.consumer,
        )


def _release_escrow(job):
    """Release payment to provider when job completes."""
    if EscrowTransaction:
        EscrowTransaction.objects.create(
            job=job,
            tx_type="release",
            amount_usd=job.total_cost_usd,
            from_user=job.consumer,
            to_user=job.provider,
        )


def _refund_escrow(job):
    """Refund consumer if job fails (partial or full)."""
    if EscrowTransaction:
        EscrowTransaction.objects.create(
            job=job,
            tx_type="refund",
            amount_usd=job.total_cost_usd or 0,
            from_user=job.provider,  # comes from the platform? In real life from escrow pool
        )


# ═══════════════════════════════════════════════════════════════
# SIGNAL HANDLERS
# ═══════════════════════════════════════════════════════════════


@receiver(post_save, sender=ComputeOrder)
def on_order_created(sender, instance, created, **kwargs):
    if created:
        if instance.order_type == "buy":
            _log(
                instance.user,
                "order_placed",
                f"Buy order for {instance.gpu_model_preference} @ ≤${instance.max_price_per_hour}/hr",
            )
        else:
            _log(
                instance.user,
                "listing_offered",
                f"Sell order for {instance.gpu_model_preference}",
            )

        # If market order, try immediate match
        if instance.order_mode == "market":
            transaction.on_commit(lambda: _try_immediate_match(instance))


@receiver(post_save, sender=ComputeJob)
def on_job_created(sender, instance, created, **kwargs):
    if created:
        _notify(
            instance.provider,
            "job_matched",
            f"New job matched: {instance.listing.gpu_model}",
        )
        _notify(
            instance.consumer,
            "job_matched",
            f"Your compute job matched with {instance.provider.username}",
        )
        _log(
            instance.consumer,
            "job_matched",
            f"Job matched on {instance.listing.gpu_model} with {instance.provider.username}",
            content_object=instance,
        )
        _log(
            instance.provider,
            "job_received",
            f"Received job from {instance.consumer.username}",
            content_object=instance,
        )

        # Lock funds in escrow
        _lock_escrow(instance)

        # Trigger provisioning
        transaction.on_commit(lambda: provision_job.delay(str(instance.id)))


@receiver(post_save, sender=ComputeJob)
def on_job_completed(sender, instance, **kwargs):
    if instance.status == "completed":
        _notify(
            instance.consumer,
            "job_completed",
            f"Job completed on {instance.listing.gpu_model}",
        )
        _notify(
            instance.provider,
            "payment_released",
            f"Payment of ${instance.total_cost_usd} released",
        )
        _log(
            instance.consumer,
            "job_completed",
            "Compute job completed",
            content_object=instance,
        )
        _log(
            instance.provider,
            "payment_earned",
            f"Earned ${instance.total_cost_usd}",
            content_object=instance,
        )

        # Release payment from escrow
        _release_escrow(instance)

    elif instance.status == "failed":
        _notify(
            instance.consumer,
            "job_failed",
            f"Job failed on {instance.listing.gpu_model}",
        )
        _notify(instance.provider, "job_failed", "Job failed on your GPU")
        _refund_escrow(instance)

    elif instance.status == "disputed":
        _notify(
            instance.provider,
            "dispute_raised",
            f"A dispute was raised for job {instance.id}",
        )
        _log(
            instance.consumer,
            "dispute_raised",
            f"Disputed job {instance.id}",
            content_object=instance,
        )


@receiver(post_save, sender=GPUListing)
def on_listing_created(sender, instance, created, **kwargs):
    if created:
        _log(
            instance.provider,
            "gpu_listed",
            f"Listed {instance.gpu_model} ×{instance.gpu_count} @ ${instance.base_price_per_hour}/hr",
            content_object=instance,
        )
        _notify(instance.provider, "listing_live", "Your GPU listing is now live")


def _try_immediate_match(order):
    from .matching_engine import MatchingEngine

    MatchingEngine().run()


from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import Trade, TradeOrder


@receiver(post_save, sender=Trade)
def broadcast_trade(sender, instance, created, **kwargs):
    if created:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"trading_{instance.gpu_model}",
            {
                "type": "trade_update",
                "data": {
                    "price": float(instance.price),
                    "quantity": instance.quantity,
                    "buyer": instance.buyer.username,
                    "seller": instance.seller.username,
                    "timestamp": instance.created_at.isoformat(),
                },
            },
        )


@receiver(post_save, sender=TradeOrder)
def broadcast_order_book(sender, instance, **kwargs):
    if instance.status in ["open", "partially_filled", "filled", "canceled"]:
        channel_layer = get_channel_layer()
        # Recompute order book snapshot and send
        from .trading_views import _aggregate_levels

        bids = (
            TradeOrder.objects.filter(
                gpu_model=instance.gpu_model,
                side="buy",
                status__in=["open", "partially_filled"],
            )
            .order_by("-price")
            .values("price", "quantity")
        )
        asks = (
            TradeOrder.objects.filter(
                gpu_model=instance.gpu_model,
                side="sell",
                status__in=["open", "partially_filled"],
            )
            .order_by("price")
            .values("price", "quantity")
        )
        data = {
            "bids": _aggregate_levels(bids),
            "asks": _aggregate_levels(asks),
        }
        async_to_sync(channel_layer.group_send)(
            f"trading_{instance.gpu_model}", {"type": "order_book_update", "data": data}
        )
