# gpu_escrow/signals.py
"""
Automatically move funds when compute jobs change status.
Listens to gpu_marketplace.ComputeJob post_save signal.
"""

import logging
from decimal import Decimal

from django.db import transaction
from django.db.models.signals import post_save
from django.dispatch import receiver

logger = logging.getLogger(__name__)


@receiver(post_save, sender="gpu_marketplace.ComputeJob")
def handle_job_escrow(sender, instance, created, **kwargs):
    """
    - On job creation (status='queued'): lock consumer funds (estimated total cost).
    - On completion: release funds from consumer's locked balance to provider's available balance.
    - On failure / dispute: refund locked funds to consumer.
    """
    from gpu_escrow.models import EscrowAccount

    # Only act on status transitions that matter.
    # We can't reliably detect transitions here, so we check the current status.
    # In production, use a state machine or compare old/new in pre_save.
    # For simplicity, we'll handle each status without duplication by using a flag.
    try:
        if instance.status == "queued" and created:
            # Lock estimated total cost
            total_hours = instance.order.required_hours
            estimated_cost = instance.price_per_hour * Decimal(total_hours)
            consumer_account, _ = EscrowAccount.objects.get_or_create(
                user=instance.consumer
            )
            # Avoid double-locking if already processed
            already_locked = instance.escrow_transactions.filter(
                tx_type="lock"
            ).exists()
            if not already_locked:
                with transaction.atomic():
                    consumer_account.lock_funds(estimated_cost, job=instance)

        elif instance.status == "completed":
            # Release actual cost from locked balance to provider
            actual_cost = instance.total_cost_usd or Decimal("0")
            consumer_account, _ = EscrowAccount.objects.get_or_create(
                user=instance.consumer
            )
            provider_account, _ = EscrowAccount.objects.get_or_create(
                user=instance.provider
            )
            # Avoid double-release
            already_released = instance.escrow_transactions.filter(
                tx_type="release"
            ).exists()
            if not already_released:
                with transaction.atomic():
                    consumer_account.release_funds(
                        actual_cost, to_account=provider_account, job=instance
                    )

        elif instance.status in ("failed", "disputed"):
            # Refund locked funds (remaining locked) back to consumer's available balance
            # Determine how much was locked for this job originally.
            # We'll refund all currently locked funds for this job (or we can track the original lock amount).
            consumer_account, _ = EscrowAccount.objects.get_or_create(
                user=instance.consumer
            )
            already_refunded = instance.escrow_transactions.filter(
                tx_type="refund"
            ).exists()
            if not already_refunded:
                # Refund the original locked amount (we'll read the lock transaction)
                lock_tx = instance.escrow_transactions.filter(tx_type="lock").first()
                if lock_tx:
                    refund_amount = lock_tx.amount_usd
                    with transaction.atomic():
                        consumer_account.refund_locked_funds(
                            refund_amount, job=instance
                        )

    except Exception as e:
        logger.exception("Escrow signal handler failed for job %s: %s", instance.id, e)
