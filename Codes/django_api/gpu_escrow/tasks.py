# gpu_escrow/tasks.py
"""Periodic tasks for escrow reconciliation."""

from celery import shared_task


@shared_task
def daily_reconciliation():
    """Placeholder: compare balances and flag discrepancies."""
    pass
