"""Celery tasks for GPU finance — index calculation, settlement, invoicing."""
import logging
from celery import shared_task
from django.utils import timezone
from datetime import date, timedelta
from .models import ComputeIndex, FuturesContract, Invoice, InvoiceLineItem, UsageRecord
from .pricing_engine import recalculate_all_indices, calculate_settlement_price

logger = logging.getLogger(__name__)


@shared_task
def daily_index_calculation():
    """Calculate all compute indices daily."""
    return recalculate_all_indices()


@shared_task
def settle_expired_futures():
    """Settle all futures contracts whose last trading date has passed."""
    today = timezone.now().date()
    expired = FuturesContract.objects.filter(
        status='open', last_trading_date__lt=today
    )
    for contract in expired:
        final_price = calculate_settlement_price(contract)
        contract.settle(final_price)
        logger.info(f"Settled {contract.contract_code} at ${final_price}/hr")
    return expired.count()


@shared_task
def generate_monthly_invoices():
    """Generate invoices for all consumers for the previous month."""
    today = timezone.now().date()
    first_of_month = today.replace(day=1)
    last_month_end = first_of_month - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)

    users_with_usage = UsageRecord.objects.filter(
        recorded_at__date__gte=last_month_start,
        recorded_at__date__lte=last_month_end,
    ).values('user').distinct()

    count = 0
    for entry in users_with_usage:
        user_id = entry['user']
        from django.contrib.auth import get_user_model
        User = get_user_model()
        user = User.objects.get(id=user_id)

        # Create invoice
        invoice_number = f"INV-{last_month_end.strftime('%Y%m')}-{user.username[:4].upper()}-{count:04d}"
        invoice = Invoice.objects.create(
            user=user,
            invoice_number=invoice_number,
            status='open',
            period_start=last_month_start,
            period_end=last_month_end,
            due_date=today + timedelta(days=15),
        )

        # Add line items from usage records
        records = UsageRecord.objects.filter(
            user=user,
            recorded_at__date__gte=last_month_start,
            recorded_at__date__lte=last_month_end,
        )
        for rec in records:
            InvoiceLineItem.objects.create(
                invoice=invoice,
                charge_type='gpu_rental',
                description=f"{rec.gpu_model} — {rec.hours_consumed}h",
                job=rec.job,
                quantity=rec.hours_consumed,
                unit_price_usd=rec.cost_usd / rec.hours_consumed if rec.hours_consumed > 0 else 0,
                total_usd=rec.cost_usd,
            )

        invoice.recalculate_totals()
        count += 1

    return count