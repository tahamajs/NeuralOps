"""
Compute Index pricing engine — like CME's Silicon Data benchmark.

Daily index calculation using volume‑weighted average price (VWAP)
from live marketplace listings. Supports historical snapshots.
"""

import logging
from decimal import Decimal

from django.utils import timezone

from .models import ComputeIndex, IndexHistory

logger = logging.getLogger(__name__)


def recalculate_all_indices():
    """Recalculate all compute indices from live marketplace data."""
    indices = ComputeIndex.objects.all()
    results = []
    for idx in indices:
        idx.calculate()
        # Take a historical snapshot
        IndexHistory.objects.get_or_create(
            index=idx,
            recorded_at=timezone.now().date(),
            defaults={
                "value": idx.current_value,
                "data_points": idx.data_points,
            },
        )
        results.append(
            {
                "index": idx.index_name,
                "value": float(idx.current_value),
                "data_points": idx.data_points,
                "change_pct": idx.daily_change_pct,
            }
        )
    return results


def get_index_history(index_name: str, days: int = 30):
    """Return historical index values for charting."""
    try:
        idx = ComputeIndex.objects.get(index_name=index_name)
    except ComputeIndex.DoesNotExist:
        return []
    qs = IndexHistory.objects.filter(index=idx).order_by("-recorded_at")[:days]
    return [{"date": str(h.recorded_at), "value": float(h.value)} for h in reversed(qs)]


def calculate_settlement_price(contract) -> Decimal:
    """
    Calculate the final settlement price for a futures contract.
    Uses the VWAP of the underlying index over the last 5 trading days.
    """
    idx = contract.index
    history = IndexHistory.objects.filter(
        index=idx, recorded_at__lte=contract.last_trading_date or timezone.now().date()
    ).order_by("-recorded_at")[:5]
    if not history:
        return idx.current_value
    return sum(h.value for h in history) / len(history)
