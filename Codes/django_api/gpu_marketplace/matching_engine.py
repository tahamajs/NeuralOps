"""
Matching Engine — Price-Time Priority (PTP) algorithm.

Inspired by exchange matching engines and Vast.ai's marketplace.
Price-Time priority: best price first, then earliest order.

Architecture:
    - Single-threaded per GPU model (like Dexilon's design)[reference:0]
    - OrderBook class: sorted list of buy/sell orders
    - MatchingEngine class: orchestrates matching across all GPU models
    - Called by Celery beat every 30 seconds
"""
import logging
from decimal import Decimal
from django.db import transaction
from django.utils import timezone
from .models import ComputeOrder, GPUListing, ComputeJob, JobEvent

logger = logging.getLogger(__name__)


class OrderBook:
    """
    A price-time priority order book for one GPU model.
    Buy orders: sorted descending by price, then by created_at.
    Sell orders: sorted ascending by price, then by created_at.
    """
    def __init__(self, gpu_model: str):
        self.gpu_model = gpu_model
        self.buy_orders = []   # list of ComputeOrder
        self.sell_orders = []  # list of GPUListing

    def load(self):
        """Load open orders from database for this GPU model."""
        self.buy_orders = list(
            ComputeOrder.objects.filter(
                order_type='buy', status='open',
                gpu_model_preference__iexact=self.gpu_model,
            ).order_by('-max_price_per_hour', 'created_at')
        )
        self.sell_orders = list(
            GPUListing.objects.filter(
                gpu_model__iexact=self.gpu_model,
                status='online',
            ).order_by('current_price_per_hour', 'created_at')
        )

    def best_buy(self):
        return self.buy_orders[0] if self.buy_orders else None

    def best_sell(self):
        return self.sell_orders[0] if self.sell_orders else None

    def can_match(self):
        buy = self.best_buy()
        sell = self.best_sell()
        if not buy or not sell:
            return False
        return float(buy.max_price_per_hour or 0) >= float(sell.current_price_per_hour)


class MatchingEngine:
    """
    Orchestrates matching across all GPU models.
    Runs as a Celery periodic task (every 30 seconds).
    """

    def __init__(self):
        self.trades = []

    def run(self):
        """Run a full matching cycle across all distinct GPU models."""
        gpu_models = (
            ComputeOrder.objects.filter(order_type='buy', status='open')
            .values_list('gpu_model_preference', flat=True)
            .distinct()
        )
        matched_count = 0
        for gpu_model in gpu_models:
            if not gpu_model:
                continue
            book = OrderBook(gpu_model)
            book.load()
            count = self._match_book(book)
            matched_count += count
        logger.info(f"Matching cycle complete: {matched_count} jobs matched")
        return matched_count

    def _match_book(self, book: OrderBook) -> int:
        """Match buy and sell orders in a single order book."""
        count = 0
        while book.can_match():
            buy = book.best_buy()
            sell = book.best_sell()
            if not buy or not sell:
                break

            match_price = Decimal(str(sell.current_price_per_hour))

            with transaction.atomic():
                # Refresh from DB to avoid race conditions
                buy.refresh_from_db()
                sell.refresh_from_db()

                if buy.status != 'open' or sell.status != 'online':
                    book.load()  # Reload
                    continue

                # Create the job
                job = ComputeJob.objects.create(
                    order=buy,
                    listing=sell,
                    consumer=buy.user,
                    provider=sell.provider,
                    status='queued',
                    docker_image=buy.docker_image or 'pytorch/pytorch:latest',
                    command=buy.command or '',
                    input_data_url=buy.input_data_url or '',
                    price_per_hour=match_price,
                )
                JobEvent.objects.create(job=job, event_type='created')
                JobEvent.objects.create(job=job, event_type='matched')

                # Update order and listing
                buy.matched_listing = sell
                buy.final_price_per_hour = match_price
                buy.status = 'filled'
                buy.filled_at = timezone.now()
                buy.save()

                sell.status = 'busy'
                sell.total_rentals += 1
                sell.save()

                self.trades.append({
                    'buy_order': str(buy.id),
                    'listing': str(sell.id),
                    'price': str(match_price),
                    'job': str(job.id),
                })
                count += 1

            # Reload book after each match
            book.load()

        return count


def match_all_orders():
    """Convenience function for Celery task."""
    engine = MatchingEngine()
    return engine.run()