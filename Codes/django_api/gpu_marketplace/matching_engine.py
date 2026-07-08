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

from .models import ComputeJob, ComputeOrder, GPUListing, JobEvent

logger = logging.getLogger(__name__)


class OrderBook:
    """
    A price-time priority order book for one GPU model.
    Buy orders: sorted descending by price, then by created_at.
    Sell orders: sorted ascending by price, then by created_at.
    """

    def __init__(self, gpu_model: str):
        self.gpu_model = gpu_model
        self.buy_orders = []  # list of ComputeOrder
        self.sell_orders = []  # list of GPUListing

    def load(self):
        """Load open orders from database for this GPU model."""
        self.buy_orders = list(
            ComputeOrder.objects.filter(
                order_type="buy",
                status="open",
                gpu_model_preference__iexact=self.gpu_model,
            ).order_by("-max_price_per_hour", "created_at")
        )
        self.sell_orders = list(
            GPUListing.objects.filter(
                gpu_model__iexact=self.gpu_model,
                status="online",
            ).order_by("current_price_per_hour", "created_at")
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
            ComputeOrder.objects.filter(order_type="buy", status="open")
            .values_list("gpu_model_preference", flat=True)
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

                if buy.status != "open" or sell.status != "online":
                    book.load()  # Reload
                    continue

                # Create the job
                job = ComputeJob.objects.create(
                    order=buy,
                    listing=sell,
                    consumer=buy.user,
                    provider=sell.provider,
                    status="queued",
                    docker_image=buy.docker_image or "pytorch/pytorch:latest",
                    command=buy.command or "",
                    input_data_url=buy.input_data_url or "",
                    price_per_hour=match_price,
                )
                JobEvent.objects.create(job=job, event_type="created")
                JobEvent.objects.create(job=job, event_type="matched")

                # Update order and listing
                buy.matched_listing = sell
                buy.final_price_per_hour = match_price
                buy.status = "filled"
                buy.filled_at = timezone.now()
                buy.save()

                sell.status = "busy"
                sell.total_rentals += 1
                sell.save()

                self.trades.append(
                    {
                        "buy_order": str(buy.id),
                        "listing": str(sell.id),
                        "price": str(match_price),
                        "job": str(job.id),
                    }
                )
                count += 1

            # Reload book after each match
            book.load()

        return count


def match_all_orders():
    """Convenience function for Celery task."""
    engine = MatchingEngine()
    return engine.run()


import logging

from .models import Candlestick, Portfolio, Trade, TradeOrder

logger = logging.getLogger(__name__)


def match_order(order):
    """Match a buy/sell order against the order book."""
    with transaction.atomic():
        opposite_side = "sell" if order.side == "buy" else "buy"
        # Find matching orders
        candidates = TradeOrder.objects.filter(
            gpu_model=order.gpu_model,
            side=opposite_side,
            status__in=["open", "partially_filled"],
        ).order_by("price" if order.side == "buy" else "-price")

        remaining = order.quantity
        for match in candidates:
            if remaining <= 0:
                break
            # Determine if price matches
            if order.order_type == "limit" and order.price:
                if order.side == "buy" and match.price > order.price:
                    break
                if order.side == "sell" and match.price < order.price:
                    break

            trade_qty = min(remaining, match.quantity - match.filled_quantity)
            if trade_qty <= 0:
                continue

            # Execute trade
            trade_price = match.price if match.order_type == "limit" else order.price
            Trade.objects.create(
                gpu_model=order.gpu_model,
                price=trade_price,
                quantity=trade_qty,
                buyer=order.user if order.side == "buy" else match.user,
                seller=match.user if order.side == "sell" else order.user,
                buy_order=order if order.side == "buy" else match,
                sell_order=match if order.side == "sell" else order,
            )

            # Update portfolio for both parties
            update_portfolio(
                order.user,
                order.gpu_model,
                trade_qty if order.side == "buy" else -trade_qty,
            )
            update_portfolio(
                match.user,
                order.gpu_model,
                -trade_qty if match.side == "buy" else trade_qty,
            )

            # Update order fill status
            order.filled_quantity += trade_qty
            order.status = (
                "filled"
                if order.filled_quantity >= order.quantity
                else "partially_filled"
            )
            order.save()

            match.filled_quantity += trade_qty
            match.status = (
                "filled"
                if match.filled_quantity >= match.quantity
                else "partially_filled"
            )
            match.save()

            remaining -= trade_qty

        # Update candlestick (simplified: use last trade price)
        if remaining < order.quantity:
            last_trade = (
                Trade.objects.filter(gpu_model=order.gpu_model)
                .order_by("-created_at")
                .first()
            )
            if last_trade:
                update_candlestick(order.gpu_model, float(last_trade.price), trade_qty)


def update_portfolio(user, gpu_model, delta):
    portfolio, _ = Portfolio.objects.get_or_create(user=user, gpu_model=gpu_model)
    portfolio.quantity += delta
    portfolio.save()


def update_candlestick(gpu_model, price, volume):
    from django.utils import timezone

    now = timezone.now()
    minute_start = now.replace(second=0, microsecond=0)
    candle, _ = Candlestick.objects.get_or_create(
        gpu_model=gpu_model,
        interval_start=minute_start,
        defaults={
            "open_price": price,
            "high_price": price,
            "low_price": price,
            "close_price": price,
        },
    )
    candle.high_price = max(candle.high_price, price)
    candle.low_price = min(candle.low_price, price)
    candle.close_price = price
    candle.volume += volume
    candle.save()
