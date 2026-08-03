import logging

from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .matching_engine import match_order
from .models import Candlestick, Portfolio, Trade, TradeOrder
from .trading_serializers import (
    CandlestickSerializer,
    PortfolioSerializer,
    TradeOrderSerializer,
    TradeSerializer,
)

logger = logging.getLogger(__name__)


class TradeOrderViewSet(viewsets.ModelViewSet):
    serializer_class = TradeOrderSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return TradeOrder.objects.filter(user=self.request.user).order_by("-created_at")

    def perform_create(self, serializer):
        order = serializer.save(user=self.request.user)
        # For market orders, execute immediately
        if order.order_type == "market":
            match_order(order)

    @action(detail=False, methods=["get"])
    def order_book(self, request):
        gpu_model = request.query_params.get("gpu_model", "H100")
        bids = (
            TradeOrder.objects.filter(gpu_model=gpu_model, side="buy", status="open")
            .order_by("-price")
            .values("price", "quantity")
        )
        asks = (
            TradeOrder.objects.filter(gpu_model=gpu_model, side="sell", status="open")
            .order_by("price")
            .values("price", "quantity")
        )

        bids_agg = _aggregate_levels(bids)
        asks_agg = _aggregate_levels(asks)
        last_trade = (
            Trade.objects.filter(gpu_model=gpu_model).order_by("-created_at").first()
        )
        return Response(
            {
                "bids": bids_agg,
                "asks": asks_agg,
                "spread": (
                    asks_agg[0][0] - bids_agg[0][0] if bids_agg and asks_agg else 0
                ),
                "last_price": last_trade.price if last_trade else None,
            }
        )

    @action(detail=False, methods=["get"])
    def trades(self, request):
        gpu_model = request.query_params.get("gpu_model", "H100")
        trades = Trade.objects.filter(gpu_model=gpu_model).order_by("-created_at")[:50]
        serializer = TradeSerializer(trades, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=["get"])
    def candles(self, request):
        gpu_model = request.query_params.get("gpu_model", "H100")
        interval = request.query_params.get("interval", "1m")
        limit = int(request.query_params.get("limit", 100))
        # Determine time range
        from datetime import timedelta

        from django.utils import timezone

        if interval == "1m":
            delta = timedelta(minutes=limit)
        elif interval == "5m":
            delta = timedelta(minutes=5 * limit)
        elif interval == "1h":
            delta = timedelta(hours=limit)
        else:
            delta = timedelta(hours=limit * 24)
        since = timezone.now() - delta
        candles = Candlestick.objects.filter(
            gpu_model=gpu_model, interval_start__gte=since
        ).order_by("interval_start")
        serializer = CandlestickSerializer(candles, many=True)
        return Response(serializer.data)


class PortfolioViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = PortfolioSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Portfolio.objects.filter(user=self.request.user)


def _aggregate_levels(orders):
    """Aggregate orders at same price level."""
    levels = {}
    for order in orders:
        price = float(order["price"])
        qty = order["quantity"]
        levels[price] = levels.get(price, 0) + qty
    # Sort: bids descending, asks ascending (caller handles order)
    return [
        [p, q]
        for p, q in sorted(
            levels.items(),
            reverse=orders[0]["price"] > orders[-1]["price"] if orders else True,
        )
    ]
