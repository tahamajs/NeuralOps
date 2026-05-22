from rest_framework import serializers
from .models import TradeOrder, Trade, Candlestick, Portfolio


class TradeOrderSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)

    class Meta:
        model = TradeOrder
        fields = '__all__'
        read_only_fields = ['id', 'user', 'filled_quantity', 'status', 'created_at', 'updated_at']


class OrderBookSerializer(serializers.Serializer):
    """Aggregated order book for a GPU model."""
    bids = serializers.ListField(child=serializers.ListField())
    asks = serializers.ListField(child=serializers.ListField())
    spread = serializers.FloatField()
    last_price = serializers.DecimalField(max_digits=10, decimal_places=4)


class TradeSerializer(serializers.ModelSerializer):
    buyer_username = serializers.CharField(source='buyer.username', read_only=True)
    seller_username = serializers.CharField(source='seller.username', read_only=True)

    class Meta:
        model = Trade
        fields = '__all__'


class CandlestickSerializer(serializers.ModelSerializer):
    class Meta:
        model = Candlestick
        fields = ['id', 'gpu_model', 'interval_start', 'open_price', 'high_price',
                  'low_price', 'close_price', 'volume']


class PortfolioSerializer(serializers.ModelSerializer):
    current_price = serializers.SerializerMethodField()
    market_value = serializers.SerializerMethodField()

    class Meta:
        model = Portfolio
        fields = ['id', 'user', 'gpu_model', 'quantity', 'current_price', 'market_value']

    def get_current_price(self, obj):
        last_trade = Trade.objects.filter(gpu_model=obj.gpu_model).order_by('-created_at').first()
        return last_trade.price if last_trade else None

    def get_market_value(self, obj):
        last_trade = Trade.objects.filter(gpu_model=obj.gpu_model).order_by('-created_at').first()
        if last_trade and obj.quantity:
            return obj.quantity * float(last_trade.price)
        return None