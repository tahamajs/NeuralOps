from django.urls import re_path

from .consumers import ProviderConsumer
from .trading_consumers import TradingConsumer

websocket_urlpatterns = [
    re_path(r"ws/trading/(?P<gpu_model>\w+)/$", TradingConsumer.as_asgi()),
    re_path(r"ws/provider/$", ProviderConsumer.as_asgi()),
]
