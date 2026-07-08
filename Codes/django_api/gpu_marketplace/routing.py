from django.urls import re_path

from .trading_consumers import TradingConsumer

websocket_urlpatterns = [
    re_path(r"ws/trading/(?P<gpu_model>\w+)/$", TradingConsumer.as_asgi()),
]
