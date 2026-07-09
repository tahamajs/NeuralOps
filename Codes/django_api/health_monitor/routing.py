from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/health/(?P<dashboard_id>[^/]+)/$", consumers.GlobalHealthConsumer.as_asgi()),
]
