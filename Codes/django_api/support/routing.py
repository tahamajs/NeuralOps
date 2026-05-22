from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/support/(?P<ticket_id>[^/]+)/$', consumers.SupportConsumer.as_asgi()),
]