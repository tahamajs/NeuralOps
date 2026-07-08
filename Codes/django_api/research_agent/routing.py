from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(
        r"ws/agent/(?P<session_id>[^/]+)/$", consumers.ResearchAgentConsumer.as_asgi()
    ),
]
