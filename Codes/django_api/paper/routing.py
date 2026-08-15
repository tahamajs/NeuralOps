# paper/routing.py
from django.urls import re_path

from . import consumers

websocket_urlpatterns = [
    re_path(
        r"ws/paper/(?P<project_id>[^/]+)/$",
        consumers.PaperCollabConsumer.as_asgi(),
    ),
    re_path(
        r"ws/paper/collab/(?P<project_id>[^/]+)/$",
        consumers.PaperCollabConsumer.as_asgi(),
    ),
    re_path(
        r"ws/collab/(?P<project_id>[^/]+)/$",
        consumers.CollaborationConsumer.as_asgi(),
    ),
]
