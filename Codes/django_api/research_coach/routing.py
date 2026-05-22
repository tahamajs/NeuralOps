from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/coach/$', consumers.ResearchCoachConsumer.as_asgi()),
    re_path(r'ws/coach/(?P<session_id>[^/]+)/$', consumers.ResearchCoachConsumer.as_asgi()),
]