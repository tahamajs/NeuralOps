# api/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/jobs/(?P<job_id>[^/]+)/$', consumers.JobConsumer.as_asgi()),
    re_path(r'ws/terminal/$', consumers.TerminalConsumer.as_asgi()),
]