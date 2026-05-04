from django.urls import re_path
from .consumers import JobConsumer, TerminalConsumer

websocket_urlpatterns = [
    re_path(r'ws/jobs/(?P<experiment_id>[^/]+)/$', JobConsumer.as_asgi()),
    re_path(r'ws/terminal/$', TerminalConsumer.as_asgi()),   # terminal endpoint
]