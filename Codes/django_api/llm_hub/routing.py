from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/prompt/(?P<version_id>[^/]+)/$', consumers.PromptCollabConsumer.as_asgi()),
]