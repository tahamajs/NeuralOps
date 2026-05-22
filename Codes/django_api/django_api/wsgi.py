"""
ASGI config for the project.

It exposes the ASGI callable as a module-level variable named ``application``,
and sets up WebSocket routing alongside HTTP.
"""

import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")

# Import your WebSocket patterns from each app
from coding_assistant.routing import websocket_urlpatterns as code_ws
from research_coach.routing import websocket_urlpatterns as coach_ws

# Merge all WebSocket patterns
websocket_urlpatterns = code_ws + coach_ws

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})