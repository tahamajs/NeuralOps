import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

# Set Django settings module before any other import
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")

from api.routing import websocket_urlpatterns as api_ws
# Import WebSocket URL patterns from each app
from messaging.routing import websocket_urlpatterns as messaging_ws
from notes.routing import websocket_urlpatterns as notes_ws
from research_agent.routing import websocket_urlpatterns as agent_ws

try:
    from community.routing import websocket_urlpatterns as community_ws
except ImportError:
    community_ws = []

try:
    from support.routing import websocket_urlpatterns as support_ws
except ImportError:
    support_ws = []

# Combine all WebSocket patterns
websocket_urlpatterns = (
    messaging_ws + notes_ws + api_ws + agent_ws + support_ws + community_ws
)

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": AllowedHostsOriginValidator(
            AuthMiddlewareStack(URLRouter(websocket_urlpatterns))
        ),
    }
)
