import os
import sys

parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set Django settings module before any other imports
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")

from django.core.asgi import get_asgi_application

# Initialize Django ASGI application early to ensure app registry is populated
django_asgi_app = get_asgi_application()

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator

from django_api.routing import websocket_urlpatterns

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AuthMiddlewareStack(URLRouter(websocket_urlpatterns)),
    }
)
