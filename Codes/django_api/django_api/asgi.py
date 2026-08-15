import os
import sys

parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

# Set Django settings module before any other import
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")

# Import WebSocket URL patterns from each app
def _safe_import_ws(module_path):
    try:
        mod = __import__(module_path, fromlist=["websocket_urlpatterns"])
        return getattr(mod, "websocket_urlpatterns", [])
    except (ImportError, AttributeError):
        return []

websocket_urlpatterns = []
for _app_route in [
    "messaging.routing",
    "notes.routing",
    "api.routing",
    "research_agent.routing",
    "agents.routing",
    "support.routing",
    "community.routing",
    "gpu_marketplace.routing",
    "health_monitor.routing",
    "paper.routing",
    "coding_assistant.routing",
    "llm_hub.routing",
    "research_coach.routing",
    "terminal.routing",
    "web_dashboard.routing",
    "experiment.routing",
]:
    websocket_urlpatterns.extend(_safe_import_ws(_app_route))

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": AllowedHostsOriginValidator(
            AuthMiddlewareStack(URLRouter(websocket_urlpatterns))
        ),
    }
)

