import os
from importlib import import_module

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')

websocket_urlpatterns = []
for module_name in [
    'messaging.routing',
    'notes.routing',
    'api.routing',
    'research_agent.routing',
    'support.routing',
    'agents.routing',
    'web_dashboard.routing',
    'coding_assistant.routing',
    'terminal.routing',
    'gpu_marketplace.routing',
    'llm_hub.routing',
    'research_coach.routing',
    'experiment.routing',
]:
    try:
        module = import_module(module_name)
        websocket_urlpatterns += getattr(module, 'websocket_urlpatterns', [])
    except ModuleNotFoundError:
        continue

application = ProtocolTypeRouter(
    {
        'http': get_asgi_application(),
        'websocket': AllowedHostsOriginValidator(AuthMiddlewareStack(URLRouter(websocket_urlpatterns))),
    }
)