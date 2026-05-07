import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from terminal.routing import websocket_urlpatterns as terminal_ws

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')

application = ProtocolTypeRouter({
    'http': get_asgi_application(),
    'websocket': AuthMiddlewareStack(
        URLRouter(terminal_ws)
    ),
})