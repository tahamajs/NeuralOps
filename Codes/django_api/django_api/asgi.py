import os
from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.urls import path
from api.consumers import JobConsumer, TerminalConsumer

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter([
            path("ws/jobs/<str:job_id>/", JobConsumer.as_asgi()),
            path("ws/terminal/", TerminalConsumer.as_asgi()),
        ])
    ),
})