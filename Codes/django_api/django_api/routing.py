from channels.routing import ProtocolTypeRouter, URLRouter
from messaging.routing import websocket_urlpatterns as messaging_ws
from notes.routing import websocket_urlpatterns as notes_ws
from api.routing import websocket_urlpatterns as api_ws

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter(
        messaging_ws + notes_ws + api_ws
    ),
})