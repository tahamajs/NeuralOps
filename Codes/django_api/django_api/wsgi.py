import os
from django.core.wsgi import get_wsgi_application
from coding_assistant.routing import websocket_urlpatterns as code_ws

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")
application = get_wsgi_application()