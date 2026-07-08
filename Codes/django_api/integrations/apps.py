# integrations/apps.py
from django.apps import AppConfig
try:
    from django_api.urls import urlpatterns
except ImportError:
    urlpatterns = []

class IntegrationsConfig(AppConfig):
    name = "integrations"

    def ready(self):
        # Dynamically add plugin API routes when Django starts
        pass

def _register_plugin_route(route_config):
    pass

