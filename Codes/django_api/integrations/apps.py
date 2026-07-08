# integrations/apps.py
class IntegrationsConfig(AppConfig):
    name = "integrations"

    def ready(self):
        # Dynamically add plugin API routes when Django starts
        for plugin in PluginDefinition.objects.filter(is_active=True):
            if plugin.manifest.get("api_extensions"):
                for route in plugin.manifest["api_extensions"]:
                    _register_plugin_route(route)


def _register_plugin_route(route_config):
    from django.urls import path
    from integrations.views.plugins import plugin_proxy_view

    urlpatterns.append(path(route_config["path"], plugin_proxy_view))
