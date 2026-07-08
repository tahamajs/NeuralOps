from django.apps import AppConfig


class LiteratureDiscoveryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "literature_discovery"

    def ready(self):
        import literature_discovery.signals  # noqa
