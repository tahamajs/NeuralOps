from django.apps import AppConfig


class HealthMonitorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "health_monitor"

    def ready(self):
        pass
