from django.apps import AppConfig


class GpuFederationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "gpu_federation"

    def ready(self):
        pass
