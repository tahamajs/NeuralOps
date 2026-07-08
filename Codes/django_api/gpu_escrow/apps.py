# gpu_escrow/apps.py
from django.apps import AppConfig


class GpuEscrowConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "gpu_escrow"

    def ready(self):
        import gpu_escrow.signals  # noqa
