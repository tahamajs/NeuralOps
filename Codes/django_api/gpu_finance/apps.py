from django.apps import AppConfig


class GpuFinanceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "gpu_finance"

    def ready(self):
        pass
