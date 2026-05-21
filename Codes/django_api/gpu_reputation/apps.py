from django.apps import AppConfig

class GpuReputationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'gpu_reputation'

    def ready(self):
        import gpu_reputation.signals