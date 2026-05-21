from django.apps import AppConfig

class GpuBenchmarkConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'gpu_benchmark'

    def ready(self):
        import gpu_benchmark.signals