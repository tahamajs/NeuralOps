from django.apps import AppConfig

class GPUMarketplaceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'gpu_marketplace'

    def ready(self):
        import gpu_marketplace.signals