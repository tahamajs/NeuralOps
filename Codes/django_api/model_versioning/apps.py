from django.apps import AppConfig

class ModelVersioningConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'model_versioning'

    def ready(self):
        import model_versioning.signals  # noqa: wire all signal receivers