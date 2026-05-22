from django.apps import AppConfig

class ReputationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'reputation'

    def ready(self):
        import reputation.signals  # noqa