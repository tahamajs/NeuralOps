from django.apps import AppConfig

class FailureForensicsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'failure_forensics'

    def ready(self):
        import failure_forensics.signals  # noqa