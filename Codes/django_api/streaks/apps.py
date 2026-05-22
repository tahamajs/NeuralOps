from django.apps import AppConfig

class StreaksConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'streaks'

    def ready(self):
        import streaks.signals  # noqa