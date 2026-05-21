# leaderboards/apps.py
from django.apps import AppConfig

class LeaderboardsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'leaderboards'

    def ready(self):
        try:
            import leaderboards.signals  # noqa
        except Exception:
            pass