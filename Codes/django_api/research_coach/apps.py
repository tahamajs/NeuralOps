from django.apps import AppConfig


class ResearchCoachConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "research_coach"

    def ready(self):
        # import research_coach.signals  # noqa
        pass
