from django.apps import AppConfig


class ExpertWallConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "expert_wall"

    def ready(self):
        import expert_wall.signals  # noqa
