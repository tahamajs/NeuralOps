from django.apps import AppConfig


class CollaboratorMatchingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "collaborator_matching"

    def ready(self):
        import collaborator_matching.signals  # noqa
