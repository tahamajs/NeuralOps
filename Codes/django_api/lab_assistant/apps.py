from django.apps import AppConfig


class LabAssistantConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "lab_assistant"

    def ready(self):
        pass
