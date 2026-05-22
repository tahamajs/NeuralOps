from django.apps import AppConfig

class LLMHubConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'llm_hub'

    def ready(self):
        import llm_hub.signals  # noqa