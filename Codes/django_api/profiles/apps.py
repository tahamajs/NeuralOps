# profiles/apps.py
from django.apps import AppConfig

class ProfilesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'profiles'

    def ready(self):
        # Register the HuggingFace provider only if allauth is installed
        try:
            from allauth.socialaccount.providers import registry
            from profiles.providers.huggingface.provider import HuggingFaceProvider
            registry.register(HuggingFaceProvider)
        except ImportError:
            pass

        import profiles.signals  # noqa