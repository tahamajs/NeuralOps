# profiles/apps.py
from django.apps import AppConfig
import allauth 
class ProfilesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'profiles'

    def ready(self):
        from allauth.socialaccount.providers import registry
        registry.register(profiles.providers.huggingface.provider.HuggingFaceProvider)
        import profiles.signals
