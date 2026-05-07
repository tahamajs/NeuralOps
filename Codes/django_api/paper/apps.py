from django.apps import AppConfig

class PaperConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'paper'

    def ready(self):
        import paper.signals  # noqa - register signals