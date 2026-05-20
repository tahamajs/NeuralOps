from django.apps import AppConfig

class NotesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'notes'

    def ready(self):
        # Create default templates on startup
        from .templates import create_default_templates
        create_default_templates()
        
        # Connect signals
        import notes.signals  # noqa