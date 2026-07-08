import sys

from django.apps import AppConfig
from django.db.utils import OperationalError, ProgrammingError


class NotesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "notes"

    def ready(self):
        # Avoid running when Django is collecting static files or making migrations
        if "migrate" in sys.argv or "makemigrations" in sys.argv:
            return

        try:
            from .models import NoteTemplate

            # Try a simple query to see if the table exists
            NoteTemplate.objects.first()
        except (OperationalError, ProgrammingError):
            # Table doesn't exist yet – skip creation
            return

        # Only now safe to create default templates
        from .templates import create_default_templates

        create_default_templates()
