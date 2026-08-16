# jobscheduler/apps.py
import os

from django.apps import AppConfig


class JobschedulerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "jobscheduler"

    def ready(self):
        import sys

        # Avoid starting scheduler during tests, checks, migrations, or verification scripts
        cli_commands = ["test", "check", "verify", "showmigrations", "migrate", "collectstatic"]
        if any(cmd in " ".join(sys.argv) for cmd in cli_commands):
            return

        if os.environ.get("RUN_MAIN") != "true" and not os.environ.get(
            "DJANGO_AUTORELOAD"
        ):
            from .scheduler import scheduler

            # Ensure scheduler is started only once
            if not scheduler.worker_thread or not scheduler.worker_thread.is_alive():
                scheduler.start()
