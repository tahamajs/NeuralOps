#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""

import os
import signal
import sys


def main():
    # Add parent directory to sys.path so 'framework' module can be imported
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # Register shutdown handler AFTER settings are loaded (inside main)
    def handle_shutdown(signum, frame):
        print("\nShutting down – stopping all training jobs...")
        try:
            from api.jobs import stop_all_jobs

            stop_all_jobs()
        except Exception as e:
            print(f"Could not stop jobs: {e}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
