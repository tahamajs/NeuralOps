"""
WSGI config for the project.

It exposes the WSGI callable as a module-level variable named ``application``.
"""

import os
import sys

parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")

application = get_wsgi_application()
