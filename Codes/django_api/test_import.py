import os
import sys

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_api.settings")
import django

django.setup()
print("Importing views...")
print("Done!")
