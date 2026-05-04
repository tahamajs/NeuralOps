# model_versioning/apps.py
from django.apps import AppConfig

class ModelVersioningConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'model_versioning'