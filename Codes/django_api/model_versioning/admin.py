from django.contrib import admin
from .models import ModelVersion

@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ('id', 'run', 'version_number', 'is_production', 'is_published_to_hub', 'created_at')
    list_filter = ('storage_tier', 'is_production')
