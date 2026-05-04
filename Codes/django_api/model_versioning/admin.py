# model_versioning/admin.py
from django.contrib import admin
from .models import ModelVersion, ModelVersionDiff

@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ('id', 'run', 'version_number', 'version', 'is_published_to_hub', 'storage_tier', 'created_at')
    list_filter = ('storage_tier', 'is_published_to_hub', 'validation_passed')
    search_fields = ('version', 'run__id', 'run__name')
    readonly_fields = ('id', 'created_at', 'updated_at')

@admin.register(ModelVersionDiff)
class ModelVersionDiffAdmin(admin.ModelAdmin):
    list_display = ('id', 'from_version', 'to_version', 'diff_size_mb', 'created_at')