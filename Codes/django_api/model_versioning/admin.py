from django.contrib import admin
from .models import ModelVersion, ModelVersionDiff, ModelLineage


class ModelVersionDiffInline(admin.TabularInline):
    model = ModelVersionDiff
    fk_name = 'from_version'
    extra = 0
    fields = ('to_version', 'diff_size_mb', 'original_size_saved_mb')
    readonly_fields = fields


class ModelLineageInline(admin.TabularInline):
    model = ModelLineage
    fk_name = 'child'
    extra = 0
    fields = ('parent', 'relationship_type', 'notes')
    readonly_fields = fields


@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'run', 'version_number', 'version',
        'artifact_size_mb', 'compressed_size_mb',
        'storage_tier', 'is_production', 'is_published_to_hub',
        'validation_passed', 'created_at',
    )
    list_filter = (
        'storage_tier', 'is_production', 'is_published_to_hub',
        'validation_passed', 'compression_method',
    )
    search_fields = ('version', 'run__name', 'hub_repo_url')
    readonly_fields = ('id', 'created_at', 'updated_at')
    inlines = [ModelVersionDiffInline, ModelLineageInline]
    fieldsets = (
        (None, {
            'fields': (
                'id', 'run', 'version', 'version_number',
                'is_production', 'is_published_to_hub', 'hub_repo_url',
            )
        }),
        ('Artifact', {
            'fields': (
                'artifact_path', 'compressed_artifact_path',
                'artifact_size_mb', 'compressed_size_mb',
                'compression_method', 'compression_config',
            )
        }),
        ('Metrics & Validation', {
            'fields': ('metrics', 'validation_passed', 'git_commit', 'config_snapshot_id')
        }),
        ('Storage', {
            'fields': ('storage_tier',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
        }),
    )


@admin.register(ModelVersionDiff)
class ModelVersionDiffAdmin(admin.ModelAdmin):
    list_display = ('id', 'from_version', 'to_version', 'diff_size_mb', 'created_at')
    search_fields = ('from_version__version', 'to_version__version')
    readonly_fields = ('id', 'created_at')


@admin.register(ModelLineage)
class ModelLineageAdmin(admin.ModelAdmin):
    list_display = ('id', 'parent', 'child', 'relationship_type', 'created_at')
    list_filter = ('relationship_type',)
    search_fields = ('parent__version', 'child__version')
    readonly_fields = ('id', 'created_at')