from django.contrib import admin
from .models import Job, VersionedConfig, CustomMetricEntry

@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'status', 'priority', 'gpu_requirements', 'created_at')
    list_filter = ('status', 'priority', 'task_name')
    search_fields = ('name', 'id')
    readonly_fields = ('id', 'created_at', 'started_at', 'completed_at', 'logs', 'training_history')

@admin.register(VersionedConfig)
class VersionedConfigAdmin(admin.ModelAdmin):
    list_display = ('id', 'run', 'version_number', 'created_at', 'change_description')
    list_filter = ('run',)
    readonly_fields = ('id', 'created_at')

@admin.register(CustomMetricEntry)
class CustomMetricEntryAdmin(admin.ModelAdmin):
    list_display = ('id', 'run', 'name', 'type', 'step', 'created_at')
    list_filter = ('run', 'type')