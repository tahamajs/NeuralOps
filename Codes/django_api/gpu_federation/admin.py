from django.contrib import admin
from .models import *

@admin.register(FederationCluster)
class FederationClusterAdmin(admin.ModelAdmin):
    list_display = ('name', 'owner', 'status', 'model_name', 'sharding_strategy', 'created_at')
    list_filter = ('status', 'sharding_strategy')
    search_fields = ('name', 'owner__username')

@admin.register(FederationNode)
class FederationNodeAdmin(admin.ModelAdmin):
    list_display = ('cluster', 'listing', 'status', 'shard_start', 'shard_end')
    list_filter = ('status',)

@admin.register(FederationTask)
class FederationTaskAdmin(admin.ModelAdmin):
    list_display = ('cluster', 'owner', 'status', 'progress_percent', 'created_at')
    list_filter = ('status',)

@admin.register(FederationTaskShard)
class FederationTaskShardAdmin(admin.ModelAdmin):
    list_display = ('task', 'node', 'status', 'steps_completed')

@admin.register(FederationResult)
class FederationResultAdmin(admin.ModelAdmin):
    list_display = ('task', 'final_loss', 'final_accuracy', 'total_time_seconds')