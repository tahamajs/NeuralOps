from django.contrib import admin
from .models import ResearchProject, ImpactPrediction


@admin.register(ResearchProject)
class ResearchProjectAdmin(admin.ModelAdmin):
    list_display = ('title', 'user', 'domain', 'status', 'created_at')
    list_filter = ('status', 'domain')
    search_fields = ('title', 'user__username')


@admin.register(ImpactPrediction)
class ImpactPredictionAdmin(admin.ModelAdmin):
    list_display = ('project', 'impact_score', 'percentile', 'funding_probability', 'publication_tier')
    readonly_fields = ('created_at',)