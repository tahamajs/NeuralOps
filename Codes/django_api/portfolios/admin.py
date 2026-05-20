from django.contrib import admin
from .models import Portfolio, ShowcaseItem, Endorsement, ActivityEvent, WidgetEmbed

@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    list_display = ('user', 'headline', 'is_public', 'created_at')
    search_fields = ('user__username', 'headline')

@admin.register(ShowcaseItem)
class ShowcaseItemAdmin(admin.ModelAdmin):
    list_display = ('portfolio', 'item_type', 'title', 'is_pinned')
    list_filter = ('item_type', 'is_pinned')

@admin.register(Endorsement)
class EndorsementAdmin(admin.ModelAdmin):
    list_display = ('portfolio', 'endorser', 'skill', 'created_at')

@admin.register(ActivityEvent)
class ActivityEventAdmin(admin.ModelAdmin):
    list_display = ('user', 'event_type', 'date', 'count')
    list_filter = ('event_type',)

@admin.register(WidgetEmbed)
class WidgetEmbedAdmin(admin.ModelAdmin):
    list_display = ('portfolio', 'theme', 'created_at')