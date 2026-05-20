from django.contrib import admin
from .models import DashboardWidget

@admin.register(DashboardWidget)
class DashboardWidgetAdmin(admin.ModelAdmin):
    list_display = ['user', 'widget_type', 'title', 'position_x', 'position_y', 'is_visible']
    list_filter = ['widget_type', 'is_visible']
    search_fields = ['user__username', 'title']