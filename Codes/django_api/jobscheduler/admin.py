from django.contrib import admin
from .models import Job

@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'status', 'priority', 'gpu_requirements', 'created_at')
    list_filter = ('status', 'priority')
    search_fields = ('name', 'id')