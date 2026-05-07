from django.contrib import admin
from experiment.models import Sweep, SweepTrial

@admin.register(Sweep)
class SweepAdmin(admin.ModelAdmin):
    list_display = ['name', 'project', 'status', 'direction', 'created_at']
    list_filter = ['status', 'project']

@admin.register(SweepTrial)
class SweepTrialAdmin(admin.ModelAdmin):
    list_display = ['sweep', 'trial_number', 'status', 'objective_value', 'created_at']
    list_filter = ['sweep', 'status']