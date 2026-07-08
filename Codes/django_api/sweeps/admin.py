from django.contrib import admin

from .models import Sweep, SweepTrial


@admin.register(Sweep)
class SweepAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "project",
        "status",
        "algorithm",
        "pruner",
        "direction",
        "created_at",
    ]
    list_filter = ["status", "algorithm", "pruner", "project"]
    search_fields = ["name"]


@admin.register(SweepTrial)
class SweepTrialAdmin(admin.ModelAdmin):
    list_display = [
        "sweep",
        "trial_number",
        "status",
        "objective_value",
        "is_pareto",
        "created_at",
    ]
    list_filter = ["sweep", "status"]
