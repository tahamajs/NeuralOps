from django.contrib import admin

from .models import Experiment, TrainingRun, TrainingSnapshot


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ["name", "created_at"]
    search_fields = ["name"]


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ["name", "experiment", "status", "job_type", "created_at"]
    list_filter = ["status", "job_type"]
    search_fields = ["name", "experiment__name"]
    readonly_fields = [
        "id",
        "created_at",
        "started_at",
        "completed_at",
        "id",
        "logs",
        "training_history",
    ]


@admin.register(TrainingSnapshot)
class TrainingSnapshotAdmin(admin.ModelAdmin):
    list_display = [
        "run",
        "step",
        "train_loss",
        "eval_loss",
        "eval_accuracy",
        "timestamp",
    ]
    list_filter = ["run"]
