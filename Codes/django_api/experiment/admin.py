# experiments/admin.py
from django.contrib import admin
from .models import Experiment, TrainingRun, TrainingSnapshot

@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'created_at', 'tags')
    search_fields = ('name', 'tags')
    list_filter = ('created_at',)

@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ('id', 'experiment', 'name', 'job_type', 'status', 'created_at')
    list_filter = ('status', 'job_type', 'experiment')
    search_fields = ('name',)

@admin.register(TrainingSnapshot)
class TrainingSnapshotAdmin(admin.ModelAdmin):
    list_display = ('run', 'step', 'train_loss', 'eval_loss', 'timestamp')
    list_filter = ('run',)