import uuid
from django.db import models
from django.conf import settings
from experiment.models import Experiment
class Sweep(models.Model):
    class Status(models.TextChoices):
        ACTIVE = 'active', 'Active'
        PAUSED = 'paused', 'Paused'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'

    class Direction(models.TextChoices):
        MAXIMIZE = 'maximize', 'Maximize'
        MINIMIZE = 'minimize', 'Minimize'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name='sweep_set')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    config = models.JSONField(default=dict, blank=True,
        help_text='Full sweep configuration: search space, base_config, etc.')
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.ACTIVE
    )
    direction = models.CharField(
        max_length=20, choices=Direction.choices, default=Direction.MAXIMIZE
    )
    directions = models.JSONField(default=list, blank=True)
    metric_name = models.CharField(max_length=100, default='eval_accuracy')
    metric_names = models.JSONField(default=list, blank=True)
    algorithm = models.CharField(max_length=50, default='tpe')
    pruner = models.CharField(max_length=50, default='hyperband')
    scheduler_type = models.CharField(max_length=50, default='fifo')
    max_trials = models.IntegerField(default=20)
    max_failures = models.IntegerField(default=3)
    parallel_workers = models.IntegerField(default=1)
    warm_start_from = models.ForeignKey(
        'self', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='warm_started_sweeps'
    )
    # Preferential optimization
    preference_enabled = models.BooleanField(default=False)
    preference_queries = models.JSONField(default=list, blank=True)
    # Artifact store
    artifact_dir = models.CharField(max_length=500, blank=True)
    # Tags
    tags = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['project', 'status']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return self.name

    @property
    def best_trial(self):
        if self.metric_names:
            return None
        ordering = (
            '-objective_value'
            if self.direction == self.Direction.MAXIMIZE
            else 'objective_value'
        )
        return self.trials.exclude(
            objective_value__isnull=True
        ).order_by(ordering).first()

    @property
    def pareto_trials(self):
        if not self.metric_names:
            return []
        return self.trials.filter(is_pareto=True)


class SweepTrial(models.Model):
    class Status(models.TextChoices):
        QUEUED = 'queued', 'Queued'
        RUNNING = 'running', 'Running'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
        PRUNED = 'pruned', 'Pruned'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sweep = models.ForeignKey(Sweep, on_delete=models.CASCADE, related_name='trials')
    trial_number = models.IntegerField()
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.QUEUED
    )
    objective_value = models.FloatField(null=True, blank=True)
    objectives = models.JSONField(null=True, blank=True)
    params = models.JSONField(default=dict, blank=True)
    user_attrs = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)
    is_pareto = models.BooleanField(default=False)
    run = models.OneToOneField(
        'experiment.TrainingRun', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='sweep_trial'
    )
    # Artifact store
    artifacts = models.JSONField(default=list, blank=True)
    # Timing
    duration_seconds = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['sweep', 'trial_number']
        unique_together = [['sweep', 'trial_number']]
        indexes = [
            models.Index(fields=['sweep', 'status']),
            models.Index(fields=['objective_value']),
        ]

    def __str__(self):
        return f"{self.sweep.name} trial {self.trial_number}"