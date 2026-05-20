import uuid
from django.db import models

class Experiment(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    default_config = models.JSONField(default=dict, blank=True)
    tags = models.CharField(max_length=200, blank=True, default="")

    class Meta:
        app_label = 'experiment'          # <-- ADD THIS LINE
        ordering = ['-created_at']
        indexes = [models.Index(fields=['created_at']), models.Index(fields=['tags'])]

    def __str__(self):
        return self.name


class TrainingRun(models.Model):
    JOB_TYPES = (("training", "Training"), ("evaluation", "Evaluation"), ("compression", "Compression"))
    STATUS_CHOICES = (
        ("queued", "Queued"), ("running", "Running"), ("completed", "Completed"),
        ("failed", "Failed"), ("stopped", "Stopped")
    )
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name='runs')
    name = models.CharField(max_length=200, blank=True)
    job_type = models.CharField(max_length=20, default='training', choices=JOB_TYPES)
    status = models.CharField(max_length=20, default='queued', choices=STATUS_CHOICES)
    config = models.JSONField(default=dict)
    final_metrics = models.JSONField(default=dict)
    training_history = models.JSONField(default=dict, blank=True)
    logs = models.TextField(blank=True)
    artifacts_dir = models.CharField(max_length=500, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    current_step = models.IntegerField(default=0)
    total_steps = models.IntegerField(default=0)
    estimated_remaining_seconds = models.FloatField(null=True, blank=True)
    cpu_usage = models.FloatField(default=0.0)
    ram_usage_mb = models.FloatField(default=0.0)
    gpu_usage = models.FloatField(null=True, blank=True)
    gpu_memory_mb = models.FloatField(null=True, blank=True)

    class Meta:
        app_label = 'experiment'          # <-- ADD THIS LINE
        ordering = ['-created_at']
        indexes = [models.Index(fields=['experiment', 'status']), models.Index(fields=['job_type'])]

    def __str__(self):
        return f"{self.experiment.name} - {self.name or self.job_type}"


class TrainingSnapshot(models.Model):
    run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE, related_name='snapshots', null=True, blank=True)
    step = models.IntegerField()
    epoch = models.FloatField(null=True, blank=True)
    train_loss = models.FloatField(null=True, blank=True)
    eval_loss = models.FloatField(null=True, blank=True)
    eval_accuracy = models.FloatField(null=True, blank=True)
    learning_rate = models.FloatField(null=True, blank=True)
    cpu_percent = models.FloatField(null=True)
    ram_mb = models.FloatField(null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'experiment'          # <-- ADD THIS LINE
        ordering = ['run', 'step']
        unique_together = [['run', 'step']]


class ExperimentTemplate(models.Model):
    """Reusable experiment configuration template."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # author = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, related_name='templates')
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    config = models.JSONField(default=dict)
    category = models.CharField(max_length=100, blank=True, default="general")
    is_public = models.BooleanField(default=False)
    usage_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'experiment'   # <-- add this line

        ordering = ['-usage_count', '-created_at']


class Sweep(models.Model):
    STATUS_CHOICES = (
        ('active', 'Active'),
        ('paused', 'Paused'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    DIRECTION_CHOICES = (
        ('maximize', 'Maximize'),
        ('minimize', 'Minimize'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name='sweeps')
    name = models.CharField(max_length=200)
    config = models.JSONField(default=dict)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    direction = models.CharField(max_length=20, choices=DIRECTION_CHOICES, default='maximize')
    metric_name = models.CharField(max_length=100, default='eval_accuracy')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'experiment'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['project', 'status']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return self.name

    @property
    def best_trial(self):
        ordering = '-objective_value' if self.direction == 'maximize' else 'objective_value'
        return self.trials.exclude(objective_value__isnull=True).order_by(ordering).first()


class SweepTrial(models.Model):
    STATUS_CHOICES = (
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sweep = models.ForeignKey(Sweep, on_delete=models.CASCADE, related_name='trials')
    trial_number = models.IntegerField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    objective_value = models.FloatField(null=True, blank=True)
    params = models.JSONField(default=dict, blank=True)
    run = models.ForeignKey(TrainingRun, on_delete=models.SET_NULL, null=True, blank=True, related_name='sweep_trials')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'experiment'
        ordering = ['trial_number']
        unique_together = [['sweep', 'trial_number']]
        indexes = [
            models.Index(fields=['sweep', 'status']),
            models.Index(fields=['objective_value']),
        ]

    def __str__(self):
        return f"{self.sweep.name} trial {self.trial_number}"


# Compatibility name used by agents and sweeps modules.
Project = Experiment
