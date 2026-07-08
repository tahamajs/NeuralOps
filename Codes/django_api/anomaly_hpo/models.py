import uuid

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class ExperimentLogAnalysis(models.Model):
    """
    LLM-powered analysis of experiment logs.
    Detects anomalies, patterns, and issues.
    """

    ANOMALY_TYPES = [
        ("loss_spike", "Loss Spike"),
        ("loss_plateau", "Loss Plateau"),
        ("divergence", "Divergence"),
        ("gradient_explosion", "Gradient Explosion"),
        ("gradient_vanishing", "Gradient Vanishing"),
        ("memory_leak", "Memory Leak"),
        ("slow_convergence", "Slow Convergence"),
        ("overfitting", "Overfitting"),
        ("underfitting", "Underfitting"),
        ("hardware_issue", "Hardware Issue"),
        ("data_corruption", "Data Corruption"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        "experiment.TrainingRun", on_delete=models.CASCADE, related_name="log_analyses"
    )
    anomaly_type = models.CharField(max_length=30, choices=ANOMALY_TYPES)
    severity = models.FloatField(
        default=0.5, validators=[MinValueValidator(0), MaxValueValidator(1)]
    )
    detected_at_step = models.IntegerField(null=True, blank=True)
    description = models.TextField()
    suggested_fix = models.TextField(blank=True)
    auto_fix_applied = models.BooleanField(default=False)
    fix_successful = models.BooleanField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["run", "-created_at"]),
            models.Index(fields=["anomaly_type"]),
        ]

    def __str__(self):
        return f"{self.anomaly_type} in {self.run.name or self.run.id} (severity: {self.severity:.2f})"


class HPOTrial(models.Model):
    """
    A single trial in a hyperparameter optimization run.
    Uses SMAC-style Bayesian optimization.
    """

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("canceled", "Canceled"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    hpo_run = models.ForeignKey(
        "HPORun", on_delete=models.CASCADE, related_name="trials"
    )
    trial_number = models.PositiveIntegerField()
    params = models.JSONField(default=dict, help_text="Hyperparameters for this trial")
    objective_value = models.FloatField(
        null=True, blank=True, help_text="Metric being optimized"
    )
    objective_metric = models.CharField(max_length=50, default="eval_accuracy")
    direction = models.CharField(
        max_length=10,
        default="maximize",
        choices=[("maximize", "Maximize"), ("minimize", "Minimize")],
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    run = models.ForeignKey(
        "experiment.TrainingRun",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="hpo_trials",
    )
    error_message = models.TextField(blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["hpo_run", "trial_number"]]
        ordering = ["hpo_run", "trial_number"]

    def __str__(self):
        return f"Trial {self.trial_number} of {self.hpo_run.name} ({self.status})"


class HPORun(models.Model):
    """
    A complete hyperparameter optimization run.
    Uses SMAC (Sequential Model-based Algorithm Configuration).
    """

    STRATEGY_CHOICES = [
        ("smac", "SMAC"),
        ("bayesian", "Bayesian Optimization"),
        ("random", "Random Search"),
        ("grid", "Grid Search"),
        ("evolutionary", "Evolutionary"),
    ]
    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("canceled", "Canceled"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="hpo_runs"
    )
    name = models.CharField(max_length=200)
    strategy = models.CharField(max_length=20, choices=STRATEGY_CHOICES, default="smac")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")
    experiment = models.ForeignKey(
        "experiment.Experiment",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="hpo_runs",
    )

    # Search space
    search_space = models.JSONField(
        default=dict,
        help_text="""
        {
          "learning_rate": {"type": "float", "min": 1e-5, "max": 1e-2, "log": true},
          "batch_size": {"type": "int", "min": 8, "max": 64},
          "dropout": {"type": "float", "min": 0.0, "max": 0.5},
          "attention_type": {"type": "categorical", "choices": ["full", "galerkin", "linformer"]}
        }
    """,
    )
    objective_metric = models.CharField(max_length=50, default="eval_accuracy")
    objective_direction = models.CharField(
        max_length=10,
        default="maximize",
        choices=[("maximize", "Maximize"), ("minimize", "Minimize")],
    )
    max_trials = models.PositiveIntegerField(default=20)
    base_config = models.JSONField(
        default=dict, help_text="Base experiment configuration"
    )

    # Results
    best_objective_value = models.FloatField(null=True, blank=True)
    best_trial = models.OneToOneField(
        HPOTrial,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="best_of_hpo",
    )
    best_params = models.JSONField(default=dict, blank=True)

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name
