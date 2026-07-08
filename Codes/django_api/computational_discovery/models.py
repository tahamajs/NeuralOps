import uuid

from django.conf import settings
from django.db import models


class CodeVariant(models.Model):
    """
    A single variant of ML experiment code.
    """

    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("queued", "Queued for Testing"),
        ("testing", "Testing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("deprecated", "Deprecated"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    experiment = models.ForeignKey(
        "CodeExperiment", on_delete=models.CASCADE, related_name="variants"
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="code_variants"
    )
    version_number = models.PositiveIntegerField(default=1)
    code = models.TextField(help_text="Full Python/ML code of this variant")
    description = models.TextField(blank=True)
    hyperparameters = models.JSONField(default=dict)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")
    score = models.FloatField(null=True, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    parent_variant = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="children",
    )
    iteration = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["experiment", "version_number"]

    def __str__(self):
        return f"Variant {self.version_number} of {self.experiment.name}"


class CodeExperiment(models.Model):
    """
    A computational experiment that generates and tests multiple code variants.
    """

    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("generating", "Generating Variants"),
        ("testing", "Testing"),
        ("completed", "Completed"),
        ("paused", "Paused"),
        ("failed", "Failed"),
    ]
    MODE_CHOICES = [
        ("auto", "Fully Automatic"),
        ("guided", "Guided (with human feedback)"),
        ("evolutionary", "Evolutionary (AlphaEvolve-style)"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="code_experiments",
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    base_code = models.TextField(blank=True, help_text="Starting code / template")
    objective = models.TextField(
        blank=True, help_text="What this experiment aims to optimize"
    )
    mode = models.CharField(max_length=20, choices=MODE_CHOICES, default="auto")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")

    # Experiment config
    max_variants = models.PositiveIntegerField(default=100)
    max_iterations = models.PositiveIntegerField(default=10)
    parallel_workers = models.PositiveIntegerField(default=4)
    metric_to_optimize = models.CharField(max_length=100, default="accuracy")
    metric_direction = models.CharField(
        max_length=10,
        default="maximize",
        choices=[("maximize", "Maximize"), ("minimize", "Minimize")],
    )
    target_score = models.FloatField(null=True, blank=True)

    # Results
    best_variant = models.ForeignKey(
        CodeVariant,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="best_for_experiment",
    )
    total_variants_tested = models.PositiveIntegerField(default=0)
    total_iterations = models.PositiveIntegerField(default=0)
    elapsed_seconds = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class ExperimentRun(models.Model):
    """
    A single test run of a code variant against a dataset.
    """

    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    variant = models.ForeignKey(
        CodeVariant, on_delete=models.CASCADE, related_name="runs"
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")
    metrics = models.JSONField(default=dict, blank=True)
    logs = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Run for {self.variant} ({self.status})"
