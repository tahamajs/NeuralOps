import uuid

from django.conf import settings
from django.db import models


class FailureAnalysis(models.Model):
    """
    A single forensic analysis of a failed training run.
    """

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("analyzing", "Analyzing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.OneToOneField(
        "experiment.TrainingRun",
        on_delete=models.CASCADE,
        related_name="failure_analysis",
    )
    initiated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="failure_analyses",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    # Root cause classification
    root_cause_category = models.CharField(
        max_length=50,
        blank=True,
        choices=[
            ("configuration_error", "Configuration Error"),
            ("data_mismatch", "Data / Dataset Mismatch"),
            ("infrastructure_issue", "Infrastructure / Hardware Issue"),
            ("out_of_memory", "Out of Memory"),
            ("numerical_error", "Numerical Error (NaN / Inf)"),
            ("dependency_conflict", "Dependency / Package Conflict"),
            ("network_issue", "Network / Connectivity Issue"),
            ("unknown", "Unknown"),
        ],
    )
    root_cause_detail = models.TextField(blank=True)
    confidence = models.FloatField(default=0.0)

    # Log analysis results
    error_logs = models.JSONField(
        default=list, help_text="Extracted error lines from logs"
    )
    anomaly_patterns = models.JSONField(
        default=list, help_text="Detected anomaly patterns"
    )

    # Suggested fixes
    suggestions = models.JSONField(
        default=list, help_text="List of suggested corrective actions"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return (
            f"Forensics for {self.run.name or self.run.id} ({self.root_cause_category})"
        )


class FixAction(models.Model):
    """
    A concrete fix that can be applied with one click.
    """

    ACTION_TYPES = [
        ("restart", "Restart Training"),
        ("adjust_config", "Adjust Configuration"),
        ("reinstall_deps", "Reinstall Dependencies"),
        ("clear_cache", "Clear Cache / Temp Files"),
        ("increase_resources", "Increase Resources"),
        ("rollback", "Rollback to Previous Config"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis = models.ForeignKey(
        FailureAnalysis, on_delete=models.CASCADE, related_name="fix_actions"
    )
    action_type = models.CharField(max_length=30, choices=ACTION_TYPES)
    description = models.TextField()
    config_changes = models.JSONField(default=dict, blank=True)
    is_applied = models.BooleanField(default=False)
    applied_at = models.DateTimeField(null=True, blank=True)
    applied_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="applied_fixes",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.action_type}: {self.description[:80]}"


class ErrorPattern(models.Model):
    """
    A known error pattern learned from past failures.
    Used for pattern matching in new failures.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pattern_name = models.CharField(max_length=200, unique=True)
    regex_pattern = models.TextField(help_text="Regex pattern to match in logs")
    root_cause_category = models.CharField(
        max_length=50,
        choices=FailureAnalysis._meta.get_field("root_cause_category").choices,
    )
    description = models.TextField(blank=True)
    suggestion = models.TextField(blank=True)
    occurrences = models.PositiveIntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-occurrences"]

    def __str__(self):
        return self.pattern_name
