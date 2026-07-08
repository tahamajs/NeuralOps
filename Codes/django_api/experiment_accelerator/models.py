import uuid

from django.conf import settings
from django.db import models


class ExperimentInsight(models.Model):
    """
    An AI‑generated insight from experiment history.
    """

    INSIGHT_TYPES = [
        ("trend", "Trend Detected"),
        ("recommendation", "Actionable Recommendation"),
        ("anomaly", "Anomaly Detected"),
        ("winner", "Winning Pattern"),
        ("optimization", "Optimization Suggestion"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="experiment_insights",
    )
    insight_type = models.CharField(max_length=20, choices=INSIGHT_TYPES)
    title = models.CharField(max_length=300)
    description = models.TextField()
    confidence = models.FloatField(default=0.0, help_text="0–1")
    related_runs = models.JSONField(
        default=list,
        blank=True,
        help_text="List of run IDs that contributed to this insight",
    )
    metadata = models.JSONField(default=dict, blank=True)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["insight_type"]),
        ]

    def __str__(self):
        return f"{self.insight_type}: {self.title}"


class ExperimentRecommendation(models.Model):
    """
    A suggested next experiment with predicted outcome.
    """

    STATUS_CHOICES = [
        ("suggested", "Suggested"),
        ("launched", "Launched"),
        ("dismissed", "Dismissed"),
        ("completed", "Completed"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="experiment_recommendations",
    )
    title = models.CharField(max_length=300)
    description = models.TextField()
    predicted_improvement = models.FloatField(
        default=0.0, help_text="Predicted % improvement over current best"
    )
    confidence = models.FloatField(default=0.0)
    configuration = models.JSONField(
        default=dict, blank=True, help_text="Suggested experiment configuration"
    )
    rationale = models.TextField(
        blank=True, help_text="Why this experiment is recommended"
    )
    based_on_runs = models.JSONField(default=list, blank=True)
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="suggested"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-predicted_improvement"]

    def __str__(self):
        return self.title


class ExperimentSummary(models.Model):
    """
    A periodic summary of experiment performance.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="experiment_summaries",
    )
    period_start = models.DateField()
    period_end = models.DateField()
    total_experiments = models.PositiveIntegerField(default=0)
    successful_experiments = models.PositiveIntegerField(default=0)
    best_accuracy = models.FloatField(null=True, blank=True)
    best_loss = models.FloatField(null=True, blank=True)
    top_recommendations = models.JSONField(default=list, blank=True)
    insights_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-period_end"]

    def __str__(self):
        return f"Summary for {self.user.username} ({self.period_end})"
