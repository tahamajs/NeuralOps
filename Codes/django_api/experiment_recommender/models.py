import uuid

from django.conf import settings
from django.db import models


class ExperimentTemplate(models.Model):
    """
    A reusable experiment template that can be recommended.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    category = models.CharField(
        max_length=50,
        default="general",
        choices=[
            ("training", "Training"),
            ("compression", "Compression"),
            ("benchmark", "Benchmark"),
            ("hyperparameter", "Hyperparameter Tuning"),
            ("architecture", "Architecture Search"),
            ("distillation", "Knowledge Distillation"),
            ("federated", "Federated Learning"),
        ],
    )
    config = models.JSONField(
        default=dict, help_text="Default experiment configuration"
    )
    predicted_impact = models.FloatField(
        default=0.0, help_text="Estimated performance lift (0–1)"
    )
    success_rate = models.FloatField(
        default=0.0, help_text="Historical success rate of this template"
    )
    usage_count = models.PositiveIntegerField(default=0)
    tags = models.JSONField(default=list, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="experiment_templates",
    )
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-success_rate", "-usage_count"]

    def __str__(self):
        return self.name


class OpportunityIndex(models.Model):
    """
    Calculated opportunity index for a user's research area.
    Identifies missing, high‑impact attributes.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="opportunity_indices",
    )
    attribute = models.CharField(max_length=200)
    current_score = models.FloatField(
        default=0.0, help_text="Current performance in this area"
    )
    potential_score = models.FloatField(
        default=0.0, help_text="Best possible performance"
    )
    gap = models.FloatField(default=0.0, help_text="Potential - Current")
    impact_weight = models.FloatField(
        default=1.0, help_text="How important this attribute is"
    )
    opportunity_score = models.FloatField(default=0.0, help_text="gap × impact_weight")
    recommendations = models.JSONField(
        default=list,
        blank=True,
        help_text="List of template IDs or actions to improve this attribute",
    )
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "attribute"]]
        ordering = ["-opportunity_score"]

    def __str__(self):
        return f"{self.user.username}: {self.attribute} (score: {self.opportunity_score:.2f})"


class ExperimentRecommendation(models.Model):
    """
    A recommended experiment for a user.
    """

    STATUS_CHOICES = [
        ("suggested", "Suggested"),
        ("viewed", "Viewed"),
        ("accepted", "Accepted"),
        ("dismissed", "Dismissed"),
        ("completed", "Completed"),
    ]
    REASON_CHOICES = [
        ("opportunity_gap", "Opportunity Gap"),
        ("similar_success", "Similar Successful Experiment"),
        ("trending", "Trending in Community"),
        ("bayesian_opt", "Bayesian Optimization"),
        ("llm_guided", "LLM Guidance"),
        ("creative_variant", "Creative Variant"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="experiment_recommendations",
    )
    template = models.ForeignKey(
        ExperimentTemplate,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="recommendations",
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    config = models.JSONField(
        default=dict, help_text="Recommended experiment configuration"
    )
    predicted_lift = models.FloatField(
        default=0.0, help_text="Predicted performance improvement (%)"
    )
    confidence = models.FloatField(
        default=0.0, help_text="Confidence in this recommendation (0–1)"
    )
    reason = models.CharField(
        max_length=30, choices=REASON_CHOICES, default="opportunity_gap"
    )
    status = models.CharField(
        max_length=20, choices=STATUS_CHOICES, default="suggested"
    )
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-predicted_lift", "-confidence"]

    def __str__(self):
        return f"Recommendation for {self.user.username}: {self.title}"


class CreativeVariant(models.Model):
    """
    A creative variant of an existing experiment — like an A/B test suggestion.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    base_experiment = models.ForeignKey(
        "experiment.TrainingRun",
        on_delete=models.CASCADE,
        related_name="creative_variants",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="creative_variants",
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    changes = models.JSONField(
        default=dict, help_text="Dict of parameter changes from base"
    )
    predicted_lift = models.FloatField(default=0.0)
    reasoning = models.TextField(blank=True)
    status = models.CharField(
        max_length=20,
        default="suggested",
        choices=ExperimentRecommendation.STATUS_CHOICES,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-predicted_lift"]

    def __str__(self):
        return f"Variant: {self.title}"
