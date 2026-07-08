import uuid

from django.conf import settings
from django.core.validators import MinValueValidator
from django.db import models


class ReputationWeight(models.Model):
    """
    Configurable weights for each reputation component.
    Singleton pattern – only one row should exist.
    """

    OTHER_READS = models.FloatField(default=5.0, validators=[MinValueValidator(0)])
    FULL_TEXT_READS = models.FloatField(default=15.0, validators=[MinValueValidator(0)])
    RECOMMENDATIONS = models.FloatField(default=25.0, validators=[MinValueValidator(0)])
    CITATIONS = models.FloatField(default=50.0, validators=[MinValueValidator(0)])
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Reputation Weight Configuration"

    def __str__(self):
        return f"Weights: OR={self.OTHER_READS}% FTR={self.FULL_TEXT_READS}% REC={self.RECOMMENDATIONS}% CIT={self.CITATIONS}%"

    def save(self, *args, **kwargs):
        # Ensure only one row exists
        self.pk = 1
        super().save(*args, **kwargs)

    @classmethod
    def get_weights(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return {
            "OTHER_READS": obj.OTHER_READS,
            "FULL_TEXT_READS": obj.FULL_TEXT_READS,
            "RECOMMENDATIONS": obj.RECOMMENDATIONS,
            "CITATIONS": obj.CITATIONS,
        }


class ReputationScore(models.Model):
    """
    Overall reputation score for a user.
    Calculated from the sum of weighted events.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reputation_score",
    )
    total_score = models.FloatField(default=0.0)
    # Component scores
    other_reads_score = models.FloatField(default=0.0)
    full_text_reads_score = models.FloatField(default=0.0)
    recommendations_score = models.FloatField(default=0.0)
    citations_score = models.FloatField(default=0.0)

    # Stats
    total_papers = models.PositiveIntegerField(default=0)
    total_citations = models.PositiveIntegerField(default=0)
    total_recommendations = models.PositiveIntegerField(default=0)
    total_reads = models.PositiveIntegerField(default=0)

    # Ranking
    percentile = models.FloatField(null=True, blank=True)

    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-total_score"]

    def __str__(self):
        return f"{self.user.username}: {self.total_score:.2f}"

    def recalculate(self):
        """Recalculate the total score from all events."""
        events = self.user.reputation_events.all()
        weights = ReputationWeight.get_weights()

        self.other_reads_score = sum(
            e.weighted_value for e in events.filter(event_type="other_read")
        )
        self.full_text_reads_score = sum(
            e.weighted_value for e in events.filter(event_type="full_text_read")
        )
        self.recommendations_score = sum(
            e.weighted_value for e in events.filter(event_type="recommendation")
        )
        self.citations_score = sum(
            e.weighted_value for e in events.filter(event_type="citation")
        )

        self.total_score = (
            self.other_reads_score
            + self.full_text_reads_score
            + self.recommendations_score
            + self.citations_score
        )

        self.total_reads = events.filter(
            event_type__in=["other_read", "full_text_read"]
        ).count()
        self.total_recommendations = events.filter(event_type="recommendation").count()
        self.total_citations = events.filter(event_type="citation").count()
        self.total_papers = self.user.saved_papers.count()

        self.save()


class ReputationEvent(models.Model):
    """
    A single event that contributes to a user's reputation.
    """

    EVENT_TYPES = [
        ("other_read", "Other Read"),
        ("full_text_read", "Full-Text Read"),
        ("recommendation", "Recommendation"),
        ("citation", "Citation"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reputation_events",
    )
    event_type = models.CharField(max_length=20, choices=EVENT_TYPES)
    paper = models.ForeignKey(
        "agents.ResearchPaper",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reputation_events",
    )
    source_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="given_reputation_events",
    )
    weight = models.FloatField(
        help_text="Weight of this event type at the time of creation"
    )
    weighted_value = models.FloatField(
        default=1.0,
        help_text="Normalized contribution (1.0 = one unit of this event type)",
    )
    description = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "event_type"]),
            models.Index(fields=["paper"]),
        ]

    def __str__(self):
        return f"{self.user.username} +{self.weighted_value:.2f} ({self.event_type})"

    def save(self, *args, **kwargs):
        if not self.weight:
            weights = ReputationWeight.get_weights()
            weight_map = {
                "other_read": weights["OTHER_READS"],
                "full_text_read": weights["FULL_TEXT_READS"],
                "recommendation": weights["RECOMMENDATIONS"],
                "citation": weights["CITATIONS"],
            }
            self.weight = weight_map.get(self.event_type, 0)
        if not self.weighted_value:
            self.weighted_value = self.weight / 100.0  # Normalize to 0–1 per event
        super().save(*args, **kwargs)

        # Update the user's reputation score
        score, _ = ReputationScore.objects.get_or_create(user=self.user)
        score.recalculate()


class Recommendation(models.Model):
    """
    Context-aware recommendation for a user.
    Generated based on semantic content analysis and reputation.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reputation_user_recommendations",
    )
    paper = models.ForeignKey(
        "agents.ResearchPaper", on_delete=models.CASCADE, related_name="reputation_paper_recommendations"
    )
    score = models.FloatField(default=0.0, help_text="Recommendation strength (0–1)")
    reason = models.TextField(blank=True)
    is_dismissed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-score", "-created_at"]
        unique_together = [["user", "paper"]]

    def __str__(self):
        return f"Recommend {self.paper.title[:50]} to {self.user.username}"


class ReputationSnapshot(models.Model):
    """
    Daily snapshot of reputation scores for historical tracking.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reputation_snapshots",
    )
    total_score = models.FloatField()
    other_reads_score = models.FloatField()
    full_text_reads_score = models.FloatField()
    recommendations_score = models.FloatField()
    citations_score = models.FloatField()
    percentile = models.FloatField(null=True, blank=True)
    recorded_at = models.DateField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "recorded_at"]]
        ordering = ["-recorded_at"]
