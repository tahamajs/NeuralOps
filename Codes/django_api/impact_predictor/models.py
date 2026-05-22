import uuid
from django.db import models
from django.conf import settings


class ResearchProject(models.Model):
    """
    A research project submitted for impact prediction.
    """
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('submitted', 'Submitted'),
        ('scored', 'Scored'),
        ('funded', 'Funded'),
        ('rejected', 'Rejected'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='impact_projects'
    )
    title = models.CharField(max_length=500)
    abstract = models.TextField(blank=True)
    domain = models.CharField(max_length=200, blank=True)

    # Institution characteristics
    institution_name = models.CharField(max_length=300, blank=True)
    institution_age_years = models.PositiveIntegerField(null=True, blank=True)
    institution_location = models.CharField(max_length=200, blank=True)

    # Investigator quality
    investigator_h_index = models.PositiveIntegerField(null=True, blank=True)
    investigator_publications = models.PositiveIntegerField(null=True, blank=True)
    investigator_previous_grants = models.PositiveIntegerField(null=True, blank=True)

    # Digital / social traction
    social_followers = models.PositiveIntegerField(null=True, blank=True)
    preprint_views = models.PositiveIntegerField(null=True, blank=True)
    online_mentions = models.PositiveIntegerField(null=True, blank=True)

    # Funding history
    previous_funding_total = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True
    )
    previous_grants_count = models.PositiveIntegerField(null=True, blank=True)

    # Status and metadata
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title


class ImpactPrediction(models.Model):
    """
    A prediction result for a research project.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.OneToOneField(
        ResearchProject,
        on_delete=models.CASCADE,
        related_name='impact_prediction'
    )
    impact_score = models.FloatField(
        help_text="Predicted impact score (0–100)"
    )
    confidence_lower = models.FloatField()
    confidence_upper = models.FloatField()
    percentile = models.FloatField(
        help_text="Percentile rank among all projects"
    )
    funding_probability = models.FloatField(
        null=True, blank=True,
        help_text="Estimated probability of receiving funding"
    )
    publication_tier = models.CharField(
        max_length=50, blank=True,
        help_text="Predicted publication tier (e.g., 'Top 5%', 'Top 25%')"
    )
    comparison_text = models.TextField(blank=True)
    factor_scores = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Impact {self.impact_score:.1f} for {self.project.title[:50]}"