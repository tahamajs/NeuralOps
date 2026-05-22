import uuid
from django.db import models
from django.conf import settings


class MarketplaceQuery(models.Model):
    """
    A user's natural language query for GPU discovery.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='marketplace_queries'
    )
    query_text = models.TextField(help_text="Natural language query, e.g. 'Cheapest A100 with 80GB VRAM under $1/hr'")
    parsed_filters = models.JSONField(default=dict, blank=True, help_text="Structured filters extracted from query")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    result_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = "Marketplace Queries"

    def __str__(self):
        return f"Query by {self.user.username}: {self.query_text[:60]}"


class ComparisonSet(models.Model):
    """
    A saved comparison between multiple GPU listings.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='comparison_sets'
    )
    name = models.CharField(max_length=200, blank=True)
    listings = models.ManyToManyField(
        'gpu_marketplace.GPUListing',
        related_name='comparison_sets'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Comparison by {self.user.username}: {self.name or 'Untitled'}"


class ComplianceCheck(models.Model):
    """
    A compliance check result for a GPU listing against user requirements.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='compliance_checks'
    )
    listing = models.ForeignKey(
        'gpu_marketplace.GPUListing',
        on_delete=models.CASCADE,
        related_name='compliance_checks'
    )
    requirements = models.JSONField(default=dict, help_text="User requirements e.g. {'min_vram': 80, 'max_price': 1.50}")
    passed = models.BooleanField(default=False)
    failures = models.JSONField(default=list, blank=True, help_text="List of requirements that failed")
    warnings = models.JSONField(default=list, blank=True, help_text="List of requirements that are borderline")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Compliance for {self.listing.gpu_model}: {'PASS' if self.passed else 'FAIL'}"


class SmartRecommendation(models.Model):
    """
    An AI‑generated recommendation for a user.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='smart_recommendations'
    )
    listing = models.ForeignKey(
        'gpu_marketplace.GPUListing',
        on_delete=models.CASCADE,
        related_name='smart_recommendations'
    )
    score = models.FloatField(default=0.0)
    reason = models.TextField(blank=True)
    is_dismissed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-score']
        unique_together = [['user', 'listing']]

    def __str__(self):
        return f"Recommend {self.listing.gpu_model} to {self.user.username}"