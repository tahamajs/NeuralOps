import hashlib
import json
import uuid

from django.conf import settings
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone


class Dataset(models.Model):
    """
    A versioned dataset with DVC integration.
    """
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('published', 'Published'),
        ('deprecated', 'Deprecated'),
        ('archived', 'Archived'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='owned_datasets'
    )
    collaborators = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name='shared_datasets'
    )
    is_public = models.BooleanField(default=False)

    # DVC Integration
    dvc_remote = models.URLField(blank=True, help_text="DVC remote storage URL (S3, GCS, etc.)")
    dvc_revision = models.CharField(max_length=64, blank=True, help_text="Current DVC revision hash")

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')

    # Metadata
    tags = models.JSONField(default=list, blank=True)
    license = models.CharField(max_length=100, blank=True)
    citation = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return self.name

    @property
    def latest_version(self):
        return self.versions.order_by('-version_number').first()

    @property
    def size_bytes(self):
        return self.versions.aggregate(s=models.Sum('size_bytes'))['s'] or 0

    @property
    def row_count(self):
        latest = self.latest_version
        return latest.row_count if latest else 0


class DatasetVersion(models.Model):
    """
    A specific version of a dataset.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='versions')
    version_number = models.PositiveIntegerField()
    description = models.TextField(blank=True)

    # Storage
    file_path = models.CharField(max_length=500, blank=True, help_text="Local or remote file path")
    size_bytes = models.PositiveIntegerField(default=0)
    row_count = models.PositiveIntegerField(default=0)
    column_count = models.PositiveIntegerField(default=0)

    # Schema
    schema = models.JSONField(default=dict, blank=True, help_text="Column names and inferred types")
    sample_data = models.JSONField(default=list, blank=True, help_text="First 5 rows as JSON")

    # DVC
    dvc_hash = models.CharField(max_length=64, blank=True, help_text="DVC file hash (md5)")
    dvc_lock = models.TextField(blank=True, help_text="Contents of the .dvc lock file")

    # Checksums
    sha256 = models.CharField(max_length=64, blank=True)
    md5 = models.CharField(max_length=32, blank=True)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='dataset_versions'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['dataset', 'version_number']]
        ordering = ['-version_number']

    def __str__(self):
        return f"{self.dataset.name} v{self.version_number}"


class DatasetEmbedding(models.Model):
    """
    Semantic search embeddings for datasets (description + schema).
    Enables natural language dataset discovery.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.OneToOneField(Dataset, on_delete=models.CASCADE, related_name='embedding')
    embedding = models.JSONField(default=list, help_text="Vector embedding (e.g., 1536-dim from OpenAI)")
    model_used = models.CharField(max_length=100, default='text-embedding-3-small')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']


class DatasetDownload(models.Model):
    """Track dataset downloads for analytics."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='downloads')
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='dataset_downloads'
    )
    version = models.ForeignKey(DatasetVersion, on_delete=models.SET_NULL, null=True)
    downloaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-downloaded_at']


class DatasetReview(models.Model):
    """Community reviews of datasets."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='reviews')
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='dataset_reviews'
    )
    rating = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['dataset', 'reviewer']]


# ═══════════════════════════════════════════════════════════════
# ARTIFACT & LINEAGE TRACKING
# ═══════════════════════════════════════════════════════════════
class Artifact(models.Model):
    """
    Any file, model, dataset, or result tracked by the platform.
    Inspired by MLflow's artifact tracking.
    """
    ARTIFACT_TYPES = [
        ('dataset', 'Dataset'),
        ('model', 'Trained Model'),
        ('checkpoint', 'Checkpoint'),
        ('log', 'Log File'),
        ('plot', 'Plot / Chart'),
        ('report', 'Report'),
        ('config', 'Configuration'),
        ('code', 'Source Code'),
        ('other', 'Other'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=300)
    artifact_type = models.CharField(max_length=20, choices=ARTIFACT_TYPES, default='other')
    run = models.ForeignKey(
        'jobscheduler.Job', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='artifacts'
    )
    file_path = models.CharField(max_length=500, blank=True)
    file_url = models.URLField(blank=True, max_length=1000)
    size_bytes = models.PositiveIntegerField(default=0)

    # Checksums
    sha256 = models.CharField(max_length=64, blank=True)
    md5 = models.CharField(max_length=32, blank=True)

    # Metadata
    metadata = models.JSONField(default=dict, blank=True)
    tags = models.JSONField(default=list, blank=True)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='created_artifacts'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['run', '-created_at']),
            models.Index(fields=['artifact_type']),
        ]

    def __str__(self):
        return f"{self.artifact_type}: {self.name}"


class ArtifactLineage(models.Model):
    """
    Tracks which artifact was derived from which.
    Builds a DAG of artifact dependencies.
    """
    RELATIONSHIP_TYPES = [
        ('derived_from', 'Derived From'),
        ('produced_by', 'Produced By'),
        ('depends_on', 'Depends On'),
        ('evaluated_on', 'Evaluated On'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source = models.ForeignKey(Artifact, on_delete=models.CASCADE, related_name='downstream_edges')
    target = models.ForeignKey(Artifact, on_delete=models.CASCADE, related_name='upstream_edges')
    relationship_type = models.CharField(max_length=20, choices=RELATIONSHIP_TYPES)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['source', 'target', 'relationship_type']]

    def __str__(self):
        return f"{self.source.name} → {self.target.name} ({self.relationship_type})"


# ═══════════════════════════════════════════════════════════════
# COMPLIANCE & AUDIT LOGGING
# ═══════════════════════════════════════════════════════════════
class AuditLog(models.Model):
    """
    Immutable activity log for compliance (GDPR, HIPAA, SOC2).
    Records who did what, when, on which resource, with what data.
    """
    ACTION_TYPES = [
        ('create', 'Created'),
        ('update', 'Updated'),
        ('delete', 'Deleted'),
        ('read', 'Read'),
        ('download', 'Downloaded'),
        ('share', 'Shared'),
        ('publish', 'Published'),
        ('archive', 'Archived'),
        ('export', 'Exported'),
        ('import', 'Imported'),
        ('login', 'Logged In'),
        ('logout', 'Logged Out'),
        ('permission_change', 'Permission Changed'),
        ('config_change', 'Configuration Changed'),
    ]
    SEVERITY_LEVELS = [
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
        ('critical', 'Critical'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='audit_logs'
    )
    action_type = models.CharField(max_length=20, choices=ACTION_TYPES)
    resource_type = models.CharField(max_length=100, help_text="e.g., 'dataset', 'model', 'experiment'")
    resource_id = models.CharField(max_length=100, blank=True, help_text="UUID of the affected resource")
    description = models.TextField(blank=True)

    # Before / after snapshots (for update actions)
    data_before = models.JSONField(default=dict, blank=True)
    data_after = models.JSONField(default=dict, blank=True)

    # Compliance fields
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS, default='info')
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    session_id = models.CharField(max_length=100, blank=True)
    request_id = models.CharField(max_length=100, blank=True, help_text="X-Request-ID header")

    # Immutability
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['actor', '-created_at']),
            models.Index(fields=['action_type', '-created_at']),
            models.Index(fields=['resource_type', 'resource_id']),
            models.Index(fields=['severity']),
        ]
        get_latest_by = 'created_at'

    def __str__(self):
        return f"[{self.action_type}] {self.actor} on {self.resource_type}:{self.resource_id}"

    def save(self, *args, **kwargs):
        # Enforce immutability: only allow creation, never updates
        if self.pk is not None:
            raise RuntimeError("AuditLog entries are immutable. Create a new entry instead.")
        super().save(*args, **kwargs)


class AuditExport(models.Model):
    """
    A scheduled or ad‑hoc export of audit logs for compliance reporting.
    """
    FORMAT_CHOICES = [('csv', 'CSV'), ('json', 'JSON'), ('pdf', 'PDF')]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='audit_exports'
    )
    format = models.CharField(max_length=10, choices=FORMAT_CHOICES, default='csv')
    filters = models.JSONField(default=dict, blank=True, help_text="Filters applied to the export")
    file = models.FileField(upload_to='audit_exports/', blank=True, null=True)
    status = models.CharField(max_length=20, default='pending',
                               choices=[('pending','Pending'),('processing','Processing'),
                                        ('completed','Completed'),('failed','Failed')])
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']