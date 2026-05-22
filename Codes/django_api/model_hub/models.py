"""
Central Model Hub & Governance System

Architecture inspired by:
  - JFrog AI Catalog: end-to-end model governance, security scanning, access control
  - MLflow Model Registry: versioning, stage transitions, approval workflows
  - Databricks Unity Catalog: fine-grained access control, lineage tracking
  - HuggingFace Model Hub: model cards, tags, discoverability

Taxonomy:
    RegisteredModel → ModelVersion → StageTransition → ApprovalRequest
    ModelUsage → AccessPolicy → SecurityScan → AuditLog
"""
import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator


# ═══════════════════════════════════════════════════════════════
# 1. REGISTERED MODEL — The central catalog entry
# ═══════════════════════════════════════════════════════════════
class RegisteredModel(models.Model):
    """A model registered in the central hub."""
    STAGE_CHOICES = [
        ('development', 'Development'),
        ('staging', 'Staging'),
        ('production', 'Production'),
        ('archived', 'Archived'),
    ]
    VISIBILITY_CHOICES = [
        ('public', 'Public — visible to all authenticated users'),
        ('private', 'Private — only owner and collaborators'),
        ('restricted', 'Restricted — only specific users/groups'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=300, unique=True)
    slug = models.SlugField(max_length=300, unique=True)
    description = models.TextField(blank=True)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='owned_models'
    )
    collaborators = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name='collab_models'
    )

    # Metadata
    tags = models.JSONField(default=list, blank=True)
    framework = models.CharField(max_length=100, blank=True, help_text="e.g., PyTorch, TensorFlow, JAX")
    task_type = models.CharField(max_length=100, blank=True, help_text="e.g., text-classification, image-segmentation")
    license = models.CharField(max_length=100, blank=True, default='mit')
    language = models.JSONField(default=list, blank=True, help_text="e.g., ['en', 'fa']")
    datasets = models.JSONField(default=list, blank=True, help_text="Datasets used for training")

    # Visibility & access
    visibility = models.CharField(max_length=20, choices=VISIBILITY_CHOICES, default='private')
    current_stage = models.CharField(max_length=20, choices=STAGE_CHOICES, default='development')

    # Stats
    version_count = models.PositiveIntegerField(default=0)
    download_count = models.PositiveIntegerField(default=0)
    like_count = models.PositiveIntegerField(default=0)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['owner']),
            models.Index(fields=['current_stage']),
            models.Index(fields=['visibility']),
        ]

    def __str__(self):
        return self.name

    @property
    def latest_version(self):
        return self.versions.order_by('-version_number').first()


# ═══════════════════════════════════════════════════════════════
# 2. MODEL VERSION — A specific version of a registered model
# ═══════════════════════════════════════════════════════════════
class ModelVersion(models.Model):
    """A specific versioned snapshot of a model."""
    STAGE_CHOICES = [
        ('none', 'None'),
        ('staging', 'Staging'),
        ('production', 'Production'),
        ('archived', 'Archived'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(RegisteredModel, on_delete=models.CASCADE, related_name='versions')
    version_number = models.PositiveIntegerField()
    version_string = models.CharField(max_length=100, help_text="e.g., 'v1.0.0', '1.2.3'")
    description = models.TextField(blank=True)
    stage = models.CharField(max_length=20, choices=STAGE_CHOICES, default='none')

    # Artifact
    artifact_path = models.CharField(max_length=500, blank=True)
    artifact_size_mb = models.FloatField(default=0.0)
    framework_version = models.CharField(max_length=50, blank=True)
    python_version = models.CharField(max_length=20, blank=True)

    # Metrics
    metrics = models.JSONField(default=dict, blank=True)
    parameters = models.JSONField(default=dict, blank=True)

    # Source
    training_run = models.ForeignKey(
        'jobscheduler.Job', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='model_versions'
    )
    git_commit = models.CharField(max_length=64, blank=True)

    # Approval
    is_approved = models.BooleanField(default=False)
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='approved_versions'
    )
    approved_at = models.DateTimeField(null=True, blank=True)

    # Security scan results
    security_scan_status = models.CharField(max_length=20, default='pending',
        choices=[('pending','Pending'), ('scanning','Scanning'), ('passed','Passed'),
                 ('failed','Failed'), ('warning','Warning')])
    security_scan_summary = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['model', 'version_number'], ['model', 'version_string']]
        ordering = ['-version_number']

    def __str__(self):
        return f"{self.model.name} {self.version_string}"


# ═══════════════════════════════════════════════════════════════
# 3. STAGE TRANSITION — Audit trail for stage changes
# ═══════════════════════════════════════════════════════════════
class StageTransition(models.Model):
    """A record of a model version changing stages."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='stage_transitions')
    from_stage = models.CharField(max_length=20)
    to_stage = models.CharField(max_length=20)
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='stage_transitions_requested'
    )
    approved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='stage_transitions_approved'
    )
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.version}: {self.from_stage} → {self.to_stage}"


# ═══════════════════════════════════════════════════════════════
# 4. APPROVAL REQUEST — Workflow for promoting versions
# ═══════════════════════════════════════════════════════════════
class ApprovalRequest(models.Model):
    """A request to change a model version's stage or approve it."""
    STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
        ('withdrawn', 'Withdrawn'),
    ]
    TYPE_CHOICES = [
        ('promote', 'Promote to Production'),
        ('approve', 'Approve Version'),
        ('publish', 'Make Public'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='approval_requests')
    request_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='approval_requests'
    )
    reviewers = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name='review_assignments'
    )
    comment = models.TextField(blank=True)
    resolution_comment = models.TextField(blank=True)
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='resolved_approvals'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Approval for {self.version}: {self.request_type} ({self.status})"


# ═══════════════════════════════════════════════════════════════
# 5. MODEL USAGE — Track downloads and inferences
# ═══════════════════════════════════════════════════════════════
class ModelUsage(models.Model):
    """A record of model usage (download, inference, deployment)."""
    USAGE_TYPES = [
        ('download', 'Download'),
        ('inference', 'Inference'),
        ('deployment', 'Deployment'),
        ('fine_tune', 'Fine-Tuning'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='usage_records')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='model_usage')
    usage_type = models.CharField(max_length=20, choices=USAGE_TYPES)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [models.Index(fields=['version', 'usage_type'])]


# ═══════════════════════════════════════════════════════════════
# 6. ACCESS POLICY — Fine-grained access control
# ═══════════════════════════════════════════════════════════════
class AccessPolicy(models.Model):
    """Defines who can access a model and what they can do."""
    PERMISSION_CHOICES = [
        ('read', 'Read — view and download'),
        ('write', 'Write — upload new versions'),
        ('admin', 'Admin — manage access and delete'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(RegisteredModel, on_delete=models.CASCADE, related_name='access_policies')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True,
                             related_name='model_access')
    group = models.CharField(max_length=100, blank=True, help_text="e.g., 'ml-team', 'reviewers'")
    permission = models.CharField(max_length=20, choices=PERMISSION_CHOICES, default='read')
    granted_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                   related_name='granted_access')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['model', 'user', 'permission'], ['model', 'group', 'permission']]

    def __str__(self):
        return f"{self.model.name}: {self.user or self.group} → {self.permission}"


# ═══════════════════════════════════════════════════════════════
# 7. SECURITY SCAN — Vulnerability and compliance checks
# ═══════════════════════════════════════════════════════════════
class SecurityScan(models.Model):
    """A security scan of a model version."""
    SCAN_TYPES = [
        ('vulnerability', 'Vulnerability Scan'),
        ('license', 'License Compliance'),
        ('sensitive_data', 'Sensitive Data Detection'),
        ('model_integrity', 'Model Integrity Check'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.ForeignKey(ModelVersion, on_delete=models.CASCADE, related_name='security_scans')
    scan_type = models.CharField(max_length=30, choices=SCAN_TYPES)
    status = models.CharField(max_length=20, default='pending',
        choices=[('pending','Pending'), ('running','Running'), ('passed','Passed'),
                 ('failed','Failed'), ('warning','Warning')])
    findings = models.JSONField(default=dict, blank=True)
    score = models.FloatField(null=True, blank=True, validators=[MinValueValidator(0), MaxValueValidator(100)])
    scanned_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']


# ═══════════════════════════════════════════════════════════════
# 8. AUDIT LOG — Immutable governance trail
# ═══════════════════════════════════════════════════════════════
class AuditLog(models.Model):
    """Immutable audit log for model governance."""
    ACTION_CHOICES = [
        ('model_created', 'Model Created'),
        ('version_uploaded', 'Version Uploaded'),
        ('stage_changed', 'Stage Changed'),
        ('approval_requested', 'Approval Requested'),
        ('approval_granted', 'Approval Granted'),
        ('approval_rejected', 'Approval Rejected'),
        ('access_granted', 'Access Granted'),
        ('access_revoked', 'Access Revoked'),
        ('model_downloaded', 'Model Downloaded'),
        ('model_deployed', 'Model Deployed'),
        ('security_scan_completed', 'Security Scan Completed'),
        ('model_archived', 'Model Archived'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(RegisteredModel, on_delete=models.CASCADE, related_name='audit_logs')
    version = models.ForeignKey(ModelVersion, on_delete=models.SET_NULL, null=True, blank=True,
                                related_name='audit_logs')
    action = models.CharField(max_length=30, choices=ACTION_CHOICES)
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    description = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['model', '-created_at']),
            models.Index(fields=['action']),
        ]