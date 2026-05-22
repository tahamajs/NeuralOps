import uuid
from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator

# ── 1. Audit Log Dashboard ──────────────────────────────────────
# Re‑uses organizations.AuditLog. We add a saved filter for quick access.
class AuditSavedFilter(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='audit_filters')
    name = models.CharField(max_length=200)
    filters = models.JSONField(default=dict)  # {action: 'member_added', user_id: ...}
    created_at = models.DateTimeField(auto_now_add=True)

# ── 2. Automated AI Compliance Agent ────────────────────────────
class ComplianceRule(models.Model):
    """A rule that the AI compliance agent checks."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    rule_type = models.CharField(max_length=50, choices=[
        ('data_privacy', 'Data Privacy'),
        ('model_card', 'Model Card'),
        ('bias_detection', 'Bias Detection'),
        ('export_control', 'Export Control'),
        ('gpu_usage', 'GPU Usage Limits'),
        ('custom', 'Custom'),
    ])
    check_function = models.CharField(max_length=300, help_text="Dotted path to validator function")
    severity = models.CharField(max_length=20, default='warning', choices=[('info','Info'),('warning','Warning'),('critical','Critical')])
    is_active = models.BooleanField(default=True)

class ComplianceCheck(models.Model):
    """Result of a compliance check against a run/model."""
    rule = models.ForeignKey(ComplianceRule, on_delete=models.CASCADE)
    content_type = models.ForeignKey('contenttypes.ContentType', on_delete=models.CASCADE)
    object_id = models.UUIDField()
    passed = models.BooleanField(null=True)
    details = models.JSONField(default=dict)
    checked_at = models.DateTimeField(auto_now_add=True)

class ComplianceReport(models.Model):
    """An auto‑generated compliance report."""
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='compliance_reports')
    title = models.CharField(max_length=300)
    period_start = models.DateField()
    period_end = models.DateField()
    content_markdown = models.TextField()
    generated_at = models.DateTimeField(auto_now_add=True)

# ── 3. Data Contracts & Agreements ─────────────────────────────
class DataContract(models.Model):
    """A Service Level Agreement (SLA) or Terms of Service (ToS)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    provider_org = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, related_name='contracts_as_provider')
    consumer_org = models.ForeignKey('organizations.Organization', on_delete=models.CASCADE, related_name='contracts_as_consumer')
    title = models.CharField(max_length=300)
    contract_type = models.CharField(max_length=50, choices=[('sla','SLA'),('tos','ToS'),('data_usage','Data Usage Agreement')])
    content = models.TextField()
    signed_by_provider = models.BooleanField(default=False)
    signed_by_consumer = models.BooleanField(default=False)
    effective_from = models.DateField()
    expires_at = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class SLAMetric(models.Model):
    """A measurable metric within an SLA."""
    contract = models.ForeignKey(DataContract, on_delete=models.CASCADE, related_name='metrics')
    name = models.CharField(max_length=200)
    target_value = models.FloatField()
    unit = models.CharField(max_length=50)
    current_value = models.FloatField(null=True)
    last_checked = models.DateTimeField(auto_now=True)

# ── 4. Feature & Data Drift Detection ─────────────────────────
class DriftMonitor(models.Model):
    """Configuration for monitoring drift on a model/run."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.OneToOneField('experiment.TrainingRun', on_delete=models.CASCADE, related_name='drift_monitor')
    feature_columns = models.JSONField(default=list, help_text="List of feature names to monitor")
    baseline_data_path = models.CharField(max_length=500, blank=True)
    check_interval_hours = models.PositiveIntegerField(default=24)
    is_active = models.BooleanField(default=True)

class DriftAlert(models.Model):
    """An alert raised when drift is detected."""
    monitor = models.ForeignKey(DriftMonitor, on_delete=models.CASCADE, related_name='alerts')
    feature_name = models.CharField(max_length=200)
    drift_score = models.FloatField()
    threshold_exceeded = models.BooleanField(default=True)
    details = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

# ── 5. Model Security & Vulnerability Scanning ────────────────
class SecurityScan(models.Model):
    """A security scan of a model artifact."""
    model_version = models.ForeignKey('model_versioning.ModelVersion', on_delete=models.CASCADE, related_name='security_scans')
    status = models.CharField(max_length=30, default='queued')
    summary = models.TextField(blank=True)
    vulnerabilities_found = models.PositiveIntegerField(default=0)
    details_json = models.JSONField(default=dict)
    scanned_at = models.DateTimeField(auto_now_add=True)

class VulnerabilityDatabase(models.Model):
    """Known CVEs or model vulnerabilities."""
    cve_id = models.CharField(max_length=30, unique=True, null=True, blank=True)
    description = models.TextField()
    severity = models.CharField(max_length=20, choices=[('low','Low'),('medium','Medium'),('high','High'),('critical','Critical')])
    affected_packages = models.JSONField(default=list)
    remediation = models.TextField(blank=True)

# ── 6. Real‑time AI Observability ─────────────────────────────
class ModelMonitor(models.Model):
    """A live monitor on a deployed model."""
    model_version = models.OneToOneField('model_versioning.ModelVersion', on_delete=models.CASCADE, related_name='monitor')
    metrics_config = models.JSONField(default=dict)  # {"accuracy": true, "hallucination": true}
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

class ObservationMetric(models.Model):
    """A single metric snapshot."""
    monitor = models.ForeignKey(ModelMonitor, on_delete=models.CASCADE, related_name='observations')
    timestamp = models.DateTimeField(auto_now_add=True)
    metric_name = models.CharField(max_length=100)
    value = models.FloatField()
    tags = models.JSONField(default=dict, blank=True)

# ── 7. Fine‑Grained Access Control (FGAC) ─────────────────────
class AccessPolicy(models.Model):
    """A policy that defines who can access what at what level."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    applies_to_model = models.CharField(max_length=100, blank=True, help_text="e.g., 'experiment.TrainingRun'")
    conditions = models.JSONField(default=dict, help_text="{'field': 'status', 'op': 'eq', 'value': 'completed'}")
    column_mask = models.JSONField(default=list, help_text="List of column names that are visible")

class AccessGrant(models.Model):
    """A grant of an access policy to a user/group."""
    policy = models.ForeignKey(AccessPolicy, on_delete=models.CASCADE, related_name='grants')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True)
    group = models.ForeignKey('auth.Group', on_delete=models.CASCADE, null=True, blank=True)
    can_read = models.BooleanField(default=True)
    can_write = models.BooleanField(default=False)
    can_delete = models.BooleanField(default=False)