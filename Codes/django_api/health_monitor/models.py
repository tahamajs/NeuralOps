import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator


class FeatureStore(models.Model):
    """
    A registered feature store – could be Feast, Tecton, Hopsworks, or a simple DB table.
    """
    STORE_TYPES = [
        ('feast', 'Feast'),
        ('postgres', 'PostgreSQL'),
        ('redis', 'Redis'),
        ('s3_parquet', 'S3 Parquet'),
        ('custom', 'Custom'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    store_type = models.CharField(max_length=30, choices=STORE_TYPES, default='postgres')
    connection_config = models.JSONField(default=dict, help_text="DB credentials, S3 bucket, etc.")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.store_type})"


class FeatureGroup(models.Model):
    """
    A group of related features within a feature store.
    e.g., 'user_embeddings', 'product_metadata', 'training_dataset_v2'.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    feature_store = models.ForeignKey(FeatureStore, on_delete=models.CASCADE, related_name='feature_groups')
    name = models.CharField(max_length=200)
    version = models.CharField(max_length=50, default='v1')
    schema_definition = models.JSONField(default=dict, help_text="Feature names, types, descriptions")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['feature_store', 'name', 'version']]
        ordering = ['feature_store', 'name']

    def __str__(self):
        return f"{self.feature_store.name}/{self.name}:{self.version}"


class DriftMonitor(models.Model):
    """
    A configured drift monitor for a specific feature group or model.
    """
    DRIFT_TYPES = [
        ('data_drift', 'Data Drift'),
        ('concept_drift', 'Concept Drift'),
        ('prediction_drift', 'Prediction Drift'),
        ('feature_skew', 'Feature Skew'),
    ]
    METRICS = [
        ('ks_test', 'Kolmogorov-Smirnov Test'),
        ('wasserstein', 'Wasserstein Distance'),
        ('psi', 'Population Stability Index'),
        ('js_divergence', 'Jensen-Shannon Divergence'),
        ('chi2', 'Chi-Squared Test'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    feature_group = models.ForeignKey(
        FeatureGroup, on_delete=models.CASCADE, related_name='drift_monitors'
    )
    training_run = models.ForeignKey(
        'experiment.TrainingRun', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='drift_monitors'
    )
    drift_type = models.CharField(max_length=30, choices=DRIFT_TYPES, default='data_drift')
    metric = models.CharField(max_length=30, choices=METRICS, default='psi')

    # Thresholds
    warning_threshold = models.FloatField(default=0.1)
    critical_threshold = models.FloatField(default=0.2)

    # Schedule
    check_interval_minutes = models.PositiveIntegerField(default=60)
    is_active = models.BooleanField(default=True)

    # Stats
    last_checked_at = models.DateTimeField(null=True, blank=True)
    last_status = models.CharField(max_length=30, default='unknown',
                                    choices=[('healthy','Healthy'),('warning','Warning'),
                                             ('critical','Critical'),('unknown','Unknown')])

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.drift_type})"


class DriftCheck(models.Model):
    """
    A single drift check result.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    monitor = models.ForeignKey(DriftMonitor, on_delete=models.CASCADE, related_name='checks')
    metric_value = models.FloatField()
    threshold_value = models.FloatField()
    status = models.CharField(max_length=20, choices=[('healthy','Healthy'),('warning','Warning'),('critical','Critical')])
    feature_name = models.CharField(max_length=200, blank=True)
    details = models.JSONField(default=dict)
    checked_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-checked_at']
        indexes = [
            models.Index(fields=['monitor', '-checked_at']),
        ]

    def __str__(self):
        return f"{self.monitor.name} – {self.status} ({self.metric_value:.4f})"


class Alert(models.Model):
    """
    A triggered alert when a threshold is breached.
    """
    SEVERITY_CHOICES = [
        ('warning', 'Warning'),
        ('critical', 'Critical'),
    ]
    CHANNEL_CHOICES = [
        ('email', 'Email'),
        ('slack', 'Slack'),
        ('webhook', 'Webhook'),
        ('in_app', 'In-App Notification'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    monitor = models.ForeignKey(DriftMonitor, on_delete=models.CASCADE, related_name='alerts')
    drift_check = models.ForeignKey(DriftCheck, on_delete=models.CASCADE, null=True, blank=True)
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    channel = models.CharField(max_length=20, choices=CHANNEL_CHOICES, default='in_app')
    title = models.CharField(max_length=300)
    message = models.TextField()
    is_acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='acknowledged_alerts'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.severity}] {self.title}"


class AlertChannel(models.Model):
    """
    A configured notification channel for alerts.
    """
    CHANNEL_TYPES = [
        ('email', 'Email'),
        ('slack', 'Slack Webhook'),
        ('discord', 'Discord Webhook'),
        ('webhook', 'Generic Webhook'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='alert_channels')
    channel_type = models.CharField(max_length=20, choices=CHANNEL_TYPES)
    destination = models.CharField(max_length=500, help_text="Email address, Slack webhook URL, etc.")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['channel_type']

    def __str__(self):
        return f"{self.channel_type}: {self.destination}"


class ModelPerformanceSnapshot(models.Model):
    """
    A snapshot of a model's performance metrics at a point in time.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    training_run = models.ForeignKey(
        'experiment.TrainingRun', on_delete=models.CASCADE, related_name='performance_snapshots'
    )
    metrics = models.JSONField(default=dict)
    dataset_name = models.CharField(max_length=200, blank=True)
    sample_count = models.PositiveIntegerField(default=0)
    recorded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-recorded_at']

    def __str__(self):
        return f"Performance for {self.training_run_id} at {self.recorded_at}"