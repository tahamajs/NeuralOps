import uuid

from django.conf import settings
from django.db import models


# ──────────────────────────────────────────────────────────────
# FEATURE 1: Inference Endpoints Deployment
# ──────────────────────────────────────────────────────────────
class InferenceEndpoint(models.Model):
    """A deployed model endpoint for inference."""

    PROVIDER_CHOICES = [
        ("huggingface", "HuggingFace Spaces"),
        ("aws_sagemaker", "AWS SageMaker"),
        ("gcp_vertex", "GCP Vertex AI"),
        ("azure_ml", "Azure ML"),
        ("custom", "Custom Docker"),
    ]
    STATUS_CHOICES = [
        ("building", "Building"),
        ("deploying", "Deploying"),
        ("running", "Running"),
        ("failed", "Failed"),
        ("stopped", "Stopped"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="inference_endpoints",
    )
    model_version = models.ForeignKey(
        "model_versioning.ModelVersion",
        on_delete=models.CASCADE,
        related_name="endpoints",
    )
    name = models.CharField(max_length=200)
    provider = models.CharField(
        max_length=30, choices=PROVIDER_CHOICES, default="huggingface"
    )
    endpoint_url = models.URLField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="building")
    config = models.JSONField(
        default=dict,
        blank=True,
        help_text="Provider‑specific config (instance type, scaling, etc.)",
    )
    api_key = models.CharField(
        max_length=128, blank=True, help_text="Auto‑generated API key for access"
    )
    requests_count = models.PositiveIntegerField(default=0)
    avg_latency_ms = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class EndpointLog(models.Model):
    """Inference request log."""

    endpoint = models.ForeignKey(
        InferenceEndpoint, on_delete=models.CASCADE, related_name="logs"
    )
    input_data = models.TextField(blank=True)
    output_data = models.TextField(blank=True)
    latency_ms = models.FloatField()
    status_code = models.IntegerField(default=200)
    created_at = models.DateTimeField(auto_now_add=True)


# ──────────────────────────────────────────────────────────────
# FEATURE 2: Versioned ML Pipelines (Metaflow‑style)
# ──────────────────────────────────────────────────────────────
class Pipeline(models.Model):
    """A reusable ML pipeline definition."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="pipelines"
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    steps = models.JSONField(default=list, help_text="Ordered list of step definitions")
    version = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class PipelineRun(models.Model):
    """A single execution of a pipeline."""

    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("canceled", "Canceled"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.ForeignKey(
        Pipeline, on_delete=models.CASCADE, related_name="runs"
    )
    triggered_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")
    params = models.JSONField(default=dict, blank=True)
    started_at = models.DateTimeField(null=True)
    completed_at = models.DateTimeField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)


class PipelineStepRun(models.Model):
    """Execution of a single step within a pipeline run."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("skipped", "Skipped"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        PipelineRun, on_delete=models.CASCADE, related_name="step_runs"
    )
    step_name = models.CharField(max_length=200)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    input_artifacts = models.JSONField(default=dict, blank=True)
    output_artifacts = models.JSONField(default=dict, blank=True)
    started_at = models.DateTimeField(null=True)
    completed_at = models.DateTimeField(null=True)
    logs = models.TextField(blank=True)


class Artifact(models.Model):
    """A versioned artifact produced by a pipeline."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        PipelineRun, on_delete=models.CASCADE, related_name="artifacts"
    )
    name = models.CharField(max_length=200)
    artifact_type = models.CharField(max_length=50)  # model, dataset, metrics, plot
    file = models.FileField(upload_to="pipeline_artifacts/", blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


# ──────────────────────────────────────────────────────────────
# FEATURE 3: Integrated Feature Store
# ──────────────────────────────────────────────────────────────
class FeatureGroup(models.Model):
    """A logical group of features (e.g., 'user_embeddings')."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="feature_groups",
    )
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    schema = models.JSONField(default=dict, help_text="JSON Schema for features")
    version = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)


class FeatureRecord(models.Model):
    """A single feature value for an entity."""

    group = models.ForeignKey(
        FeatureGroup, on_delete=models.CASCADE, related_name="records"
    )
    entity_id = models.CharField(max_length=200, help_text="e.g., user_id, item_id")
    values = models.JSONField(default=dict)
    ingested_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["group", "entity_id"]]


class FeatureDriftLog(models.Model):
    """Drift monitoring for a feature group."""

    group = models.ForeignKey(
        FeatureGroup, on_delete=models.CASCADE, related_name="drift_logs"
    )
    drift_score = models.FloatField()  # population stability index or similar
    details = models.JSONField(default=dict, blank=True)
    checked_at = models.DateTimeField(auto_now_add=True)


# ──────────────────────────────────────────────────────────────
# FEATURE 4: Automated Data Quality Checks
# ──────────────────────────────────────────────────────────────
class DataQualityRule(models.Model):
    """A rule for data validation (pydeequ‑style)."""

    RULE_TYPES = [
        ("completeness", "Completeness"),
        ("uniqueness", "Uniqueness"),
        ("schema", "Schema Validation"),
        ("range", "Value Range"),
        ("regex", "Regex Pattern"),
        ("custom_sql", "Custom SQL"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="dq_rules"
    )
    name = models.CharField(max_length=200)
    rule_type = models.CharField(max_length=30, choices=RULE_TYPES)
    config = models.JSONField(
        default=dict, help_text="e.g., {'column':'age','min':0,'max':120}"
    )
    is_active = models.BooleanField(default=True)


class DataQualityCheck(models.Model):
    """A single execution of a DQ rule against a dataset."""

    rule = models.ForeignKey(
        DataQualityRule, on_delete=models.CASCADE, related_name="checks"
    )
    dataset_path = models.CharField(max_length=500)
    passed = models.BooleanField()
    failure_count = models.IntegerField(default=0)
    details = models.JSONField(default=dict, blank=True)
    checked_at = models.DateTimeField(auto_now_add=True)


# ──────────────────────────────────────────────────────────────
# FEATURE 5: Continuous Training & Deployment (CT/CD)
# ──────────────────────────────────────────────────────────────
class CTCDPipeline(models.Model):
    """A continuous training + deployment pipeline."""

    TRIGGER_CHOICES = [
        ("schedule", "Schedule (cron)"),
        ("new_data", "New Data Arrival"),
        ("code_push", "Code Push (Git)"),
        ("manual", "Manual"),
        ("model_drift", "Model Drift Detected"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="ctcd_pipelines",
    )
    name = models.CharField(max_length=200)
    trigger_type = models.CharField(
        max_length=20, choices=TRIGGER_CHOICES, default="manual"
    )
    cron_schedule = models.CharField(max_length=100, blank=True)
    git_repo = models.URLField(blank=True)
    git_branch = models.CharField(max_length=100, default="main")
    training_config = models.JSONField(default=dict)
    deployment_config = models.JSONField(default=dict)
    is_active = models.BooleanField(default=True)
    last_run = models.ForeignKey(
        "CTCDRun", on_delete=models.SET_NULL, null=True, blank=True, related_name="+"
    )
    created_at = models.DateTimeField(auto_now_add=True)


class CTCDRun(models.Model):
    """A single execution of a CT/CD pipeline."""

    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("training", "Training"),
        ("evaluating", "Evaluating"),
        ("deploying", "Deploying"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.ForeignKey(
        CTCDPipeline, on_delete=models.CASCADE, related_name="runs"
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")
    training_run = models.ForeignKey(
        "jobscheduler.Job", on_delete=models.SET_NULL, null=True, blank=True
    )
    new_model_version = models.ForeignKey(
        "model_versioning.ModelVersion",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    metrics = models.JSONField(default=dict, blank=True)
    started_at = models.DateTimeField(null=True)
    completed_at = models.DateTimeField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)


# ──────────────────────────────────────────────────────────────
# FEATURE 6: LLM Quality & Safety Monitors
# ──────────────────────────────────────────────────────────────
class LLMMonitor(models.Model):
    """A quality/safety monitor for an inference endpoint."""

    METRIC_CHOICES = [
        ("rouge", "ROUGE"),
        ("bert_score", "BERTScore"),
        ("hallucination", "Hallucination Score"),
        ("toxicity", "Toxicity"),
        ("bias", "Bias Detection"),
        ("relevance", "Relevance"),
        ("latency", "Latency P95"),
        ("cost_per_token", "Cost/Token"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    endpoint = models.ForeignKey(
        InferenceEndpoint, on_delete=models.CASCADE, related_name="monitors"
    )
    metric = models.CharField(max_length=30, choices=METRIC_CHOICES)
    threshold_warning = models.FloatField(null=True, blank=True)
    threshold_critical = models.FloatField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)


class LLMMonitorAlert(models.Model):
    """An alert triggered by a monitor."""

    monitor = models.ForeignKey(
        LLMMonitor, on_delete=models.CASCADE, related_name="alerts"
    )
    value = models.FloatField()
    severity = models.CharField(
        max_length=20, choices=[("warning", "Warning"), ("critical", "Critical")]
    )
    message = models.TextField()
    resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)


# ──────────────────────────────────────────────────────────────
# FEATURE 7: Model Distillation & Quantization Tools
# ──────────────────────────────────────────────────────────────
class DistillationJob(models.Model):
    """A knowledge distillation run."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="distillation_jobs",
    )
    teacher_model = models.ForeignKey(
        "model_versioning.ModelVersion",
        on_delete=models.CASCADE,
        related_name="distill_teacher",
    )
    student_model_config = models.JSONField(default=dict)
    compression_config = models.JSONField(
        default=dict, help_text="Quantization, pruning settings"
    )
    status = models.CharField(max_length=20, default="queued")
    output_model = models.ForeignKey(
        "model_versioning.ModelVersion",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="distill_output",
    )
    metrics = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


# ──────────────────────────────────────────────────────────────
# FEATURE 8: One‑Click Multi‑Node Fine‑Tuning
# ──────────────────────────────────────────────────────────────
class MultiNodeTraining(models.Model):
    """A distributed fine‑tuning job across multiple GPUs."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="multi_node_trainings",
    )
    federation_cluster = models.ForeignKey(
        "gpu_federation.FederationCluster",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    base_model = models.CharField(max_length=200)
    training_config = models.JSONField(default=dict)
    strategy = models.CharField(
        max_length=30,
        default="fsdp",
        choices=[("ddp", "DDP"), ("fsdp", "FSDP"), ("deepspeed", "DeepSpeed")],
    )
    status = models.CharField(max_length=20, default="queued")
    output_model = models.ForeignKey(
        "model_versioning.ModelVersion",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="multi_node_output",
    )
    created_at = models.DateTimeField(auto_now_add=True)
