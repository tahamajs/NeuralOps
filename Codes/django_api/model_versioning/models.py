import os
import shutil
import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone
from jobscheduler.models import Job


class ModelVersion(models.Model):
    STORAGE_TIERS = [
        ("hot", "Hot (Local)"),
        ("warm", "Warm (Cloud)"),
        ("cold", "Cold (Archive)"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(
        Job, on_delete=models.CASCADE, related_name="versioning_model_versions"
    )
    version = models.CharField(max_length=100)
    version_number = models.IntegerField()
    artifact_path = models.CharField(max_length=500)
    compressed_artifact_path = models.CharField(max_length=500, blank=True)
    compression_method = models.CharField(max_length=100, blank=True)
    compression_config = models.JSONField(default=dict, blank=True)
    artifact_size_mb = models.FloatField(default=0.0)
    compressed_size_mb = models.FloatField(default=0.0)
    storage_tier = models.CharField(max_length=20, choices=STORAGE_TIERS, default="hot")
    is_published_to_hub = models.BooleanField(default=False)
    hub_repo_url = models.URLField(max_length=500, blank=True, null=True)
    metrics = models.JSONField(default=dict)
    git_commit = models.CharField(max_length=40, null=True, blank=True)
    config_snapshot_id = models.UUIDField(null=True, blank=True)
    validation_passed = models.BooleanField(default=False)
    is_production = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-version_number"]
        unique_together = [["run", "version_number"]]
        indexes = [
            models.Index(fields=["run", "-version_number"]),
            models.Index(fields=["is_published_to_hub"]),
            models.Index(fields=["storage_tier"]),
        ]

    def __str__(self):
        return f"v{self.version_number} ({self.version})"

    def delete_local_artifact(self):
        if os.path.exists(self.artifact_path):
            shutil.rmtree(self.artifact_path)
        if self.compressed_artifact_path and os.path.exists(
            self.compressed_artifact_path
        ):
            os.remove(self.compressed_artifact_path)

    def archive_to_cold_storage(self, bucket_name: str):
        """
        Move the artifact (and compressed artifact) to an S3 bucket,
        update storage tier to 'cold', and remove local copies.
        """
        import boto3

        s3 = boto3.client("s3")
        # Upload original artifact directory as a tar.gz
        import io
        import os
        import tarfile

        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
            tar.add(self.artifact_path, arcname=os.path.basename(self.artifact_path))
        tar_buffer.seek(0)
        key = f"run_{self.run_id}/v{self.version_number}/artifact.tar.gz"
        s3.upload_fileobj(tar_buffer, bucket_name, key)

        # If compressed artifact exists and is separate, also upload it
        if self.compressed_artifact_path and os.path.exists(
            self.compressed_artifact_path
        ):
            comp_key = f"run_{self.run_id}/v{self.version_number}/{os.path.basename(self.compressed_artifact_path)}"
            with open(self.compressed_artifact_path, "rb") as f:
                s3.upload_fileobj(f, bucket_name, comp_key)

        # Delete local artifacts
        self.delete_local_artifact()
        self.storage_tier = "cold"
        self.save(update_fields=["storage_tier"])

    def restore_from_cold_storage(self, bucket_name: str):
        """
        Download the artifact from cold storage (S3) back to local disk,
        set storage tier to 'hot'.
        """
        import os

        import boto3

        s3 = boto3.client("s3")
        # Restore the original artifact
        artifact_key = f"run_{self.run_id}/v{self.version_number}/artifact.tar.gz"
        local_tar = os.path.join(self.artifact_path, "artifact.tar.gz")
        os.makedirs(self.artifact_path, exist_ok=True)
        s3.download_file(bucket_name, artifact_key, local_tar)

        # Extract
        import tarfile

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(self.artifact_path)
        os.remove(local_tar)

        # Restore compressed artifact if it existed
        if self.compressed_artifact_path:
            comp_filename = os.path.basename(self.compressed_artifact_path)
            comp_key = f"run_{self.run_id}/v{self.version_number}/{comp_filename}"
            try:
                s3.download_file(bucket_name, comp_key, self.compressed_artifact_path)
            except Exception:
                pass  # compressed artifact may be missing, ignore

        self.storage_tier = "hot"
        self.save(update_fields=["storage_tier"])


class ModelVersionDiff(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    from_version = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name="diffs_from"
    )
    to_version = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name="diffs_to"
    )
    diff_artifact = models.CharField(max_length=500)
    diff_size_mb = models.FloatField(default=0.0)
    original_size_saved_mb = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["from_version", "to_version"]]

    def __str__(self):
        return f"Diff v{self.from_version.version_number}→v{self.to_version.version_number}"


class ModelLineage(models.Model):
    """Tracks parent‑child relationships between model versions."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    parent = models.ForeignKey(
        ModelVersion,
        on_delete=models.CASCADE,
        related_name="children",
        null=True,
        blank=True,
    )
    child = models.ForeignKey(
        ModelVersion,
        on_delete=models.CASCADE,
        related_name="parents",
    )
    relationship_type = models.CharField(
        max_length=50,
        default="fine_tuned",
        choices=[
            ("fine_tuned", "Fine-Tuned"),
            ("compressed", "Compressed"),
            ("quantized", "Quantized"),
            ("distilled", "Distilled"),
            ("merged", "Merged"),
            ("continued", "Continued Training"),
        ],
    )
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["parent", "child", "relationship_type"]]

    def __str__(self):
        return f"v{self.parent.version_number} → v{self.child.version_number} ({self.relationship_type})"


def get_lineage_graph(
    model_version_id: str, direction: str = "both", max_depth: int = 5
):
    """
    Return a clean lineage graph for a model version.
    direction: 'ancestors', 'descendants', or 'both'
    """
    visited = set()
    nodes = []
    edges = []

    def traverse_ancestors(current_id, depth=0):
        if depth > max_depth or current_id in visited:
            return
        visited.add(current_id)
        try:
            mv = ModelVersion.objects.get(id=current_id)
        except ModelVersion.DoesNotExist:
            return
        nodes.append(
            {
                "id": str(mv.id),
                "version": mv.version,
                "version_number": mv.version_number,
                "metrics": mv.metrics,
                "created_at": mv.created_at.isoformat(),
                "depth": depth,
            }
        )
        links = ModelLineage.objects.filter(child_id=current_id).select_related(
            "parent"
        )
        for link in links:
            parent_id = str(link.parent_id)
            edges.append(
                {
                    "from": parent_id,
                    "to": str(current_id),
                    "type": link.relationship_type,
                }
            )
            traverse_ancestors(parent_id, depth + 1)

    def traverse_descendants(current_id, depth=0):
        if depth > max_depth or current_id in visited:
            return
        visited.add(current_id)
        try:
            mv = ModelVersion.objects.get(id=current_id)
        except ModelVersion.DoesNotExist:
            return
        nodes.append(
            {
                "id": str(mv.id),
                "version": mv.version,
                "version_number": mv.version_number,
                "metrics": mv.metrics,
                "created_at": mv.created_at.isoformat(),
                "depth": depth,
            }
        )
        links = ModelLineage.objects.filter(parent_id=current_id).select_related(
            "child"
        )
        for link in links:
            child_id = str(link.child_id)
            edges.append(
                {
                    "from": str(current_id),
                    "to": child_id,
                    "type": link.relationship_type,
                }
            )
            traverse_descendants(child_id, depth + 1)

    if direction in ("ancestors", "both"):
        visited.clear()
        traverse_ancestors(model_version_id)

    if direction in ("descendants", "both"):
        visited.clear()
        traverse_descendants(model_version_id)

    return {"nodes": nodes, "edges": edges}


# model_versioning/models.py — add these classes after the existing ModelLineage class


class ModelApproval(models.Model):
    """
    Governance approval workflow for a model version.
    pending → approved → rejected (or back to pending for re‑review)
    """

    STATUS_CHOICES = [
        ("pending", "Pending Review"),
        ("in_review", "In Review"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
        ("changes_requested", "Changes Requested"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.OneToOneField(
        ModelVersion, on_delete=models.CASCADE, related_name="approval"
    )
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default="pending")
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="versioning_approval_requests",
    )
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="approval_reviews",
    )
    notes = models.TextField(blank=True)
    review_notes = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Approval for {self.version}: {self.status}"

    def approve(self, reviewer, notes=""):
        self.status = "approved"
        self.reviewer = reviewer
        self.review_notes = notes
        self.save()
        self.version.is_published_to_hub = True
        self.version.save(update_fields=["is_published_to_hub"])

    def reject(self, reviewer, notes=""):
        self.status = "rejected"
        self.reviewer = reviewer
        self.review_notes = notes
        self.save()


class ModelAccessControl(models.Model):
    """
    Fine‑grained access control for a model version.
    """

    ACCESS_LEVELS = [
        ("read", "Read"),
        ("download", "Download"),
        ("deploy", "Deploy"),
        ("manage", "Manage"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name="access_controls"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="versioning_model_access"
    )
    access_level = models.CharField(
        max_length=20, choices=ACCESS_LEVELS, default="read"
    )
    granted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="versioning_granted_access",
    )
    granted_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [["version", "user"]]
        ordering = ["-granted_at"]

    def __str__(self):
        return f"{self.user.username} → {self.access_level} on {self.version}"

    @property
    def is_expired(self):
        if self.expires_at and timezone.now() > self.expires_at:
            return True
        return False


class ModelSecurityScan(models.Model):
    """
    Security scan result for a model version.
    Inspired by JFrog's AI Catalogue security scanning.
    """

    SEVERITY_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("critical", "Critical"),
    ]
    SCAN_TYPES = [
        ("vulnerability", "Vulnerability Scan"),
        ("license", "License Compliance"),
        ("malware", "Malware Scan"),
        ("secrets", "Secrets Detection"),
        ("dependency", "Dependency Scan"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name="security_scans"
    )
    scan_type = models.CharField(
        max_length=30, choices=SCAN_TYPES, default="vulnerability"
    )
    status = models.CharField(
        max_length=30,
        default="pending",
        choices=[
            ("pending", "Pending"),
            ("running", "Running"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
    )
    severity = models.CharField(
        max_length=20, choices=SEVERITY_CHOICES, null=True, blank=True
    )
    findings = models.JSONField(default=dict, blank=True)
    passed = models.BooleanField(default=False)
    report_url = models.URLField(blank=True, max_length=500)
    scanned_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-scanned_at"]

    def __str__(self):
        return f"Scan {self.scan_type} for {self.version}: {self.status}"


class ModelUsageLog(models.Model):
    """
    Audit trail for model usage — downloads, deployments, API calls.
    """

    ACTION_TYPES = [
        ("downloaded", "Downloaded"),
        ("deployed", "Deployed"),
        ("inference", "Inference API Call"),
        ("viewed", "Viewed"),
        ("shared", "Shared"),
        ("exported", "Exported"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    version = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name="usage_logs"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="versioning_model_usage"
    )
    action = models.CharField(max_length=20, choices=ACTION_TYPES)
    metadata = models.JSONField(default=dict, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["version", "-created_at"]),
            models.Index(fields=["user", "-created_at"]),
        ]

    def __str__(self):
        return f"{self.user.username} {self.action} {self.version}"


class ModelBranch(models.Model):
    """A named branch for model versions (like git branch)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=150)
    root_version = models.ForeignKey(
        "ModelVersion", on_delete=models.CASCADE, related_name="branches"
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["name", "root_version"]]

    def __str__(self):
        return f"{self.name} (from {self.root_version})"


class ModelCommit(models.Model):
    """A snapshot commit in a branch, linking versions with parent/merge info."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    branch = models.ForeignKey(
        ModelBranch, on_delete=models.CASCADE, related_name="commits"
    )
    version = models.ForeignKey(
        "ModelVersion", on_delete=models.CASCADE, related_name="commits"
    )
    parent = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="children",
    )
    merge_parent = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="merge_children",
        help_text="Second parent for merge commits",
    )
    message = models.TextField(blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
