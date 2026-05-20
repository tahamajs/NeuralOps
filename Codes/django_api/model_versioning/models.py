import uuid
import os
import shutil
from django.db import models
from django.conf import settings
from jobscheduler.models import Job 

class ModelVersion(models.Model):
    STORAGE_TIERS = [
        ('hot', 'Hot (Local)'),
        ('warm', 'Warm (Cloud)'),
        ('cold', 'Cold (Archive)'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='model_versions')
    version = models.CharField(max_length=100)
    version_number = models.IntegerField()
    artifact_path = models.CharField(max_length=500)
    compressed_artifact_path = models.CharField(max_length=500, blank=True)
    compression_method = models.CharField(max_length=100, blank=True)
    compression_config = models.JSONField(default=dict, blank=True)
    artifact_size_mb = models.FloatField(default=0.0)
    compressed_size_mb = models.FloatField(default=0.0)
    storage_tier = models.CharField(max_length=20, choices=STORAGE_TIERS, default='hot')
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
        ordering = ['-version_number']
        unique_together = [['run', 'version_number']]
        indexes = [
            models.Index(fields=['run', '-version_number']),
            models.Index(fields=['is_published_to_hub']),
            models.Index(fields=['storage_tier']),
        ]

    def __str__(self):
        return f"v{self.version_number} ({self.version})"

    def delete_local_artifact(self):
        if os.path.exists(self.artifact_path):
            shutil.rmtree(self.artifact_path)
        if self.compressed_artifact_path and os.path.exists(self.compressed_artifact_path):
            os.remove(self.compressed_artifact_path)


class ModelVersionDiff(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    from_version = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name='diffs_from'
    )
    to_version = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name='diffs_to'
    )
    diff_artifact = models.CharField(max_length=500)
    diff_size_mb = models.FloatField(default=0.0)
    original_size_saved_mb = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['from_version', 'to_version']]

    def __str__(self):
        return f"Diff v{self.from_version.version_number}→v{self.to_version.version_number}"


class ModelLineage(models.Model):
    """Tracks parent‑child relationships between model versions."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    parent = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name='children',
        null=True, blank=True
    )
    child = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name='parents',
    )
    relationship_type = models.CharField(
        max_length=50, default='fine_tuned',
        choices=[
            ('fine_tuned', 'Fine-Tuned'),
            ('compressed', 'Compressed'),
            ('quantized', 'Quantized'),
            ('distilled', 'Distilled'),
            ('merged', 'Merged'),
            ('continued', 'Continued Training'),
        ]
    )
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['parent', 'child', 'relationship_type']]

    def __str__(self):
        return f"v{self.parent.version_number} → v{self.child.version_number} ({self.relationship_type})"


def get_lineage_graph(model_version_id: str, direction: str = 'both', max_depth: int = 5):
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
        nodes.append({
            'id': str(mv.id),
            'version': mv.version,
            'version_number': mv.version_number,
            'metrics': mv.metrics,
            'created_at': mv.created_at.isoformat(),
            'depth': depth,
        })
        links = ModelLineage.objects.filter(child_id=current_id).select_related('parent')
        for link in links:
            parent_id = str(link.parent_id)
            edges.append({
                'from': parent_id,
                'to': str(current_id),
                'type': link.relationship_type,
            })
            traverse_ancestors(parent_id, depth + 1)

    def traverse_descendants(current_id, depth=0):
        if depth > max_depth or current_id in visited:
            return
        visited.add(current_id)
        try:
            mv = ModelVersion.objects.get(id=current_id)
        except ModelVersion.DoesNotExist:
            return
        nodes.append({
            'id': str(mv.id),
            'version': mv.version,
            'version_number': mv.version_number,
            'metrics': mv.metrics,
            'created_at': mv.created_at.isoformat(),
            'depth': depth,
        })
        links = ModelLineage.objects.filter(parent_id=current_id).select_related('child')
        for link in links:
            child_id = str(link.child_id)
            edges.append({
                'from': str(current_id),
                'to': child_id,
                'type': link.relationship_type,
            })
            traverse_descendants(child_id, depth + 1)

    if direction in ('ancestors', 'both'):
        visited.clear()
        traverse_ancestors(model_version_id)

    if direction in ('descendants', 'both'):
        visited.clear()
        traverse_descendants(model_version_id)

    return {'nodes': nodes, 'edges': edges}