"""
Model lineage – track parent-child relationships between model versions.
"""
from django.db import models
import uuid
from django.contrib.auth import get_user_model
from jobscheduler.models import ModelVersion


class ModelLineage(models.Model):
    """Track which model was derived from which."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    parent = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name='children',
        null=True, blank=True
    )
    child = models.ForeignKey(
        ModelVersion, on_delete=models.CASCADE, related_name='parents',
    )
    relationship_type = models.CharField(max_length=50, default='fine_tuned',
        choices=[('fine_tuned', 'Fine-Tuned'), ('compressed', 'Compressed'),
                  ('quantized', 'Quantized'), ('distilled', 'Distilled'),
                  ('merged', 'Merged'), ('continued', 'Continued Training')])
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['parent', 'child', 'relationship_type']]


def get_lineage_graph(model_version_id: str, direction: str = 'both', max_depth: int = 5):
    """
    Return the lineage graph for a model version.
    direction: 'ancestors', 'descendants', or 'both'
    """
    visited = set()
    nodes = []
    edges = []

    def traverse(current_id, depth=0, reverse=False):
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
            'metrics': mv.metrics,
            'created_at': mv.created_at.isoformat(),
            'depth': depth,
        })

        if reverse:
            links = ModelLineage.objects.filter(child_id=current_id).select_related('parent')
            for link in links:
                parent_id = str(link.parent_id)
                edges.append({'from': parent_id, 'to': str(current_id), 'type': link.relationship_type})
                traverse(parent_id, depth + 1, reverse=True)
        else:
            links = ModelLineage.objects.filter(parent_id=current_id).select_related('child')
            for link in links:
                child_id = str(link.child_id)
                edges.append({'from': str(current_id), 'to': child_id, 'type': link.relationship_type})
                traverse(child_id, depth + 1, reverse=False)

    if direction in ('ancestors', 'both'):
        traverse(model_version_id, reverse=True)
    if direction in ('descendants', 'both'):
        visited.clear()
        nodes.clear()
        edges.clear()
        traverse(model_version_id, reverse=False)

    return {'nodes': nodes, 'edges': edges}