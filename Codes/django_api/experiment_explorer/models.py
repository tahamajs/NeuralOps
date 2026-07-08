import uuid

from django.conf import settings
from django.db import models


class ExperimentDataset(models.Model):
    """
    A dataset of experiment results uploaded by a user.
    Stored as JSON – each row is a dict of hyperparameters + metrics.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="experiment_datasets",
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    data = models.JSONField(
        default=list, help_text="List of dicts – each dict is one experiment run"
    )
    hyperparameter_columns = models.JSONField(default=list)
    metric_columns = models.JSONField(default=list)
    row_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if self.data:
            self.row_count = len(self.data)
            # Auto-detect columns from the first row
            if (
                self.data
                and not self.hyperparameter_columns
                and not self.metric_columns
            ):
                first = self.data[0]
                all_keys = list(first.keys())
                self.hyperparameter_columns = [
                    k
                    for k in all_keys
                    if k.startswith("hp_")
                    or k in ("learning_rate", "batch_size", "epochs", "rank", "dropout")
                ]
                self.metric_columns = [
                    k for k in all_keys if k not in self.hyperparameter_columns
                ]
        super().save(*args, **kwargs)


class SavedVisualization(models.Model):
    """
    A saved visualization configuration (parallel coordinates, radar, heatmap).
    """

    VIZ_TYPES = [
        ("parallel_coordinates", "Parallel Coordinates"),
        ("radar", "Radar Chart"),
        ("heatmap", "Correlation Heatmap"),
        ("scatter_matrix", "Scatter Matrix"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="saved_visualizations",
    )
    dataset = models.ForeignKey(
        ExperimentDataset, on_delete=models.CASCADE, related_name="visualizations"
    )
    name = models.CharField(max_length=200)
    viz_type = models.CharField(max_length=30, choices=VIZ_TYPES)
    config = models.JSONField(
        default=dict, help_text="Selected columns, filters, color mappings"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.get_viz_type_display()})"
