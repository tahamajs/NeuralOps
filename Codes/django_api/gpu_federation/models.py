import uuid

from django.conf import settings
from django.db import models


class FederationCluster(models.Model):
    """A pool of GPU nodes assembled for distributed training."""

    STATUS_CHOICES = [
        ("forming", "Forming"),
        ("active", "Active"),
        ("degraded", "Degraded"),
        ("inactive", "Inactive"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="federation_clusters",
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)

    # Configuration
    model_name = models.CharField(
        max_length=200, help_text="e.g., 'bert-base-uncased', 'llama-2-7b'"
    )
    model_config = models.JSONField(
        default=dict, blank=True, help_text="e.g., {'hidden_size':768, 'num_layers':12}"
    )
    sharding_strategy = models.CharField(
        max_length=30,
        default="layer_wise",
        choices=[
            ("layer_wise", "Layer‑Wise"),
            ("tensor_parallel", "Tensor Parallel"),
            ("data_parallel", "Data Parallel"),
            ("hybrid", "Hybrid"),
        ],
    )
    min_nodes = models.PositiveIntegerField(default=2)
    max_nodes = models.PositiveIntegerField(default=50)
    target_nodes = models.PositiveIntegerField(default=4)

    # Status & stats
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="forming")
    total_flops = models.FloatField(
        default=0, help_text="Aggregate TFLOPS of all nodes"
    )
    total_vram_gb = models.FloatField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.status})"

    def recalculate_resources(self):
        """Aggregate total TFLOPS and VRAM from active nodes."""
        active = self.nodes.filter(status="active")
        self.total_flops = sum(
            (n.listing.benchmark_score.composite_score or 0) * n.listing.gpu_count
            for n in active
        )
        self.total_vram_gb = sum(
            n.listing.vram_gb * n.listing.gpu_count for n in active
        )
        self.save(update_fields=["total_flops", "total_vram_gb"])

    @property
    def active_nodes(self):
        return self.nodes.filter(status="active")

    @property
    def node_count(self):
        return self.nodes.count()


class FederationNode(models.Model):
    """A GPU listing that has joined a cluster."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("active", "Active"),
        ("disconnected", "Disconnected"),
        ("failed", "Failed"),
        ("removed", "Removed"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    cluster = models.ForeignKey(
        FederationCluster, on_delete=models.CASCADE, related_name="nodes"
    )
    listing = models.ForeignKey(
        "gpu_marketplace.GPUListing",
        on_delete=models.CASCADE,
        related_name="federation_nodes",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    # Assigned shard (e.g., layer range)
    shard_start = models.IntegerField(
        null=True, blank=True, help_text="First layer index"
    )
    shard_end = models.IntegerField(
        null=True, blank=True, help_text="Last layer index (exclusive)"
    )

    joined_at = models.DateTimeField(auto_now_add=True)
    last_heartbeat = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [["cluster", "listing"]]
        ordering = ["shard_start"]

    def __str__(self):
        return f"Node {self.listing.gpu_model} in {self.cluster.name} ({self.status})"


class FederationTask(models.Model):
    """A distributed training task running on a cluster."""

    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("initializing", "Initializing"),
        ("running", "Running"),
        ("paused", "Paused"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("canceled", "Canceled"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    cluster = models.ForeignKey(
        FederationCluster, on_delete=models.CASCADE, related_name="tasks"
    )
    training_run = models.ForeignKey(
        "jobscheduler.Job",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="federation_tasks",
    )
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="federation_tasks",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")

    # Training config
    batch_size_per_node = models.PositiveIntegerField(default=8)
    micro_batch_size = models.PositiveIntegerField(default=1)
    epochs = models.PositiveIntegerField(default=1)
    learning_rate = models.FloatField(default=2e-5)
    optimizer = models.CharField(max_length=50, default="adamw")

    # Progress
    current_step = models.PositiveIntegerField(default=0)
    total_steps = models.PositiveIntegerField(default=0)
    progress_percent = models.FloatField(default=0.0)

    # Metrics
    total_loss = models.FloatField(null=True, blank=True)
    total_throughput = models.FloatField(
        null=True, blank=True, help_text="Samples/sec across cluster"
    )

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Task {self.id.hex[:8]} on {self.cluster.name} ({self.status})"


class FederationTaskShard(models.Model):
    """A shard of a federation task assigned to a specific node."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.ForeignKey(
        FederationTask, on_delete=models.CASCADE, related_name="shards"
    )
    node = models.ForeignKey(
        FederationNode, on_delete=models.CASCADE, related_name="task_shards"
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    # Progress
    steps_completed = models.PositiveIntegerField(default=0)
    samples_processed = models.PositiveIntegerField(default=0)
    avg_loss = models.FloatField(null=True, blank=True)

    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = [["task", "node"]]

    def __str__(self):
        return f"Shard on {self.node.listing.gpu_model} (Task {self.task_id.hex[:8]})"


class FederationResult(models.Model):
    """Aggregated results from a completed federation task."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.OneToOneField(
        FederationTask, on_delete=models.CASCADE, related_name="result"
    )
    final_loss = models.FloatField(null=True, blank=True)
    final_accuracy = models.FloatField(null=True, blank=True)
    total_time_seconds = models.FloatField(null=True, blank=True)
    total_cost_usd = models.DecimalField(
        max_digits=12, decimal_places=2, null=True, blank=True
    )
    metrics = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Results for Task {self.task_id.hex[:8]}"
