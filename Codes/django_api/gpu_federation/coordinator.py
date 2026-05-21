"""
Centralized coordinator for federation tasks (Ray-style).

Handles:
- Assigning shards to nodes
- Monitoring heartbeats
- Handling node failures (reassigning shards)
- Aggregating progress
"""
import logging, time
from django.utils import timezone
from .models import FederationTask, FederationTaskShard, FederationNode

logger = logging.getLogger(__name__)


class FederationCoordinator:
    """Orchestrates a single federation task."""

    def __init__(self, task: FederationTask):
        self.task = task
        self.cluster = task.cluster

    def initialize(self):
        """Create shard assignments and mark task as initializing."""
        self.task.status = 'initializing'
        self.task.save(update_fields=['status'])

        # Get active nodes
        active_nodes = self.cluster.active_nodes.order_by('shard_start')
        if active_nodes.count() < self.cluster.min_nodes:
            self.task.status = 'failed'
            self.task.save(update_fields=['status'])
            logger.error("Not enough active nodes for task %s", self.task.id)
            return

        # Compute shard assignments
        from .sharding import compute_shard_assignment
        num_layers = self.cluster.model_config.get('num_layers', 12)
        strategy = self.cluster.sharding_strategy

        assignments = compute_shard_assignment(num_layers, active_nodes, strategy)

        for node, start, end in assignments:
            # Update node shard assignment
            node.shard_start = start
            node.shard_end = end
            node.save(update_fields=['shard_start', 'shard_end'])

            # Create task shard record
            FederationTaskShard.objects.create(
                task=self.task,
                node=node,
                status='pending'
            )

        self.task.total_steps = num_layers * self.task.epochs * 1000  # example
        self.task.status = 'running'
        self.task.started_at = timezone.now()
        self.task.save(update_fields=['status', 'total_steps', 'started_at'])

    def update_progress(self, node_id: int, steps: int, loss: float, samples: int):
        """Update progress for a node's shard."""
        try:
            shard = FederationTaskShard.objects.get(task=self.task, node_id=node_id)
            shard.steps_completed += steps
            shard.samples_processed += samples
            shard.avg_loss = loss
            shard.save(update_fields=['steps_completed', 'samples_processed', 'avg_loss'])

            # Aggregate task progress
            all_shards = self.task.shards.all()
            total_steps_done = sum(s.steps_completed for s in all_shards)
            self.task.current_step = total_steps_done
            self.task.progress_percent = min(
                100.0,
                (total_steps_done / max(self.task.total_steps, 1)) * 100
            )
            self.task.total_loss = loss  # simplified
            self.task.save(update_fields=['current_step', 'progress_percent', 'total_loss'])

        except FederationTaskShard.DoesNotExist:
            logger.warning("Shard not found for node %s in task %s", node_id, self.task.id)

    def handle_node_failure(self, node_id: int):
        """Reassign a failed node's shard to another active node."""
        # Simplified: just mark the shard as failed and the node as disconnected
        try:
            node = FederationNode.objects.get(id=node_id)
            node.status = 'failed'
            node.save(update_fields=['status'])

            shard = FederationTaskShard.objects.get(task=self.task, node=node)
            shard.status = 'failed'
            shard.save(update_fields=['status'])

            # Try to find another active node to take over
            spare = self.cluster.active_nodes.exclude(id=node_id).first()
            if spare:
                spare.shard_start = node.shard_start
                spare.shard_end = node.shard_end
                spare.save(update_fields=['shard_start', 'shard_end'])
                FederationTaskShard.objects.create(
                    task=self.task, node=spare, status='pending'
                )
        except FederationNode.DoesNotExist:
            pass

    def complete(self):
        """Mark task as completed and aggregate final results."""
        self.task.status = 'completed'
        self.task.completed_at = timezone.now()
        self.task.save(update_fields=['status', 'completed_at'])

        from .models import FederationResult
        FederationResult.objects.create(
            task=self.task,
            final_loss=self.task.total_loss,
            total_time_seconds=(
                (self.task.completed_at - self.task.started_at).total_seconds()
                if self.task.started_at else 0
            ),
        )