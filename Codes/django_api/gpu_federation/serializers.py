from rest_framework import serializers

from .models import (
    FederationCluster,
    FederationNode,
    FederationResult,
    FederationTask,
    FederationTaskShard,
)


class FederationNodeSerializer(serializers.ModelSerializer):
    listing_gpu_model = serializers.CharField(
        source="listing.gpu_model", read_only=True
    )
    provider_username = serializers.CharField(
        source="listing.provider.username", read_only=True
    )
    vram_gb = serializers.IntegerField(source="listing.vram_gb", read_only=True)

    class Meta:
        model = FederationNode
        fields = [
            "id",
            "cluster",
            "listing",
            "listing_gpu_model",
            "provider_username",
            "vram_gb",
            "status",
            "shard_start",
            "shard_end",
            "joined_at",
            "last_heartbeat",
        ]
        read_only_fields = ["id", "joined_at", "last_heartbeat"]


class FederationClusterListSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    node_count = serializers.SerializerMethodField()
    active_nodes = serializers.SerializerMethodField()

    class Meta:
        model = FederationCluster
        fields = [
            "id",
            "name",
            "owner",
            "owner_username",
            "status",
            "model_name",
            "sharding_strategy",
            "target_nodes",
            "node_count",
            "active_nodes",
            "total_flops",
            "total_vram_gb",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "owner",
            "status",
            "total_flops",
            "total_vram_gb",
            "created_at",
            "updated_at",
        ]

    def get_node_count(self, obj):
        return obj.nodes.count()

    def get_active_nodes(self, obj):
        return obj.active_nodes.count()


class FederationClusterDetailSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    nodes = FederationNodeSerializer(many=True, read_only=True)
    active_tasks = serializers.SerializerMethodField()

    class Meta:
        model = FederationCluster
        fields = [
            "id",
            "name",
            "description",
            "owner",
            "owner_username",
            "model_name",
            "model_config",
            "sharding_strategy",
            "min_nodes",
            "max_nodes",
            "target_nodes",
            "status",
            "total_flops",
            "total_vram_gb",
            "nodes",
            "active_tasks",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "owner",
            "status",
            "total_flops",
            "total_vram_gb",
            "nodes",
            "created_at",
            "updated_at",
        ]

    def get_active_tasks(self, obj):
        tasks = obj.tasks.filter(status__in=["running", "initializing"])
        return FederationTaskSerializer(tasks, many=True).data


class FederationTaskShardSerializer(serializers.ModelSerializer):
    node_gpu = serializers.CharField(source="node.listing.gpu_model", read_only=True)

    class Meta:
        model = FederationTaskShard
        fields = [
            "id",
            "task",
            "node",
            "node_gpu",
            "status",
            "steps_completed",
            "samples_processed",
            "avg_loss",
            "started_at",
            "completed_at",
        ]
        read_only_fields = ["id", "started_at", "completed_at"]


class FederationTaskSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    cluster_name = serializers.CharField(source="cluster.name", read_only=True)
    shards = FederationTaskShardSerializer(many=True, read_only=True)
    result = serializers.SerializerMethodField()

    class Meta:
        model = FederationTask
        fields = [
            "id",
            "cluster",
            "cluster_name",
            "training_run",
            "owner",
            "owner_username",
            "status",
            "batch_size_per_node",
            "micro_batch_size",
            "epochs",
            "learning_rate",
            "optimizer",
            "current_step",
            "total_steps",
            "progress_percent",
            "total_loss",
            "total_throughput",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
            "shards",
            "result",
        ]
        read_only_fields = [
            "id",
            "owner",
            "status",
            "current_step",
            "total_steps",
            "progress_percent",
            "total_loss",
            "total_throughput",
            "started_at",
            "completed_at",
            "created_at",
            "updated_at",
        ]

    def get_result(self, obj):
        try:
            r = obj.result
            return {
                "final_loss": r.final_loss,
                "final_accuracy": r.final_accuracy,
                "total_time_seconds": r.total_time_seconds,
                "total_cost_usd": str(r.total_cost_usd),
            }
        except:
            return None


class FederationResultSerializer(serializers.ModelSerializer):
    task_id = serializers.CharField(source="task.id", read_only=True)

    class Meta:
        model = FederationResult
        fields = [
            "id",
            "task",
            "task_id",
            "final_loss",
            "final_accuracy",
            "total_time_seconds",
            "total_cost_usd",
            "metrics",
            "created_at",
        ]
        read_only_fields = ["id", "task", "task_id", "created_at"]
