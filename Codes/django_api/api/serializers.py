from experiment.models import (ActivityLog, Experiment, TrainingRun,
                               TrainingSnapshot)
from framework.core.registry import (ATTENTION_REGISTRY, LOSS_REGISTRY,
                                     METHOD_REGISTRY, OPTIMIZER_REGISTRY,
                                     TRAINER_REGISTRY)
from rest_framework import serializers


class ExperimentSerializer(serializers.ModelSerializer):
    runs_count = serializers.IntegerField(source="runs.count", read_only=True)

    class Meta:
        model = Experiment
        fields = "__all__"
        read_only_fields = ("id", "created_at", "updated_at")


class TrainingRunSerializer(serializers.ModelSerializer):
    experiment_name = serializers.CharField(source="experiment.name", read_only=True)

    class Meta:
        model = TrainingRun
        fields = "__all__"
        read_only_fields = (
            "id",
            "created_at",
            "started_at",
            "completed_at",
            "logs",
            "training_history",
        )


class TrainingSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingSnapshot
        fields = "__all__"
        read_only_fields = ("id", "timestamp")


class ActivityLogSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = ActivityLog
        fields = "__all__"
        read_only_fields = ("id", "created_at")


class TrainingConfigSerializer(serializers.Serializer):
    model_name = serializers.CharField()
    task_name = serializers.CharField()
    attn_type = serializers.ChoiceField(choices=[])
    method = serializers.ChoiceField(choices=[])
    compressor = serializers.CharField(required=False, allow_blank=True)
    trainer_type = serializers.ChoiceField(choices=[])
    loss_type = serializers.ChoiceField(choices=[])
    optimizer_type = serializers.ChoiceField(choices=[])
    selected_callbacks = serializers.ListField(
        child=serializers.CharField(), required=False
    )
    epochs = serializers.IntegerField()
    batch_size = serializers.IntegerField()
    learning_rate = serializers.FloatField()
    max_seq_length = serializers.IntegerField()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["attn_type"].choices = list(ATTENTION_REGISTRY.keys())
        self.fields["method"].choices = list(METHOD_REGISTRY.keys())
        self.fields["trainer_type"].choices = list(TRAINER_REGISTRY.keys())
        self.fields["loss_type"].choices = list(LOSS_REGISTRY.keys())
        self.fields["optimizer_type"].choices = list(OPTIMIZER_REGISTRY.keys())
