from rest_framework import serializers

from .models import Experiment, TrainingRun, TrainingSnapshot


class ExperimentSerializer(serializers.ModelSerializer):
    runs_count = serializers.IntegerField(source="runs.count", read_only=True)

    class Meta:
        model = Experiment
        fields = "__all__"
        read_only_fields = ("id", "created_at", "updated_at")


class TrainingSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrainingSnapshot
        fields = "__all__"
        read_only_fields = ("id", "timestamp")


class TrainingRunSerializer(serializers.ModelSerializer):
    experiment_name = serializers.CharField(source="experiment.name", read_only=True)
    snapshots = TrainingSnapshotSerializer(many=True, read_only=True)

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
