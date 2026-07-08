from rest_framework import serializers

from .models import ExperimentDataset, SavedVisualization


class ExperimentDatasetSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = ExperimentDataset
        fields = [
            "id",
            "user",
            "username",
            "name",
            "description",
            "data",
            "hyperparameter_columns",
            "metric_columns",
            "row_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "username",
            "hyperparameter_columns",
            "metric_columns",
            "row_count",
            "created_at",
            "updated_at",
        ]


class SavedVisualizationSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source="dataset.name", read_only=True)

    class Meta:
        model = SavedVisualization
        fields = [
            "id",
            "user",
            "dataset",
            "dataset_name",
            "name",
            "viz_type",
            "config",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]
