from rest_framework import serializers

from .models import CodeExperiment, CodeVariant, ExperimentRun


class CodeVariantSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(
        source="created_by.username", read_only=True
    )

    class Meta:
        model = CodeVariant
        fields = [
            "id",
            "experiment",
            "created_by",
            "created_by_username",
            "version_number",
            "description",
            "hyperparameters",
            "status",
            "score",
            "metrics",
            "parent_variant",
            "iteration",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "created_by",
            "version_number",
            "status",
            "score",
            "metrics",
            "created_at",
            "updated_at",
        ]


class CodeExperimentSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    variant_count = serializers.SerializerMethodField()
    best_score = serializers.SerializerMethodField()

    class Meta:
        model = CodeExperiment
        fields = [
            "id",
            "name",
            "description",
            "owner",
            "owner_username",
            "mode",
            "status",
            "max_variants",
            "max_iterations",
            "metric_to_optimize",
            "metric_direction",
            "target_score",
            "best_variant",
            "variant_count",
            "best_score",
            "total_variants_tested",
            "total_iterations",
            "elapsed_seconds",
            "created_at",
            "updated_at",
            "completed_at",
        ]
        read_only_fields = [
            "id",
            "owner",
            "status",
            "best_variant",
            "total_variants_tested",
            "total_iterations",
            "elapsed_seconds",
            "created_at",
            "updated_at",
            "completed_at",
        ]

    def get_variant_count(self, obj):
        return obj.variants.count()

    def get_best_score(self, obj):
        if obj.best_variant and obj.best_variant.score is not None:
            return obj.best_variant.score
        return None


class ExperimentRunSerializer(serializers.ModelSerializer):
    variant_info = serializers.SerializerMethodField()

    class Meta:
        model = ExperimentRun
        fields = [
            "id",
            "variant",
            "variant_info",
            "status",
            "metrics",
            "logs",
            "error_message",
            "duration_seconds",
            "started_at",
            "completed_at",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "status",
            "metrics",
            "logs",
            "error_message",
            "duration_seconds",
            "started_at",
            "completed_at",
            "created_at",
        ]

    def get_variant_info(self, obj):
        return {
            "version_number": obj.variant.version_number,
            "experiment_name": obj.variant.experiment.name,
        }
