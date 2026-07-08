from rest_framework import serializers

from .models import (CreativeVariant, ExperimentRecommendation,
                     ExperimentTemplate, OpportunityIndex)


class ExperimentTemplateSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(
        source="created_by.username", read_only=True
    )

    class Meta:
        model = ExperimentTemplate
        fields = [
            "id",
            "name",
            "description",
            "category",
            "config",
            "predicted_impact",
            "success_rate",
            "usage_count",
            "tags",
            "created_by",
            "created_by_username",
            "is_public",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "created_by",
            "usage_count",
            "created_at",
            "updated_at",
        ]


class OpportunityIndexSerializer(serializers.ModelSerializer):
    class Meta:
        model = OpportunityIndex
        fields = [
            "id",
            "user",
            "attribute",
            "current_score",
            "potential_score",
            "gap",
            "impact_weight",
            "opportunity_score",
            "recommendations",
            "updated_at",
        ]
        read_only_fields = ["id", "user", "updated_at"]


class ExperimentRecommendationSerializer(serializers.ModelSerializer):
    template_name = serializers.CharField(
        source="template.name", read_only=True, default=None
    )

    class Meta:
        model = ExperimentRecommendation
        fields = [
            "id",
            "user",
            "template",
            "template_name",
            "title",
            "description",
            "config",
            "predicted_lift",
            "confidence",
            "reason",
            "status",
            "metadata",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "user", "status", "created_at", "updated_at"]


class CreativeVariantSerializer(serializers.ModelSerializer):
    base_experiment_name = serializers.CharField(
        source="base_experiment.name", read_only=True, default=None
    )

    class Meta:
        model = CreativeVariant
        fields = [
            "id",
            "base_experiment",
            "base_experiment_name",
            "user",
            "title",
            "description",
            "changes",
            "predicted_lift",
            "reasoning",
            "status",
            "created_at",
        ]
        read_only_fields = ["id", "user", "status", "created_at"]
