from rest_framework import serializers

from .models import ExperimentInsight, ExperimentRecommendation, ExperimentSummary


class ExperimentInsightSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExperimentInsight
        fields = [
            "id",
            "user",
            "insight_type",
            "title",
            "description",
            "confidence",
            "related_runs",
            "metadata",
            "is_read",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]


class ExperimentRecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExperimentRecommendation
        fields = [
            "id",
            "user",
            "title",
            "description",
            "predicted_improvement",
            "confidence",
            "configuration",
            "rationale",
            "based_on_runs",
            "status",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]


class ExperimentSummarySerializer(serializers.ModelSerializer):
    class Meta:
        model = ExperimentSummary
        fields = [
            "id",
            "user",
            "period_start",
            "period_end",
            "total_experiments",
            "successful_experiments",
            "best_accuracy",
            "best_loss",
            "top_recommendations",
            "insights_text",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]
