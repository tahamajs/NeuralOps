from rest_framework import serializers

from .models import (AutonomousDecision, PipelineArtifact, PipelineStep,
                     PipelineTemplate, ResearchPipeline)


class PipelineStepSerializer(serializers.ModelSerializer):
    class Meta:
        model = PipelineStep
        fields = [
            "id",
            "step_type",
            "order",
            "status",
            "title",
            "description",
            "input_data",
            "output_data",
            "llm_tokens_used",
            "started_at",
            "completed_at",
            "error_message",
            "created_at",
        ]
        read_only_fields = ["id", "status", "started_at", "completed_at", "created_at"]


class AutonomousDecisionSerializer(serializers.ModelSerializer):
    class Meta:
        model = AutonomousDecision
        fields = [
            "id",
            "pipeline",
            "step",
            "decision_type",
            "reasoning",
            "alternatives",
            "confidence",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class PipelineArtifactSerializer(serializers.ModelSerializer):
    step_title = serializers.CharField(
        source="step.title", read_only=True, default=None
    )

    class Meta:
        model = PipelineArtifact
        fields = [
            "id",
            "pipeline",
            "step",
            "step_title",
            "artifact_type",
            "title",
            "content",
            "metadata",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class ResearchPipelineSerializer(serializers.ModelSerializer):
    steps = PipelineStepSerializer(many=True, read_only=True)
    artifacts = PipelineArtifactSerializer(many=True, read_only=True)
    username = serializers.CharField(source="user.username", read_only=True)
    step_count = serializers.SerializerMethodField()
    artifact_count = serializers.SerializerMethodField()

    class Meta:
        model = ResearchPipeline
        fields = [
            "id",
            "user",
            "username",
            "name",
            "description",
            "research_question",
            "status",
            "progress",
            "auto_mode",
            "max_compute_hours",
            "max_llm_cost_cents",
            "compute_hours_used",
            "llm_cost_cents_used",
            "final_hypothesis",
            "final_manuscript",
            "final_metrics",
            "steps",
            "artifacts",
            "step_count",
            "artifact_count",
            "created_at",
            "updated_at",
            "completed_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "status",
            "progress",
            "compute_hours_used",
            "llm_cost_cents_used",
            "final_hypothesis",
            "final_manuscript",
            "final_metrics",
            "created_at",
            "updated_at",
            "completed_at",
        ]

    def get_step_count(self, obj):
        return obj.steps.count()

    def get_artifact_count(self, obj):
        return obj.artifacts.count()


class PipelineTemplateSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(
        source="created_by.username", read_only=True, default=None
    )

    class Meta:
        model = PipelineTemplate
        fields = [
            "id",
            "name",
            "description",
            "category",
            "steps_config",
            "is_public",
            "created_by",
            "created_by_username",
            "usage_count",
            "created_at",
        ]
        read_only_fields = ["id", "created_by", "usage_count", "created_at"]
