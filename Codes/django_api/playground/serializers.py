from rest_framework import serializers

from .models import *


class PromptTemplateSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(
        source="created_by.username", read_only=True
    )
    version_count = serializers.SerializerMethodField()

    class Meta:
        model = PromptTemplate
        fields = [
            "id",
            "name",
            "description",
            "content",
            "variables_schema",
            "tags",
            "is_public",
            "created_by",
            "created_by_username",
            "version",
            "parent",
            "usage_count",
            "version_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "created_by",
            "version",
            "usage_count",
            "created_at",
            "updated_at",
        ]

    def get_version_count(self, obj):
        return obj.versions.count()


class ModelConfigSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelConfig
        fields = [
            "id",
            "name",
            "created_by",
            "provider",
            "model_name",
            "temperature",
            "top_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
            "system_prompt",
            "extra_params",
            "is_default",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_by", "created_at", "updated_at"]


class PlaygroundMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = PlaygroundMessage
        fields = [
            "id",
            "session",
            "role",
            "content",
            "tool_calls",
            "token_count",
            "latency_ms",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class PlaygroundSessionSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    messages = PlaygroundMessageSerializer(many=True, read_only=True)
    model_config_name = serializers.CharField(
        source="model_config.name", read_only=True, default=None
    )
    prompt_template_name = serializers.CharField(
        source="prompt_template.name", read_only=True, default=None
    )

    class Meta:
        model = PlaygroundSession
        fields = [
            "id",
            "user",
            "username",
            "name",
            "model_config",
            "model_config_name",
            "prompt_template",
            "prompt_template_name",
            "variables",
            "messages",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "user", "created_at", "updated_at"]


class ComparisonResultSerializer(serializers.ModelSerializer):
    model_name = serializers.CharField(
        source="model_config.name", read_only=True, default=None
    )
    prompt_name = serializers.CharField(
        source="prompt_template.name", read_only=True, default=None
    )

    class Meta:
        model = ComparisonResult
        fields = [
            "id",
            "comparison_set",
            "model_config",
            "model_name",
            "prompt_template",
            "prompt_name",
            "output_text",
            "latency_ms",
            "token_count",
            "cost_cents",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class ComparisonSetSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    results = ComparisonResultSerializer(many=True, read_only=True)

    class Meta:
        model = ComparisonSet
        fields = [
            "id",
            "user",
            "username",
            "name",
            "input_text",
            "results",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]


class ToolDefinitionSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(
        source="created_by.username", read_only=True
    )

    class Meta:
        model = ToolDefinition
        fields = [
            "id",
            "name",
            "description",
            "parameters_schema",
            "implementation",
            "is_active",
            "created_by",
            "created_by_username",
            "created_at",
        ]
        read_only_fields = ["id", "created_by", "created_at"]
