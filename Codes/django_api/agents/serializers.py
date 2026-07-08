from rest_framework import serializers

from .models import ChatMessage, ChatSession, ResearchPaper, TrainingAnalysis


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ["id", "role", "content", "metadata", "created_at"]
        read_only_fields = fields


class ChatSessionSerializer(serializers.ModelSerializer):
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatSession
        fields = ["id", "title", "status", "message_count", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_message_count(self, obj):
        return obj.messages.count()


class ResearchPaperSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResearchPaper
        fields = [
            "id",
            "title",
            "authors",
            "abstract",
            "url",
            "source",
            "source_id",
            "year",
            "citation_count",
            "pdf_url",
            "keywords",
            "notes",
            "tags",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class TrainingAnalysisSerializer(serializers.ModelSerializer):
    run_id = serializers.UUIDField(source="run.id", read_only=True)
    run_name = serializers.CharField(source="run.name", read_only=True)

    class Meta:
        model = TrainingAnalysis
        fields = [
            "id",
            "run_id",
            "run_name",
            "status",
            "confidence",
            "suggestions",
            "metrics_summary",
            "created_at",
        ]
        read_only_fields = fields
