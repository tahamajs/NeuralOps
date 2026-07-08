from rest_framework import serializers

from .models import (Hypothesis, HypothesisReview, IdeaTournament,
                     LiteratureCitation, TournamentLog)


class LiteratureCitationSerializer(serializers.ModelSerializer):
    class Meta:
        model = LiteratureCitation
        fields = [
            "id",
            "hypothesis",
            "title",
            "authors",
            "year",
            "url",
            "snippet",
            "relevance_score",
        ]
        read_only_fields = ["id"]


class HypothesisReviewSerializer(serializers.ModelSerializer):
    class Meta:
        model = HypothesisReview
        fields = [
            "id",
            "hypothesis",
            "agent",
            "round_number",
            "score",
            "strengths",
            "weaknesses",
            "suggestions",
            "overall_assessment",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class HypothesisSerializer(serializers.ModelSerializer):
    reviews = HypothesisReviewSerializer(many=True, read_only=True)
    citations = LiteratureCitationSerializer(many=True, read_only=True)
    generated_by_label = serializers.CharField(
        source="get_generated_by_display", read_only=True
    )
    status_label = serializers.CharField(source="get_status_display", read_only=True)

    class Meta:
        model = Hypothesis
        fields = [
            "id",
            "tournament",
            "title",
            "content",
            "round_created",
            "status",
            "status_label",
            "score",
            "rank",
            "generated_by",
            "generated_by_label",
            "parent_hypothesis",
            "reviews",
            "citations",
            "created_at",
        ]
        read_only_fields = ["id", "score", "rank", "created_at"]


class TournamentLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = TournamentLog
        fields = [
            "id",
            "tournament",
            "round_number",
            "agent",
            "action",
            "description",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class IdeaTournamentSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    hypothesis_count = serializers.SerializerMethodField()
    status_label = serializers.CharField(source="get_status_display", read_only=True)

    class Meta:
        model = IdeaTournament
        fields = [
            "id",
            "owner",
            "owner_username",
            "research_domain",
            "research_question",
            "status",
            "status_label",
            "num_hypotheses",
            "num_rounds",
            "top_k",
            "total_hypotheses_generated",
            "current_round",
            "hypothesis_count",
            "created_at",
            "completed_at",
        ]
        read_only_fields = [
            "id",
            "owner",
            "status",
            "total_hypotheses_generated",
            "current_round",
            "created_at",
            "completed_at",
        ]

    def get_hypothesis_count(self, obj):
        return obj.hypotheses.count()


class IdeaTournamentDetailSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    hypotheses = serializers.SerializerMethodField()
    logs = serializers.SerializerMethodField()

    class Meta:
        model = IdeaTournament
        fields = [
            "id",
            "owner",
            "owner_username",
            "research_domain",
            "research_question",
            "status",
            "num_hypotheses",
            "num_rounds",
            "top_k",
            "total_hypotheses_generated",
            "current_round",
            "hypotheses",
            "logs",
            "created_at",
            "completed_at",
        ]
        read_only_fields = [
            "id",
            "owner",
            "status",
            "total_hypotheses_generated",
            "current_round",
            "created_at",
            "completed_at",
        ]

    def get_hypotheses(self, obj):
        qs = obj.hypotheses.order_by("-score")
        return HypothesisSerializer(qs, many=True).data

    def get_logs(self, obj):
        qs = obj.logs.order_by("-round_number", "-created_at")[:50]
        return TournamentLogSerializer(qs, many=True).data
