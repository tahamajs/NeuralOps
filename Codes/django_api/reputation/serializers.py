from rest_framework import serializers

from .models import (
    Recommendation,
    ReputationEvent,
    ReputationScore,
    ReputationSnapshot,
    ReputationWeight,
)


class ReputationWeightSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReputationWeight
        fields = [
            "OTHER_READS",
            "FULL_TEXT_READS",
            "RECOMMENDATIONS",
            "CITATIONS",
            "updated_at",
        ]


class ReputationScoreSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    display_name = serializers.SerializerMethodField()
    rank = serializers.SerializerMethodField()

    class Meta:
        model = ReputationScore
        fields = [
            "id",
            "user",
            "username",
            "display_name",
            "rank",
            "total_score",
            "other_reads_score",
            "full_text_reads_score",
            "recommendations_score",
            "citations_score",
            "total_papers",
            "total_citations",
            "total_recommendations",
            "total_reads",
            "percentile",
            "updated_at",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "username",
            "display_name",
            "rank",
            "total_score",
            "other_reads_score",
            "full_text_reads_score",
            "recommendations_score",
            "citations_score",
            "total_papers",
            "total_citations",
            "total_recommendations",
            "total_reads",
            "percentile",
            "updated_at",
            "created_at",
        ]

    def get_display_name(self, obj):
        try:
            return obj.user.profile.display_name or obj.user.username
        except:
            return obj.user.username

    def get_rank(self, obj):
        return (
            ReputationScore.objects.filter(total_score__gt=obj.total_score).count() + 1
        )


class ReputationEventSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    source_username = serializers.CharField(
        source="source_user.username", read_only=True, default=None
    )
    paper_title = serializers.CharField(
        source="paper.title", read_only=True, default=None
    )

    class Meta:
        model = ReputationEvent
        fields = [
            "id",
            "user",
            "username",
            "event_type",
            "paper",
            "paper_title",
            "source_user",
            "source_username",
            "weight",
            "weighted_value",
            "description",
            "created_at",
        ]
        read_only_fields = ["id", "user", "username", "created_at"]


class RecommendationSerializer(serializers.ModelSerializer):
    paper_title = serializers.CharField(source="paper.title", read_only=True)
    paper_abstract = serializers.SerializerMethodField()
    paper_authors = serializers.JSONField(source="paper.authors", read_only=True)
    paper_year = serializers.IntegerField(source="paper.year", read_only=True)
    paper_url = serializers.URLField(source="paper.url", read_only=True)

    class Meta:
        model = Recommendation
        fields = [
            "id",
            "user",
            "paper",
            "paper_title",
            "paper_abstract",
            "paper_authors",
            "paper_year",
            "paper_url",
            "score",
            "reason",
            "is_dismissed",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]

    def get_paper_abstract(self, obj):
        return (obj.paper.abstract or "")[:300]


class ReputationSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReputationSnapshot
        fields = [
            "id",
            "user",
            "total_score",
            "other_reads_score",
            "full_text_reads_score",
            "recommendations_score",
            "citations_score",
            "percentile",
            "recorded_at",
        ]
