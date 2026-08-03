from rest_framework import serializers

from .models import (
    CitationPath,
    PaperFeedItem,
    PaperGraphCache,
    PaperRecommendation,
    ResearchLandscape,
)


class PaperGraphCacheSerializer(serializers.ModelSerializer):
    paper_title = serializers.CharField(source="paper.title", read_only=True)

    class Meta:
        model = PaperGraphCache
        fields = [
            "id",
            "paper",
            "paper_title",
            "citations",
            "references",
            "co_cited",
            "last_updated",
        ]


class CitationPathSerializer(serializers.ModelSerializer):
    source_title = serializers.CharField(source="source_paper.title", read_only=True)
    target_title = serializers.CharField(source="target_paper.title", read_only=True)

    class Meta:
        model = CitationPath
        fields = [
            "id",
            "source_paper",
            "target_paper",
            "source_title",
            "target_title",
            "path",
            "length",
            "created_at",
        ]


class PaperRecommendationSerializer(serializers.ModelSerializer):
    paper_title = serializers.CharField(source="paper.title", read_only=True)
    paper_abstract = serializers.SerializerMethodField()
    paper_authors = serializers.JSONField(source="paper.authors", read_only=True)
    paper_year = serializers.IntegerField(source="paper.year", read_only=True)
    paper_url = serializers.URLField(source="paper.url", read_only=True)
    paper_citations = serializers.IntegerField(
        source="paper.citation_count", read_only=True
    )

    class Meta:
        model = PaperRecommendation
        fields = [
            "id",
            "user",
            "paper",
            "paper_title",
            "paper_abstract",
            "paper_authors",
            "paper_year",
            "paper_url",
            "paper_citations",
            "score",
            "reason",
            "source_type",
            "is_dismissed",
            "is_saved",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]

    def get_paper_abstract(self, obj):
        return (obj.paper.abstract or "")[:300]


class ResearchLandscapeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ResearchLandscape
        fields = [
            "id",
            "user",
            "name",
            "description",
            "graph_data",
            "seed_papers",
            "created_at",
        ]


class PaperFeedItemSerializer(serializers.ModelSerializer):
    paper_title = serializers.CharField(source="paper.title", read_only=True)
    paper_abstract = serializers.SerializerMethodField()

    class Meta:
        model = PaperFeedItem
        fields = [
            "id",
            "user",
            "paper",
            "paper_title",
            "paper_abstract",
            "seen",
            "clicked",
            "created_at",
        ]
        read_only_fields = ["id", "user", "created_at"]

    def get_paper_abstract(self, obj):
        return (obj.paper.abstract or "")[:200]
