from rest_framework import serializers

from .models import ImpactPrediction, ResearchProject


class ResearchProjectSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = ResearchProject
        fields = [
            "id",
            "user",
            "username",
            "title",
            "abstract",
            "domain",
            "institution_name",
            "institution_age_years",
            "institution_location",
            "investigator_h_index",
            "investigator_publications",
            "investigator_previous_grants",
            "social_followers",
            "preprint_views",
            "online_mentions",
            "previous_funding_total",
            "previous_grants_count",
            "status",
            "notes",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "username",
            "status",
            "created_at",
            "updated_at",
        ]


class ImpactPredictionSerializer(serializers.ModelSerializer):
    project_title = serializers.CharField(source="project.title", read_only=True)

    class Meta:
        model = ImpactPrediction
        fields = [
            "id",
            "project",
            "project_title",
            "impact_score",
            "confidence_lower",
            "confidence_upper",
            "percentile",
            "funding_probability",
            "publication_tier",
            "comparison_text",
            "factor_scores",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]
