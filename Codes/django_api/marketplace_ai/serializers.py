from rest_framework import serializers

from .models import (
    ComparisonSet,
    ComplianceCheck,
    MarketplaceQuery,
    SmartRecommendation,
)


class MarketplaceQuerySerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = MarketplaceQuery
        fields = [
            "id",
            "user",
            "username",
            "query_text",
            "parsed_filters",
            "status",
            "result_count",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "parsed_filters",
            "status",
            "result_count",
            "created_at",
        ]


class ComparisonSetSerializer(serializers.ModelSerializer):
    listing_count = serializers.SerializerMethodField()

    class Meta:
        model = ComparisonSet
        fields = ["id", "user", "name", "listings", "listing_count", "created_at"]
        read_only_fields = ["id", "user", "created_at"]

    def get_listing_count(self, obj):
        return obj.listings.count()


class ComplianceCheckSerializer(serializers.ModelSerializer):
    listing_gpu = serializers.CharField(source="listing.gpu_model", read_only=True)

    class Meta:
        model = ComplianceCheck
        fields = [
            "id",
            "user",
            "listing",
            "listing_gpu",
            "requirements",
            "passed",
            "failures",
            "warnings",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "passed",
            "failures",
            "warnings",
            "created_at",
        ]


class SmartRecommendationSerializer(serializers.ModelSerializer):
    listing_gpu = serializers.CharField(source="listing.gpu_model", read_only=True)
    listing_price = serializers.DecimalField(
        source="listing.current_price_per_hour",
        max_digits=10,
        decimal_places=4,
        read_only=True,
    )

    class Meta:
        model = SmartRecommendation
        fields = [
            "id",
            "user",
            "listing",
            "listing_gpu",
            "listing_price",
            "score",
            "reason",
            "is_dismissed",
            "created_at",
        ]
        read_only_fields = ["id", "user", "score", "reason", "created_at"]
