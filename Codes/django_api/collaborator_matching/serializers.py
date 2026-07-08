from rest_framework import serializers

from .models import CollaboratorMatch, ResearcherInterestProfile


class ResearcherInterestProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = ResearcherInterestProfile
        fields = ["id", "user", "username", "topics", "paper_count", "last_generated"]
        read_only_fields = [
            "id",
            "user",
            "username",
            "topics",
            "paper_count",
            "last_generated",
        ]


class CollaboratorMatchSerializer(serializers.ModelSerializer):
    matched_username = serializers.CharField(
        source="matched_user.username", read_only=True
    )
    matched_display_name = serializers.SerializerMethodField()
    matched_avatar_url = serializers.SerializerMethodField()

    class Meta:
        model = CollaboratorMatch
        fields = [
            "id",
            "user",
            "matched_user",
            "matched_username",
            "matched_display_name",
            "matched_avatar_url",
            "score",
            "common_interests",
            "explanation",
            "status",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "score",
            "common_interests",
            "explanation",
            "status",
            "created_at",
        ]

    def get_matched_display_name(self, obj):
        try:
            profile = obj.matched_user.profile
            return profile.display_name or obj.matched_user.username
        except:
            return obj.matched_user.username

    def get_matched_avatar_url(self, obj):
        try:
            return obj.matched_user.profile.avatar_url
        except:
            return None
