# profiles/serializers.py
from django.contrib.auth.models import User
from rest_framework import serializers

from .models import Profile


class ProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    email = serializers.CharField(source="user.email", read_only=True)
    date_joined = serializers.DateTimeField(source="user.date_joined", read_only=True)
    avatar_url = serializers.CharField(read_only=True)
    full_name = serializers.SerializerMethodField()

    class Meta:
        model = Profile
        fields = [
            "id",
            "username",
            "email",
            "display_name",
            "full_name",
            "bio",
            "website",
            "location",
            "avatar",
            "avatar_url",
            "affiliation",
            "research_interests",
            "skills",
            "github",
            "twitter",
            "linkedin",
            "google_scholar",
            "orcid",
            "total_experiments",
            "total_compressions",
            "total_papers",
            "date_joined",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "username",
            "email",
            "avatar_url",
            "full_name",
            "date_joined",
            "created_at",
            "updated_at",
            "total_experiments",
            "total_compressions",
            "total_papers",
        ]

    def get_full_name(self, obj):
        if obj.display_name:
            return obj.display_name
        u = obj.user
        if u.first_name or u.last_name:
            return f"{u.first_name} {u.last_name}".strip()
        return u.username


class ProfileUpdateSerializer(serializers.ModelSerializer):
    first_name = serializers.CharField(source="user.first_name", required=False)
    last_name = serializers.CharField(source="user.last_name", required=False)

    class Meta:
        model = Profile
        fields = [
            "display_name",
            "bio",
            "website",
            "location",
            "avatar",
            "affiliation",
            "research_interests",
            "skills",
            "github",
            "twitter",
            "linkedin",
            "google_scholar",
            "orcid",
            "first_name",
            "last_name",
        ]

    def update(self, instance, validated_data):
        user_data = validated_data.pop("user", {})
        if user_data:
            user = instance.user
            user.first_name = user_data.get("first_name", user.first_name)
            user.last_name = user_data.get("last_name", user.last_name)
            user.save()
        return super().update(instance, validated_data)


class UserSearchSerializer(serializers.ModelSerializer):
    """Minimal serializer for user search results."""

    avatar_url = serializers.SerializerMethodField()
    display_name = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ["id", "username", "display_name", "avatar_url"]

    def get_avatar_url(self, obj):
        try:
            return obj.profile.avatar_url
        except Exception:
            return None

    def get_display_name(self, obj):
        try:
            return obj.profile.display_name or obj.username
        except Exception:
            return obj.username
