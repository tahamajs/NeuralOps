from rest_framework import serializers

from .models import (
    AchievementUnlock,
    Badge,
    ContributionEvent,
    LeaderboardSnapshot,
    PremiumFeature,
    UserBadge,
    UserStreak,
)


class BadgeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Badge
        fields = [
            "id",
            "name",
            "description",
            "icon",
            "tier",
            "category",
            "criteria",
            "created_at",
        ]


class UserBadgeSerializer(serializers.ModelSerializer):
    badge_detail = BadgeSerializer(source="badge", read_only=True)

    class Meta:
        model = UserBadge
        fields = ["id", "user", "badge", "badge_detail", "awarded_at"]


class ContributionEventSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = ContributionEvent
        fields = [
            "id",
            "user",
            "username",
            "event_type",
            "points",
            "description",
            "related_object_id",
            "created_at",
        ]


class UserStreakSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = UserStreak
        fields = [
            "id",
            "user",
            "username",
            "current_streak",
            "longest_streak",
            "total_points",
            "weekly_points",
            "last_active_date",
            "updated_at",
        ]


class LeaderboardSnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = LeaderboardSnapshot
        fields = ["id", "date", "field", "institution", "rankings", "created_at"]


class AchievementUnlockSerializer(serializers.ModelSerializer):
    class Meta:
        model = AchievementUnlock
        fields = ["id", "user", "achievement", "required_points", "unlocked_at"]


class PremiumFeatureSerializer(serializers.ModelSerializer):
    required_badges = BadgeSerializer(many=True, read_only=True)

    class Meta:
        model = PremiumFeature
        fields = [
            "id",
            "name",
            "description",
            "required_points",
            "required_badges",
            "is_active",
        ]
