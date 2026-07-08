from rest_framework import serializers

from .models import (ActivityEvent, Endorsement, Portfolio, ShowcaseItem,
                     WidgetEmbed)


class ShowcaseItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShowcaseItem
        fields = "__all__"
        read_only_fields = ["id", "portfolio", "created_at"]


class EndorsementSerializer(serializers.ModelSerializer):
    endorser_username = serializers.CharField(
        source="endorser.username", read_only=True
    )

    class Meta:
        model = Endorsement
        fields = [
            "id",
            "portfolio",
            "endorser",
            "endorser_username",
            "skill",
            "created_at",
        ]
        read_only_fields = ["id", "portfolio", "endorser", "created_at"]


class PortfolioSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    display_name = serializers.SerializerMethodField()
    avatar_url = serializers.SerializerMethodField()
    items = ShowcaseItemSerializer(many=True, read_only=True)
    endorsements = serializers.SerializerMethodField()
    star_count = serializers.SerializerMethodField()
    citation_count = serializers.SerializerMethodField()
    total_experiments = serializers.SerializerMethodField()

    class Meta:
        model = Portfolio
        fields = [
            "id",
            "username",
            "display_name",
            "avatar_url",
            "headline",
            "bio",
            "is_public",
            "theme_color",
            "items",
            "endorsements",
            "star_count",
            "citation_count",
            "total_experiments",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "user", "created_at", "updated_at"]

    def get_display_name(self, obj):
        try:
            profile = obj.user.profile
            return profile.display_name or obj.user.username
        except:
            return obj.user.username

    def get_avatar_url(self, obj):
        try:
            return obj.user.profile.avatar_url
        except:
            return None

    def get_endorsements(self, obj):
        endorsements = obj.endorsements.all()
        # Group by skill and count
        from collections import Counter

        skills = Counter(e.skill for e in endorsements)
        return [
            {
                "skill": skill,
                "count": count,
                "endorsers": [
                    e.endorser.username for e in endorsements if e.skill == skill
                ],
            }
            for skill, count in skills.most_common()
        ]

    def get_star_count(self, obj):
        # Sum of votes on user's leaderboard submissions (or models)
        from leaderboards.models import Vote

        return Vote.objects.filter(submission__user=obj.user, direction="up").count()

    def get_citation_count(self, obj):
        # Sum of citation counts from saved papers
        from agents.models import ResearchPaper

        return (
            ResearchPaper.objects.filter(saved_by=obj.user).aggregate(
                total=models.Sum("citation_count")
            )["total"]
            or 0
        )

    def get_total_experiments(self, obj):
        from jobscheduler.models import Job

        return Job.objects.filter(user=obj.user).count()


class ActivitySerializer(serializers.ModelSerializer):
    class Meta:
        model = ActivityEvent
        fields = ["date", "event_type", "count"]


class WidgetEmbedSerializer(serializers.ModelSerializer):
    class Meta:
        model = WidgetEmbed
        fields = "__all__"
        read_only_fields = ["id", "portfolio", "created_at"]
