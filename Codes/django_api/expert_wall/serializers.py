from rest_framework import serializers

from .models import ExpertProfile, ExpertReview, Project, ProjectApplication


class ExpertProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    display_name = serializers.SerializerMethodField()
    avatar_url = serializers.SerializerMethodField()

    class Meta:
        model = ExpertProfile
        fields = [
            "id",
            "user",
            "username",
            "display_name",
            "avatar_url",
            "skills",
            "domains",
            "expertise_level",
            "collaboration_preference",
            "seeking_collaborators",
            "bio",
            "orcid",
            "verified",
            "last_updated",
        ]
        read_only_fields = ["id", "user", "verified", "last_updated"]

    def get_display_name(self, obj):
        try:
            return obj.user.profile.display_name or obj.user.username
        except:
            return obj.user.username

    def get_avatar_url(self, obj):
        try:
            return obj.user.profile.avatar_url
        except:
            return None


class ProjectApplicationSerializer(serializers.ModelSerializer):
    applicant_username = serializers.CharField(
        source="applicant.username", read_only=True
    )
    applicant_avatar_url = serializers.SerializerMethodField()

    class Meta:
        model = ProjectApplication
        fields = [
            "id",
            "project",
            "applicant",
            "applicant_username",
            "applicant_avatar_url",
            "message",
            "status",
            "created_at",
        ]
        read_only_fields = ["id", "applicant", "status", "created_at"]

    def get_applicant_avatar_url(self, obj):
        try:
            return obj.applicant.profile.avatar_url
        except:
            return None


class ProjectSerializer(serializers.ModelSerializer):
    owner_username = serializers.CharField(source="owner.username", read_only=True)
    applications = ProjectApplicationSerializer(many=True, read_only=True)
    application_count = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = [
            "id",
            "owner",
            "owner_username",
            "title",
            "description",
            "project_type",
            "status",
            "skills_needed",
            "domains_needed",
            "collaborators_needed",
            "current_collaborators",
            "deadline",
            "is_public",
            "applications",
            "application_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "owner",
            "current_collaborators",
            "created_at",
            "updated_at",
        ]

    def get_application_count(self, obj):
        return obj.applications.count()


class ExpertReviewSerializer(serializers.ModelSerializer):
    requester_username = serializers.CharField(
        source="requester.username", read_only=True
    )
    reviewer_username = serializers.CharField(
        source="reviewer.username", read_only=True, default=None
    )

    class Meta:
        model = ExpertReview
        fields = [
            "id",
            "requester",
            "reviewer",
            "requester_username",
            "reviewer_username",
            "review_type",
            "title",
            "description",
            "attachment_url",
            "status",
            "reviewer_feedback",
            "rating",
            "created_at",
            "completed_at",
        ]
        read_only_fields = [
            "id",
            "requester",
            "reviewer",
            "status",
            "created_at",
            "completed_at",
        ]
