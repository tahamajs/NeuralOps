from django.contrib.auth import get_user_model
from rest_framework import serializers

from .models import (ActivityFeed, Connection, Discussion, DiscussionReply,
                     GroupMembership, PublicationAlert, ResearchGroup)

User = get_user_model()


class ConnectionSerializer(serializers.ModelSerializer):
    from_user_username = serializers.CharField(
        source="from_user.username", read_only=True
    )
    to_user_username = serializers.CharField(source="to_user.username", read_only=True)

    class Meta:
        model = Connection
        fields = [
            "id",
            "from_user",
            "to_user",
            "from_user_username",
            "to_user_username",
            "connection_type",
            "status",
            "created_at",
        ]
        read_only_fields = ["id", "from_user", "created_at", "updated_at"]


class GroupMembershipSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    avatar_url = serializers.SerializerMethodField()

    class Meta:
        model = GroupMembership
        fields = ["id", "user", "username", "avatar_url", "role", "joined_at"]
        read_only_fields = ["id", "joined_at"]

    def get_avatar_url(self, obj):
        try:
            return obj.user.profile.avatar_url
        except:
            return None


class ResearchGroupSerializer(serializers.ModelSerializer):
    member_count = serializers.IntegerField(read_only=True)
    current_user_role = serializers.SerializerMethodField()

    class Meta:
        model = ResearchGroup
        fields = [
            "id",
            "name",
            "slug",
            "description",
            "avatar",
            "member_count",
            "is_private",
            "tags",
            "discussion_count",
            "created_at",
            "current_user_role",
        ]
        read_only_fields = [
            "id",
            "member_count",
            "discussion_count",
            "created_at",
            "updated_at",
        ]

    def get_current_user_role(self, obj):
        request = self.context.get("request")
        if request and request.user.is_authenticated:
            membership = obj.groupmembership_set.filter(user=request.user).first()
            return membership.role if membership else None
        return None


class ResearchGroupDetailSerializer(serializers.ModelSerializer):
    members = serializers.SerializerMethodField()
    member_count = serializers.IntegerField(read_only=True)
    current_user_role = serializers.SerializerMethodField()

    class Meta:
        model = ResearchGroup
        fields = [
            "id",
            "name",
            "slug",
            "description",
            "avatar",
            "members",
            "member_count",
            "is_private",
            "tags",
            "discussion_count",
            "created_at",
            "current_user_role",
        ]
        read_only_fields = [
            "id",
            "member_count",
            "discussion_count",
            "created_at",
            "updated_at",
        ]

    def get_members(self, obj):
        return GroupMembershipSerializer(obj.groupmembership_set.all(), many=True).data

    def get_current_user_role(self, obj):
        request = self.context.get("request")
        if request and request.user.is_authenticated:
            membership = obj.groupmembership_set.filter(user=request.user).first()
            return membership.role if membership else None
        return None


class DiscussionReplySerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source="author.username", read_only=True)

    class Meta:
        model = DiscussionReply
        fields = [
            "id",
            "discussion",
            "author",
            "author_username",
            "content",
            "created_at",
        ]
        read_only_fields = ["id", "author", "created_at"]


class DiscussionSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source="author.username", read_only=True)
    reply_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Discussion
        fields = [
            "id",
            "group",
            "author",
            "author_username",
            "title",
            "content",
            "is_pinned",
            "is_closed",
            "view_count",
            "reply_count",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "author",
            "view_count",
            "reply_count",
            "created_at",
            "updated_at",
        ]


class DiscussionDetailSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source="author.username", read_only=True)
    replies = DiscussionReplySerializer(many=True, read_only=True)
    reply_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Discussion
        fields = [
            "id",
            "group",
            "author",
            "author_username",
            "title",
            "content",
            "is_pinned",
            "is_closed",
            "view_count",
            "reply_count",
            "replies",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "author",
            "view_count",
            "reply_count",
            "replies",
            "created_at",
            "updated_at",
        ]


class ActivityFeedSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = ActivityFeed
        fields = [
            "id",
            "user",
            "username",
            "activity_type",
            "title",
            "description",
            "link",
            "metadata",
            "created_at",
        ]


class PublicationAlertSerializer(serializers.ModelSerializer):
    user_username = serializers.CharField(source="user.username", read_only=True)

    class Meta:
        model = PublicationAlert
        fields = [
            "id",
            "user",
            "follower",
            "user_username",
            "alert_type",
            "content",
            "link",
            "is_read",
            "created_at",
        ]
        read_only_fields = ["id", "is_read", "created_at"]
