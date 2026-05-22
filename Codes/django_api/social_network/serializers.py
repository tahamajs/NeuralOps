from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import (
    Follow, ResearchGroup, GroupMembership,
    Discussion, DiscussionReply, ActivityFeed, PublicationAlert
)

User = get_user_model()


class FollowSerializer(serializers.ModelSerializer):
    follower_username = serializers.CharField(source='follower.username', read_only=True)
    following_username = serializers.CharField(source='following.username', read_only=True)

    class Meta:
        model = Follow
        fields = ['id', 'follower', 'following', 'follower_username', 'following_username', 'created_at']
        read_only_fields = ['id', 'follower', 'created_at']


class GroupMembershipSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    avatar_url = serializers.SerializerMethodField()

    class Meta:
        model = GroupMembership
        fields = ['id', 'user', 'username', 'avatar_url', 'role', 'joined_at']
        read_only_fields = ['id', 'joined_at']

    def get_avatar_url(self, obj):
        try:
            return obj.user.profile.avatar_url
        except:
            return None


class ResearchGroupSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    member_count = serializers.IntegerField(read_only=True)
    discussion_count = serializers.IntegerField(read_only=True)
    my_role = serializers.SerializerMethodField()

    class Meta:
        model = ResearchGroup
        fields = [
            'id', 'name', 'slug', 'description', 'cover_image',
            'created_by', 'created_by_username', 'visibility', 'tags',
            'member_count', 'discussion_count', 'my_role',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'created_by', 'member_count', 'discussion_count',
                            'created_at', 'updated_at']

    def get_my_role(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            membership = obj.groupmembership_set.filter(user=request.user).first()
            return membership.role if membership else None
        return None


class ResearchGroupDetailSerializer(serializers.ModelSerializer):
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    members = serializers.SerializerMethodField()
    member_count = serializers.IntegerField(read_only=True)
    discussion_count = serializers.IntegerField(read_only=True)
    my_role = serializers.SerializerMethodField()

    class Meta:
        model = ResearchGroup
        fields = [
            'id', 'name', 'slug', 'description', 'cover_image',
            'created_by', 'created_by_username', 'visibility', 'tags',
            'member_count', 'discussion_count', 'members', 'my_role',
            'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'created_by', 'member_count', 'discussion_count',
                            'members', 'created_at', 'updated_at']

    def get_members(self, obj):
        memberships = obj.groupmembership_set.select_related('user').all()
        return GroupMembershipSerializer(memberships, many=True).data

    def get_my_role(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            membership = obj.groupmembership_set.filter(user=request.user).first()
            return membership.role if membership else None
        return None


class DiscussionReplySerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)

    class Meta:
        model = DiscussionReply
        fields = ['id', 'discussion', 'author', 'author_username', 'content', 'created_at']
        read_only_fields = ['id', 'author', 'created_at']


class DiscussionSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    reply_count = serializers.IntegerField(read_only=True)
    latest_reply = serializers.SerializerMethodField()

    class Meta:
        model = Discussion
        fields = [
            'id', 'group', 'author', 'author_username', 'title', 'content',
            'is_pinned', 'is_closed', 'view_count', 'reply_count',
            'latest_reply', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'author', 'view_count', 'reply_count',
                            'created_at', 'updated_at']

    def get_latest_reply(self, obj):
        last = obj.replies.order_by('-created_at').first()
        if last:
            return {
                'author': last.author.username,
                'content': last.content[:100],
                'created_at': last.created_at,
            }
        return None


class DiscussionDetailSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source='author.username', read_only=True)
    replies = DiscussionReplySerializer(many=True, read_only=True)
    reply_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Discussion
        fields = [
            'id', 'group', 'author', 'author_username', 'title', 'content',
            'is_pinned', 'is_closed', 'view_count', 'reply_count',
            'replies', 'created_at', 'updated_at',
        ]
        read_only_fields = ['id', 'author', 'view_count', 'reply_count',
                            'replies', 'created_at', 'updated_at']


class ActivityFeedSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    avatar_url = serializers.SerializerMethodField()

    class Meta:
        model = ActivityFeed
        fields = [
            'id', 'user', 'username', 'avatar_url', 'activity_type',
            'title', 'description', 'link', 'metadata', 'created_at',
        ]
        read_only_fields = ['id', 'created_at']

    def get_avatar_url(self, obj):
        try:
            return obj.user.profile.avatar_url
        except:
            return None


class PublicationAlertSerializer(serializers.ModelSerializer):
    user_username = serializers.CharField(source='user.username', read_only=True)
    follower_username = serializers.CharField(source='follower.username', read_only=True)

    class Meta:
        model = PublicationAlert
        fields = [
            'id', 'user', 'follower', 'user_username', 'follower_username',
            'alert_type', 'content', 'link', 'is_read', 'created_at',
        ]
        read_only_fields = ['id', 'is_read', 'created_at']