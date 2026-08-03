from rest_framework import serializers

from .models import (
    NoteComment,
    NoteImage,
    NoteVersion,
    Notification,
    ResearcherNote,
    WebhookSubscription,
)


class NoteVersionSerializer(serializers.ModelSerializer):
    changed_by_username = serializers.CharField(
        source="changed_by.username", read_only=True
    )

    class Meta:
        model = NoteVersion
        fields = [
            "id",
            "version_number",
            "title_snapshot",
            "content_snapshot",
            "changed_by_username",
            "change_reason",
            "created_at",
        ]


class NoteImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = NoteImage
        fields = "__all__"
        read_only_fields = ["id", "uploaded_at"]


class ResearcherNoteSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source="author.username", read_only=True)
    run_id = serializers.UUIDField(source="run.id", read_only=True, allow_null=True)
    run_name = serializers.CharField(source="run.name", read_only=True, allow_null=True)
    reply_count = serializers.IntegerField(read_only=True)
    version_count = serializers.SerializerMethodField()
    reading_time = serializers.IntegerField(
        source="reading_time_minutes", read_only=True
    )
    word_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = ResearcherNote
        fields = "__all__"
        read_only_fields = [
            "id",
            "author",
            "created_at",
            "updated_at",
            "slug",
            "version",
            "search_vector",
            "author_username",
            "run_id",
            "run_name",
            "reply_count",
            "version_count",
            "reading_time",
            "word_count",
        ]

    def get_version_count(self, obj):
        return obj.versions.count()


class NoteCommentSerializer(serializers.ModelSerializer):
    author_username = serializers.CharField(source="author.username", read_only=True)
    replies = serializers.SerializerMethodField()

    class Meta:
        model = NoteComment
        fields = [
            "id",
            "author_username",
            "content",
            "is_resolved",
            "replies",
            "created_at",
        ]

    def get_replies(self, obj):
        replies = obj.replies.all()
        return NoteCommentSerializer(replies, many=True).data


class WebhookSubscriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = WebhookSubscription
        fields = "__all__"
        read_only_fields = ["id", "created_at", "last_triggered_at", "user"]


class NotificationSerializer(serializers.ModelSerializer):
    sender_username = serializers.CharField(source="sender.username", read_only=True)

    class Meta:
        model = Notification
        fields = [
            "id",
            "type",
            "content",
            "sender_username",
            "note_id",
            "is_read",
            "created_at",
        ]
