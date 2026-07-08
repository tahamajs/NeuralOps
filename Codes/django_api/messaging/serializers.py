# messaging/serializers.py
from rest_framework import serializers

from .models import ChatRoom, Message


class MessageSerializer(serializers.ModelSerializer):
    sender_username = serializers.CharField(source="sender.username", read_only=True)
    sender_id = serializers.IntegerField(source="sender.id", read_only=True)

    class Meta:
        model = Message
        fields = [
            "id",
            "room",
            "sender",
            "sender_id",
            "sender_username",
            "content",
            "file",
            "file_name",
            "read",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "sender",
            "sender_id",
            "sender_username",
            "read",
            "created_at",
        ]


class ChatRoomListSerializer(serializers.ModelSerializer):
    other_user = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()
    unread_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatRoom
        fields = ["id", "other_user", "last_message", "unread_count", "updated_at"]

    def get_other_user(self, obj):
        request = self.context.get("request")
        if not request:
            return None
        other = obj.participants.exclude(id=request.user.id).first()
        if not other:
            return None
        return {
            "id": other.id,
            "username": other.username,
            "avatar_url": getattr(getattr(other, "profile", None), "avatar_url", None),
            "display_name": getattr(
                getattr(other, "profile", None), "display_name", None
            )
            or other.username,
        }

    def get_last_message(self, obj):
        last = obj.last_message
        if not last:
            return None
        return {
            "content": last.content[:100],
            "sender_username": last.sender.username,
            "created_at": last.created_at,
        }

    def get_unread_count(self, obj):
        request = self.context.get("request")
        if not request:
            return 0
        return obj.unread_count(request.user)


class ChatRoomDetailSerializer(serializers.ModelSerializer):
    messages = serializers.SerializerMethodField()
    other_user = serializers.SerializerMethodField()

    class Meta:
        model = ChatRoom
        fields = ["id", "other_user", "messages", "created_at"]

    def get_other_user(self, obj):
        request = self.context.get("request")
        if not request:
            return None
        other = obj.participants.exclude(id=request.user.id).first()
        if not other:
            return None
        return {
            "id": other.id,
            "username": other.username,
        }

    def get_messages(self, obj):
        msgs = obj.messages.order_by("created_at")
        return MessageSerializer(msgs, many=True).data
