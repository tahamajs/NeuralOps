# messaging/models.py
import uuid

from django.conf import settings
from django.db import models


class ChatRoom(models.Model):
    """A private conversation between two users."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    participants = models.ManyToManyField(
        settings.AUTH_USER_MODEL, related_name="chat_rooms"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        names = [u.username for u in self.participants.all()]
        return f"Chat: {' & '.join(names)}"

    @property
    def last_message(self):
        return self.messages.order_by("-created_at").first()

    @property
    def unread_count(self, user):
        return self.messages.filter(read=False).exclude(sender=user).count()


class Message(models.Model):
    """A single message inside a chat room."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    room = models.ForeignKey(
        ChatRoom, on_delete=models.CASCADE, related_name="messages"
    )
    sender = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="sent_messages"
    )
    content = models.TextField(blank=True)
    file = models.FileField(upload_to="chat_files/", blank=True, null=True)
    file_name = models.CharField(max_length=300, blank=True)
    read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["room", "-created_at"]),
            models.Index(fields=["sender", "read"]),
        ]

    def __str__(self):
        return f"{self.sender.username}: {self.content[:50]}"
