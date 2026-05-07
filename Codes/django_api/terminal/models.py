import uuid
from django.db import models
from django.conf import settings


class TerminalSession(models.Model):
    """
    Represents an active terminal session.

    Created when a user connects via WebSocket, deleted on disconnect.
    Used for monitoring, resource limiting, and admin oversight.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='terminal_sessions',
    )
    channel_name = models.CharField(max_length=255, unique=True)

    # Process info
    pid = models.IntegerField(null=True, blank=True)
    fd = models.IntegerField(null=True, blank=True)

    # Terminal dimensions (cols × rows)
    cols = models.IntegerField(default=80)
    rows = models.IntegerField(default=24)

    # Status
    is_active = models.BooleanField(default=True)

    # Timestamps
    connected_at = models.DateTimeField(auto_now_add=True)
    last_activity_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-connected_at']
        indexes = [
            models.Index(fields=['user']),
            models.Index(fields=['is_active']),
            models.Index(fields=['last_activity_at']),
        ]

    def __str__(self):
        return f"Session {self.id.hex[:8]} - {self.user.username}"

    def mark_inactive(self):
        self.is_active = False
        self.save(update_fields=['is_active'])