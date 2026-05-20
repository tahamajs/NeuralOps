import uuid
import os
from django.db import models
from django.conf import settings

class TerminalSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='terminal_sessions')
    pid = models.IntegerField(null=True, blank=True)
    fd = models.IntegerField(null=True, blank=True)
    cols = models.IntegerField(default=80)
    rows = models.IntegerField(default=24)
    is_active = models.BooleanField(default=True)
    connected_at = models.DateTimeField(auto_now_add=True)
    last_activity_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-last_activity_at']
        indexes = [
            models.Index(fields=['user', 'is_active']),
        ]

    def __str__(self):
        return f'Terminal {self.id.hex[:8]} (user={self.user.username}, active={self.is_active})'

    def mark_inactive(self):
        self.is_active = False
        self.save(update_fields=['is_active'])
        if self.fd:
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.fd = None
        self.pid = None
        self.save(update_fields=['fd', 'pid'])