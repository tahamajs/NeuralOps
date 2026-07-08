import uuid

from django.contrib.auth import get_user_model
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models

User = get_user_model()


class ActivityLog(models.Model):
    ACTION_TYPES = [
        ("create", "Created"),
        ("update", "Updated"),
        ("delete", "Deleted"),
        ("start", "Started"),
        ("stop", "Stopped"),
        ("complete", "Completed"),
        ("fail", "Failed"),
        ("publish", "Published"),
        ("share", "Shared"),
        ("comment", "Commented"),
        ("react", "Reacted"),
        ("export", "Exported"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    action_type = models.CharField(max_length=20, choices=ACTION_TYPES)
    description = models.TextField()
    content_type = models.ForeignKey(
        ContentType, on_delete=models.CASCADE, null=True, blank=True
    )
    object_id = models.CharField(max_length=36, null=True, blank=True)
    content_object = GenericForeignKey("content_type", "object_id")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["action_type", "created_at"]),
        ]

    def __str__(self):
        return f"[{self.action_type}] {self.user} – {self.description[:80]}"


def log_activity(
    user, action_type: str, description: str, content_object=None, metadata=None
):
    entry = ActivityLog(
        user=user,
        action_type=action_type,
        description=description,
        metadata=metadata or {},
    )
    if content_object:
        entry.content_type = ContentType.objects.get_for_model(content_object)
        entry.object_id = str(content_object.pk)
    entry.save()
    return entry
