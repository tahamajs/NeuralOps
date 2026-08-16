import uuid

from django.conf import settings
from django.db import models


class CloudConnection(models.Model):
    """A user's connected external account."""
    PROVIDERS = [
        ('github', 'GitHub'),
        ('gitlab', 'GitLab'),
        ('s3', 'AWS S3'),
        ('gcs', 'Google Cloud Storage'),
        ('gdrive', 'Google Drive'),
        ('dropbox', 'Dropbox'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='cloud_connections')
    provider = models.CharField(max_length=20, choices=PROVIDERS)
    display_name = models.CharField(max_length=200, blank=True)
    access_token = models.TextField(blank=True)  # encrypted in production
    refresh_token = models.TextField(blank=True)
    token_expires_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['user', 'provider', 'display_name']]

class SyncTask(models.Model):
    """A file sync operation."""
    STATUSES = [('pending','Pending'),('running','Running'),('completed','Completed'),('failed','Failed')]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    connection = models.ForeignKey(CloudConnection, on_delete=models.CASCADE, related_name='sync_tasks')
    direction = models.CharField(max_length=10, choices=[('push','Push'),('pull','Pull')])
    source_path = models.CharField(max_length=500)
    target_path = models.CharField(max_length=500)
    status = models.CharField(max_length=20, choices=STATUSES, default='pending')
    error_message = models.TextField(blank=True)
    files_synced = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)