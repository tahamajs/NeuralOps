# notes/models.py
import uuid
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField
from jobscheduler.models import Job  # or TrainingRun – adjust import

class ResearcherNote(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(Job, on_delete=models.CASCADE, null=True, blank=True, related_name='notes')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notes')
    title = models.CharField(max_length=200, blank=True)
    content = models.TextField()  # Markdown content
    tags = ArrayField(models.CharField(max_length=50), blank=True, default=list)
    reactions = models.JSONField(default=dict, blank=True)  # e.g., {"like": ["user1"], "important": ["user2"]}
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['run']),
            models.Index(fields=['author']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"Note by {self.author.username} on {self.created_at.date()}"