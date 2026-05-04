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
    attachment = models.FileField(upload_to='note_attachments/', blank=True, null=True)
    image = models.ImageField(upload_to='note_images/', blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['run']),
            models.Index(fields=['author']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"Note by {self.author.username} on {self.created_at.date()}"

class NoteImage(models.Model):
    note = models.ForeignKey(ResearcherNote, on_delete=models.CASCADE, related_name='embedded_images')
    image = models.ImageField(upload_to='note_embedded/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


# jobscheduler/models.py

class MetricType(models.TextChoices):
    SCALAR = 'scalar'
    IMAGE = 'image'
    TABLE = 'table'
    FIGURE = 'figure'
    TEXT = 'text'

class CustomMetricEntry(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    run = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='custom_metrics')
    
    # Core fields
    step = models.IntegerField()
    epoch = models.FloatField(null=True, blank=True)
    name = models.CharField(max_length=100)  # e.g., 'train_loss', 'accuracy_per_class'
    type = models.CharField(max_length=20, choices=MetricType.choices, default=MetricType.SCALAR)
    
    # The actual data stored as flexible JSON
    value = models.JSONField()  # For scalars, it's a number; for images, it's a URL; for tables, it's a list of dicts
    
    # For images and figures
    media_url = models.URLField(max_length=500, blank=True, null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['run', 'name', 'step']),
            models.Index(fields=['run', 'type']),
        ]