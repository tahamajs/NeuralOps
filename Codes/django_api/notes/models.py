# notes/models.py
"""
Complete Research Notebook System for ML Experiment Tracking.
Supports auto-generated experiment reports, rich Markdown notes,
version history, collaborative editing, mentions, notifications,
full-text search, and webhook integrations.
"""
import uuid
import logging
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.utils.text import slugify
from django.urls import reverse
from django.contrib.postgres.search import SearchVectorField, SearchVector
from jobscheduler.models import Job

logger = logging.getLogger(__name__)


class ResearcherNote(models.Model):
    """
    Comprehensive ML research notebook entry.
    Inspired by W&B Reports, MLflow notes, and Benchling ELN.
    """
    NOTE_TYPES = [
        ('general', 'General Note'),
        ('experiment_log', 'Experiment Log'),
        ('training_report', 'Training Report'),
        ('compression_report', 'Compression Report'),
        ('literature_review', 'Literature Review'),
        ('hypothesis', 'Hypothesis / Idea'),
        ('meeting_notes', 'Meeting Notes'),
        ('code_snippet', 'Code Snippet'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # ── Relationships ──────────────────────────────────────────
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notes')
    run = models.ForeignKey(
        Job, on_delete=models.SET_NULL, null=True, blank=True,
        related_name='researcher_notes',
        help_text="Training run this note is linked to"
    )
    linked_runs = models.ManyToManyField(
        Job, blank=True, related_name='linked_notes',
        help_text="Additional runs referenced in this note"
    )
    parent_note = models.ForeignKey(
        'self', on_delete=models.CASCADE, null=True, blank=True,
        related_name='replies'
    )
    
    # ── Content ────────────────────────────────────────────────
    note_type = models.CharField(
        max_length=30, choices=NOTE_TYPES, default='general',
        help_text="Type of research note"
    )
    title = models.CharField(max_length=300, blank=True)
    content = models.TextField(blank=True, help_text="Markdown content with LaTeX support")
    
    # ── Metadata ───────────────────────────────────────────────
    tags = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(default=dict, blank=True, help_text="""
        Flexible metadata: {
            "hyperparameters": {...},
            "metrics_snapshot": {...},
            "model_version": "v3",
            "dataset": "glue-mrpc",
            "git_commit": "abc123",
            "linked_artifacts": ["/path/to/model.pth"],
            "charts": [{"name": "loss_curve", "data": [...]}]
        }
    """)
    
    # ── Visibility ─────────────────────────────────────────────
    is_public = models.BooleanField(default=False)
    is_pinned = models.BooleanField(default=False)
    is_favorite = models.BooleanField(default=False)
    shared_with = models.ManyToManyField(User, blank=True, related_name='shared_notes')
    
    # ── Reactions ──────────────────────────────────────────────
    reactions = models.JSONField(default=dict, blank=True)
    
    # ── Attachments ────────────────────────────────────────────
    attachment = models.FileField(upload_to='note_attachments/', blank=True, null=True)
    image = models.ImageField(upload_to='note_images/', blank=True, null=True)
    
    # ── Versioning ─────────────────────────────────────────────
    version = models.IntegerField(default=1)
    slug = models.SlugField(max_length=300, blank=True)
    
    # ── Full-Text Search ───────────────────────────────────────
    search_vector = SearchVectorField(null=True, blank=True)
    
    # ── Timestamps ─────────────────────────────────────────────
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-is_pinned', '-created_at']
        indexes = [
            models.Index(fields=['run']),
            models.Index(fields=['author']),
            models.Index(fields=['note_type']),
            models.Index(fields=['created_at']),
            models.Index(fields=['is_public']),
            models.Index(fields=['is_pinned']),
            models.Index(fields=['is_favorite']),
            models.Index(fields=['deleted_at']),
            models.Index(fields=['slug']),
        ]
    
    def __str__(self):
        return f"[{self.note_type}] {self.title[:60] or 'Untitled'} by {self.author.username}"
    
    def get_absolute_url(self):
        return reverse('note_detail', kwargs={'note_id': self.id})
    
    def soft_delete(self):
        self.deleted_at = timezone.now()
        self.save(update_fields=['deleted_at'])
    
    def toggle_favorite(self):
        self.is_favorite = not self.is_favorite
        self.save(update_fields=['is_favorite'])
    
    def toggle_pin(self):
        self.is_pinned = not self.is_pinned
        self.save(update_fields=['is_pinned'])
    
    @property
    def reply_count(self):
        return self.replies.filter(deleted_at__isnull=True).count()
    
    @property
    def word_count(self):
        return len(self.content.split()) if self.content else 0
    
    @property
    def reading_time_minutes(self):
        words = self.word_count
        return max(1, round(words / 200))  # 200 wpm
    
    def save(self, *args, **kwargs):
        is_new = self.pk is None
        old_content = None
        
        if not is_new:
            try:
                old = ResearcherNote.objects.only('content').get(pk=self.pk)
                old_content = old.content
            except ResearcherNote.DoesNotExist:
                pass
        
        # Auto-slug
        if not self.slug and self.title:
            self.slug = self._unique_slug()
        
        super().save(*args, **kwargs)
        
        # Version tracking
        if not is_new and old_content is not None and old_content != self.content:
            NoteVersion.objects.create(
                note=self,
                version_number=self.version,
                content_snapshot=old_content,
                title_snapshot=getattr(self, '_old_title', self.title),
                changed_by=self.author,
            )
            self.version += 1
            # Save again to update version (avoid recursion)
            ResearcherNote.objects.filter(pk=self.pk).update(version=self.version)
    
    def _unique_slug(self):
        base = slugify(self.title)[:250]
        unique = base
        counter = 1
        while ResearcherNote.objects.filter(slug=unique).exclude(pk=self.pk).exists():
            unique = f"{base}-{counter}"
            counter += 1
        return unique


class NoteVersion(models.Model):
    """Full snapshot of a note at a point in time."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(ResearcherNote, on_delete=models.CASCADE, related_name='versions')
    version_number = models.IntegerField()
    content_snapshot = models.TextField()
    title_snapshot = models.CharField(max_length=300, blank=True)
    changed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    change_reason = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-version_number']
        unique_together = [['note', 'version_number']]
    
    def __str__(self):
        return f"v{self.version_number} of {self.note_id}"


class NoteImage(models.Model):
    """Embedded images in notes."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(ResearcherNote, on_delete=models.CASCADE, related_name='embedded_images')
    image = models.ImageField(upload_to='note_embedded/')
    caption = models.CharField(max_length=300, blank=True)
    order = models.PositiveIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['order', 'uploaded_at']


class NoteMention(models.Model):
    """User mentioned in a note."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(ResearcherNote, on_delete=models.CASCADE, related_name='mentions')
    mentioned_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='note_mentions')
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    
    class Meta:
        unique_together = [['note', 'mentioned_user']]
        indexes = [
            models.Index(fields=['mentioned_user', 'is_read']),
        ]


class NoteComment(models.Model):
    """Threaded comments on notes."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(ResearcherNote, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='note_comments')
    parent_comment = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='replies')
    content = models.TextField()
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['created_at']


class NoteTemplate(models.Model):
    """Pre-defined templates for different types of research notes."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    note_type = models.CharField(max_length=30, choices=ResearcherNote.NOTE_TYPES)
    content_template = models.TextField(
        help_text="Markdown template with {{variable}} placeholders"
    )
    variables_schema = models.JSONField(
        default=dict, blank=True,
        help_text='{"model_name": "string", "task_name": "string", ...}'
    )
    is_default = models.BooleanField(default=False)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['note_type', 'name']
    
    def __str__(self):
        return f"[{self.note_type}] {self.name}"
    
    def render(self, variables: dict) -> str:
        """Render template with given variables."""
        content = self.content_template
        for key, value in variables.items():
            content = content.replace(f"{{{{{key}}}}}", str(value))
        return content


class WebhookSubscription(models.Model):
    """External webhook endpoints."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='webhooks')
    url = models.URLField(max_length=500)
    events = models.JSONField(default=list, blank=True)
    secret = models.CharField(max_length=100, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_triggered_at = models.DateTimeField(null=True, blank=True)


class Notification(models.Model):
    """User notifications system."""
    TYPES = [
        ('mention', 'Mention'),
        ('reply', 'Reply'),
        ('reaction', 'Reaction'),
        ('comment', 'Comment'),
        ('share', 'Shared with you'),
        ('run_complete', 'Training Complete'),
        ('system', 'System'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    sender = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='sent_notifications')
    type = models.CharField(max_length=20, choices=TYPES)
    note = models.ForeignKey(ResearcherNote, on_delete=models.CASCADE, null=True, blank=True)
    content = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['recipient', 'is_read']),
        ]