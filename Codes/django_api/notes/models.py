# notes/models.py
import uuid
import logging
from django.db import models
from django.contrib.auth.models import User
# REMOVE: from django.contrib.postgres.fields import ArrayField
from django.utils import timezone
from django.urls import reverse
from jobscheduler.models import Job

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1. ResearcherNote – Main model with all features
# ----------------------------------------------------------------------
class ResearcherNote(models.Model):
    """
    Comprehensive research notebook entry with versioning, threading,
    mentions, reactions, attachments, tags, public/private, and soft delete.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # CHANGE related_name to avoid clash with a field named 'notes' in Job model
    run = models.ForeignKey(Job, on_delete=models.CASCADE, null=True, blank=True, related_name='researcher_notes')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notes')
    
    # Content
    title = models.CharField(max_length=200, blank=True)
    content = models.TextField()
    
    # Organisation
    # REPLACE ArrayField with JSONField
    tags = models.JSONField(default=list, blank=True)   # stores list of strings
    is_public = models.BooleanField(default=False, help_text="If public, visible to all users")
    pinned = models.BooleanField(default=False, help_text="Pin to top of notes list")
    
    # Versioning & threading
    version = models.IntegerField(default=1, help_text="Note version number (increments on edit)")
    parent_note = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='replies')
    
    # Mentions (explicit many-to-many, plus separate model for tracking)
    mentioned_users = models.ManyToManyField(User, blank=True, related_name='mentioned_in_notes')
    
    # Reactions (like, important, etc.)
    reactions = models.JSONField(default=dict, blank=True)
    
    # Attachments and images
    attachment = models.FileField(upload_to='note_attachments/', blank=True, null=True)
    image = models.ImageField(upload_to='note_images/', blank=True, null=True)
    
    # Metadata (can store snapshot of run metrics, hyperparameters, etc.)
    metadata = models.JSONField(default=dict, blank=True)
    
    # URL-friendly slug (auto-generated from title)
    slug = models.SlugField(max_length=255, blank=True, unique=False, help_text="Auto-generated from title")
    
    # Timestamps and soft delete
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True, blank=True, help_text="Soft delete timestamp")
    
    class Meta:
        ordering = ['-pinned', '-created_at']
        indexes = [
            models.Index(fields=['run']),
            models.Index(fields=['author']),
            models.Index(fields=['created_at']),
            models.Index(fields=['slug']),
            models.Index(fields=['is_public']),
            models.Index(fields=['pinned']),
            models.Index(fields=['deleted_at']),
            models.Index(fields=['parent_note']),
        ]
    
    def __str__(self):
        return f"Note by {self.author.username}: {self.title[:50] or 'Untitled'}"
    
    def get_absolute_url(self):
        return reverse('note_detail', kwargs={'note_id': self.id})
    
    def soft_delete(self):
        self.deleted_at = timezone.now()
        self.save(update_fields=['deleted_at'])
        # Trigger webhook event (if configured)
        from .webhook_utils import trigger_webhook_event
        trigger_webhook_event('note.deleted', self)
    
    def increment_version(self):
        self.version += 1
        self.save(update_fields=['version'])
    
    @property
    def is_deleted(self):
        return self.deleted_at is not None
    
    @property
    def reply_count(self):
        return self.replies.count()
    
    def save(self, *args, **kwargs):
        is_new = self.pk is None
        old_content = None
        if not is_new:
            try:
                old_content = ResearcherNote.objects.get(pk=self.pk).content
            except ResearcherNote.DoesNotExist:
                pass
        
        # Auto-generate slug from title if not provided
        if not self.slug and self.title:
            from django.utils.text import slugify
            base_slug = slugify(self.title)
            unique_slug = base_slug
            counter = 1
            while ResearcherNote.objects.filter(slug=unique_slug).exclude(pk=self.pk).exists():
                unique_slug = f"{base_slug}-{counter}"
                counter += 1
            self.slug = unique_slug
        
        super().save(*args, **kwargs)
        
        # If content changed, create a version record
        if not is_new and old_content is not None and old_content != self.content:
            NoteVersion.objects.create(
                note=self,
                version_number=self.version,          # was self.version - 1
                content_snapshot=old_content,
                changed_by=self.author,
                change_reason="auto"
            )
            self.increment_version()                # now version becomes self.version + 1
        
        # Trigger webhook events
        from .webhook_utils import trigger_webhook_event
        if is_new:
            trigger_webhook_event('note.created', self)
        elif old_content != self.content:
            trigger_webhook_event('note.updated', self)

# ----------------------------------------------------------------------
# 2. NoteVersion – Full version history (optional but recommended)
# ----------------------------------------------------------------------
class NoteVersion(models.Model):
    """Stores full snapshot of note content for version history."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(ResearcherNote, on_delete=models.CASCADE, related_name='versions')
    version_number = models.IntegerField()
    content_snapshot = models.TextField()
    title_snapshot = models.CharField(max_length=200, blank=True)
    changed_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    change_reason = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-version_number']
        unique_together = [['note', 'version_number']]
    
    def __str__(self):
        return f"Version {self.version_number} of {self.note.id}"

# ----------------------------------------------------------------------
# 3. NoteImage – Embedded images (with ordering)
# ----------------------------------------------------------------------
class NoteImage(models.Model):
    """Images embedded directly into note content (drag-and-drop)."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(ResearcherNote, on_delete=models.CASCADE, related_name='embedded_images')
    image = models.ImageField(upload_to='note_embedded/')
    caption = models.CharField(max_length=200, blank=True)
    order = models.PositiveIntegerField(default=0, help_text="Display order within the note")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['order', 'uploaded_at']
    
    def __str__(self):
        return f"Image for {self.note.id}: {self.image.name}"

# ----------------------------------------------------------------------
# 4. NoteMention – Explicit mention tracking (who was mentioned where)
# ----------------------------------------------------------------------
class NoteMention(models.Model):
    """Tracks when a user is mentioned in a note."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(ResearcherNote, on_delete=models.CASCADE, related_name='mentions')
    mentioned_user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='note_mentions')
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    
    class Meta:
        unique_together = [['note', 'mentioned_user']]
        indexes = [
            models.Index(fields=['mentioned_user', 'is_read']),
            models.Index(fields=['note']),
        ]
    
    def __str__(self):
        return f"Mention of {self.mentioned_user.username} in {self.note.id}"

# ----------------------------------------------------------------------
# 5. WebhookSubscription – For external integrations
# ----------------------------------------------------------------------
class WebhookSubscription(models.Model):
    """Store webhook endpoints that external services register."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='webhooks')
    url = models.URLField(max_length=500)
    # REPLACE ArrayField with JSONField
    events = models.JSONField(default=list, blank=True, help_text="List of event types to trigger on")
    secret = models.CharField(max_length=100, blank=True, help_text="HMAC secret for signing")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_triggered_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['user', 'is_active']),
        ]
    
    def __str__(self):
        return f"Webhook {self.url} for {self.user.username}"

# ----------------------------------------------------------------------
# 6. Notification – User notifications (mentions, replies, etc.)
# ----------------------------------------------------------------------
class Notification(models.Model):
    NOTIFICATION_TYPES = [
        ('mention', 'Mention'),
        ('reply', 'Reply to your note'),
        ('reaction', 'Reaction on your note'),
        ('webhook', 'Webhook delivery'),
        ('system', 'System notification'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    sender = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='sent_notifications')
    type = models.CharField(max_length=20, choices=NOTIFICATION_TYPES)
    note = models.ForeignKey(ResearcherNote, on_delete=models.CASCADE, null=True, blank=True)
    content = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['recipient', 'is_read']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"Notification to {self.recipient.username}: {self.type}"
