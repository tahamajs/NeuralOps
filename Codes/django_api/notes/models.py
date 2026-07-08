# notes/models.py
"""
Complete Research Notebook System for ML Experiment Tracking.
Supports auto-generated experiment reports, rich Markdown notes,
version history, collaborative editing, mentions, notifications,
full-text search, and webhook integrations.
"""

import logging
import uuid

from django.contrib.auth.models import User
from django.db import models
from django.urls import reverse
from django.utils import timezone
from django.utils.text import slugify
from jobscheduler.models import Job

logger = logging.getLogger(__name__)


class ResearcherNote(models.Model):
    """
    Comprehensive ML research notebook entry.
    Inspired by W&B Reports, MLflow notes, and Benchling ELN.
    """

    NOTE_TYPES = [
        ("general", "General Note"),
        ("experiment_log", "Experiment Log"),
        ("training_report", "Training Report"),
        ("compression_report", "Compression Report"),
        ("literature_review", "Literature Review"),
        ("hypothesis", "Hypothesis / Idea"),
        ("meeting_notes", "Meeting Notes"),
        ("code_snippet", "Code Snippet"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # ── Relationships ──────────────────────────────────────────
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name="notes")
    run = models.ForeignKey(
        Job,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="researcher_notes",
        help_text="Training run this note is linked to",
    )
    linked_runs = models.ManyToManyField(
        Job,
        blank=True,
        related_name="linked_notes",
        help_text="Additional runs referenced in this note",
    )
    parent_note = models.ForeignKey(
        "self", on_delete=models.CASCADE, null=True, blank=True, related_name="replies"
    )

    # ── Content ────────────────────────────────────────────────
    note_type = models.CharField(
        max_length=30,
        choices=NOTE_TYPES,
        default="general",
        help_text="Type of research note",
    )
    title = models.CharField(max_length=300, blank=True)
    content = models.TextField(
        blank=True, help_text="Markdown content with LaTeX support"
    )

    # ── Metadata ───────────────────────────────────────────────
    tags = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="""
        Flexible metadata: {
            "hyperparameters": {...},
            "metrics_snapshot": {...},
            "model_version": "v3",
            "dataset": "glue-mrpc",
            "git_commit": "abc123",
            "linked_artifacts": ["/path/to/model.pth"],
            "charts": [{"name": "loss_curve", "data": [...]}]
        }
    """,
    )

    # ── Visibility ─────────────────────────────────────────────
    is_public = models.BooleanField(default=False)
    is_pinned = models.BooleanField(default=False)
    is_favorite = models.BooleanField(default=False)
    shared_with = models.ManyToManyField(User, blank=True, related_name="shared_notes")

    # ── Reactions ──────────────────────────────────────────────
    reactions = models.JSONField(default=dict, blank=True)

    # ── Attachments ────────────────────────────────────────────
    attachment = models.FileField(upload_to="note_attachments/", blank=True, null=True)
    image = models.ImageField(upload_to="note_images/", blank=True, null=True)

    # ── Versioning ─────────────────────────────────────────────
    version = models.IntegerField(default=1)
    slug = models.SlugField(max_length=300, blank=True)

    # ── Full-Text Search ───────────────────────────────────────
    # search_vector = SearchVectorField(null=True, blank=True)

    # ── Timestamps ─────────────────────────────────────────────
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-is_pinned", "-created_at"]
        indexes = [
            models.Index(fields=["run"]),
            models.Index(fields=["author"]),
            models.Index(fields=["note_type"]),
            models.Index(fields=["created_at"]),
            models.Index(fields=["is_public"]),
            models.Index(fields=["is_pinned"]),
            models.Index(fields=["is_favorite"]),
            models.Index(fields=["deleted_at"]),
            models.Index(fields=["slug"]),
        ]

    def __str__(self):
        return f"[{self.note_type}] {self.title[:60] or 'Untitled'} by {self.author.username}"

    def get_absolute_url(self):
        return reverse("note_detail", kwargs={"note_id": self.id})

    def soft_delete(self):
        self.deleted_at = timezone.now()
        self.save(update_fields=["deleted_at"])

    def toggle_favorite(self):
        self.is_favorite = not self.is_favorite
        self.save(update_fields=["is_favorite"])

    def toggle_pin(self):
        self.is_pinned = not self.is_pinned
        self.save(update_fields=["is_pinned"])

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
                old = ResearcherNote.objects.only("content").get(pk=self.pk)
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
                title_snapshot=getattr(self, "_old_title", self.title),
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
    note = models.ForeignKey(
        ResearcherNote, on_delete=models.CASCADE, related_name="versions"
    )
    version_number = models.IntegerField()
    content_snapshot = models.TextField()
    title_snapshot = models.CharField(max_length=300, blank=True)
    changed_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True
    )
    change_reason = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-version_number"]
        unique_together = [["note", "version_number"]]

    def __str__(self):
        return f"v{self.version_number} of {self.note_id}"


class NoteImage(models.Model):
    """Embedded images in notes."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(
        ResearcherNote, on_delete=models.CASCADE, related_name="embedded_images"
    )
    image = models.ImageField(upload_to="note_embedded/")
    caption = models.CharField(max_length=300, blank=True)
    order = models.PositiveIntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["order", "uploaded_at"]


class NoteMention(models.Model):
    """User mentioned in a note."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(
        ResearcherNote, on_delete=models.CASCADE, related_name="mentions"
    )
    mentioned_user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="note_mentions"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    class Meta:
        unique_together = [["note", "mentioned_user"]]
        indexes = [
            models.Index(fields=["mentioned_user", "is_read"]),
        ]


class NoteComment(models.Model):
    """Threaded comments on notes."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(
        ResearcherNote, on_delete=models.CASCADE, related_name="comments"
    )
    author = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="note_comments"
    )
    parent_comment = models.ForeignKey(
        "self", on_delete=models.CASCADE, null=True, blank=True, related_name="replies"
    )
    content = models.TextField()
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at"]


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
        default=dict,
        blank=True,
        help_text='{"model_name": "string", "task_name": "string", ...}',
    )
    is_default = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["note_type", "name"]

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
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="webhooks")
    url = models.URLField(max_length=500)
    events = models.JSONField(default=list, blank=True)
    secret = models.CharField(max_length=100, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_triggered_at = models.DateTimeField(null=True, blank=True)


class Notification(models.Model):
    """User notifications system."""

    TYPES = [
        ("mention", "Mention"),
        ("reply", "Reply"),
        ("reaction", "Reaction"),
        ("comment", "Comment"),
        ("share", "Shared with you"),
        ("run_complete", "Training Complete"),
        ("system", "System"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    recipient = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="notifications"
    )
    sender = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="sent_notifications",
    )
    type = models.CharField(max_length=20, choices=TYPES)
    note = models.ForeignKey(
        ResearcherNote, on_delete=models.CASCADE, null=True, blank=True
    )
    content = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["recipient", "is_read"]),
        ]


# notes/models.py – add these below the existing classes


class Workspace(models.Model):
    """Collaborative workspace to group notes, runs, and experiments."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="workspaces"
    )
    members = models.ManyToManyField(User, blank=True, related_name="joined_workspaces")
    notes = models.ManyToManyField(
        "ResearcherNote", blank=True, related_name="workspaces"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_archived = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class Task(models.Model):
    """Actionable task linked to a note."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("in_progress", "In Progress"),
        ("done", "Done"),
        ("blocked", "Blocked"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(
        "ResearcherNote", on_delete=models.CASCADE, related_name="tasks"
    )
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    assigned_to = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="assigned_tasks",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    due_date = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["due_date", "-created_at"]


class ModelRegistry(models.Model):
    """Central registry for trained models and their stages."""

    STAGE_CHOICES = [
        ("development", "Development"),
        ("staging", "Staging"),
        ("production", "Production"),
        ("archived", "Archived"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    version = models.CharField(max_length=50)
    run = models.ForeignKey(
        "jobscheduler.Job",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="registered_models",
    )
    note = models.ForeignKey(
        "ResearcherNote",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="registered_models",
    )
    stage = models.CharField(
        max_length=20, choices=STAGE_CHOICES, default="development"
    )
    description = models.TextField(blank=True)
    file = models.FileField(upload_to="models/", blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="registered_models"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [["name", "version"]]
        ordering = ["-created_at"]


class HyperparameterSet(models.Model):
    """Snapshot of hyperparameters for a training run."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(
        "ResearcherNote", on_delete=models.CASCADE, related_name="hyperparams"
    )
    run = models.ForeignKey(
        "jobscheduler.Job", on_delete=models.CASCADE, related_name="hyperparams"
    )
    params = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)


class Artifact(models.Model):
    """Any file attached to a note or run."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(
        "ResearcherNote",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="artifacts",
    )
    run = models.ForeignKey(
        "jobscheduler.Job",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="artifacts",
    )
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to="artifacts/")
    file_type = models.CharField(max_length=50, blank=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class NoteLink(models.Model):
    """Bidirectional link between notes (knowledge graph)."""

    source = models.ForeignKey(
        "ResearcherNote", on_delete=models.CASCADE, related_name="outgoing_links"
    )
    target = models.ForeignKey(
        "ResearcherNote", on_delete=models.CASCADE, related_name="incoming_links"
    )
    link_type = models.CharField(max_length=50, default="related")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["source", "target"]]


class PDFAnnotation(models.Model):
    """Annotations on uploaded PDFs."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(
        "ResearcherNote", on_delete=models.CASCADE, related_name="pdf_annotations"
    )
    pdf_file = models.FileField(upload_to="pdfs/")
    page_number = models.IntegerField()
    highlight_text = models.TextField(blank=True)
    comment = models.TextField()
    x1 = models.FloatField()
    y1 = models.FloatField()
    x2 = models.FloatField()
    y2 = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)


class Protocol(models.Model):
    """Reusable experiment protocol."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    steps = models.JSONField(default=list)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="protocols"
    )
    created_at = models.DateTimeField(auto_now_add=True)


# notes/models.py — add these at the bottom of the file

import secrets


class ShareableLink(models.Model):
    """
    Public share link for a note. Anyone with the token can view the note.
    Supports expiration, password protection, and view counting.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(
        "ResearcherNote", on_delete=models.CASCADE, related_name="share_links"
    )
    token = models.CharField(max_length=64, unique=True, default=secrets.token_urlsafe)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="created_share_links"
    )
    password = models.CharField(
        max_length=128, blank=True, help_text="Optional password to access this link"
    )
    expires_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    view_count = models.IntegerField(default=0)
    max_views = models.IntegerField(
        null=True, blank=True, help_text="Max number of views before link expires"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    last_accessed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Share link for note {self.note_id} ({'active' if self.is_active else 'inactive'})"

    @property
    def is_expired(self):
        if self.expires_at and timezone.now() > self.expires_at:
            return True
        if self.max_views and self.view_count >= self.max_views:
            return True
        return False

    @property
    def public_url(self):
        from django.conf import settings

        base = getattr(settings, "PUBLIC_BASE_URL", "http://localhost:8000")
        return f"{base}/notes/shared/{self.token}/"

    def record_view(self):
        self.view_count += 1
        self.last_accessed_at = timezone.now()
        self.save(update_fields=["view_count", "last_accessed_at"])


class Artifact(models.Model):
    """File artifacts (checkpoints, plots, logs) associated with a note or run."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    note = models.ForeignKey(
        "ResearcherNote",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="artifacts",
    )
    run = models.ForeignKey(
        "jobscheduler.Job",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="artifacts",
    )
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to="artifacts/")
    file_type = models.CharField(max_length=50, blank=True)
    file_size_mb = models.FloatField(default=0.0)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class Protocol(models.Model):
    """Executable experiment protocol."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    steps = models.JSONField(default=list)
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="protocols"
    )
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class NoteLink(models.Model):
    """Bidirectional link between notes (knowledge graph)."""

    source = models.ForeignKey(
        "ResearcherNote", on_delete=models.CASCADE, related_name="outgoing_links"
    )
    target = models.ForeignKey(
        "ResearcherNote", on_delete=models.CASCADE, related_name="incoming_links"
    )
    link_type = models.CharField(max_length=50, default="related")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["source", "target"]]

    def __str__(self):
        return f"{self.source_id} → {self.target_id} ({self.link_type})"


# notes/models.py — add this


class DataApp(models.Model):
    """An interactive web app built from a notebook or script."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    source_notebook = models.FileField(upload_to="data_apps/", null=True, blank=True)
    source_script = models.TextField(blank=True)
    generated_html = models.TextField(blank=True)
    hugginface_space_url = models.URLField(blank=True)
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
