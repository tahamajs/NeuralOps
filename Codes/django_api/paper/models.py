# paper/models.py
import uuid, os
from django.db import models
from django.conf import settings
from django.utils import timezone

def paper_file_upload_to(instance, filename):
    return f'paper_files/{instance.project.id}/{filename}'

class PaperProject(models.Model):
    COMPILER_CHOICES = [
        ('pdflatex', 'pdfLaTeX'),
        ('xelatex', 'XeLaTeX'),
        ('lualatex', 'LuaLaTeX'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='paper_projects')
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    root_file = models.CharField(max_length=255, default='main.tex')
    compiler = models.CharField(max_length=20, choices=COMPILER_CHOICES, default='pdflatex')
    draft_mode = models.BooleanField(default=False)
    stop_on_first_error = models.BooleanField(default=False)
    version = models.PositiveIntegerField(default=1)
    collaborators = models.ManyToManyField(settings.AUTH_USER_MODEL, blank=True, related_name='shared_paper_projects')
    git_remote = models.URLField(blank=True, default='')
    git_branch = models.CharField(max_length=100, blank=True, default='main')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_compiled_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        app_label = 'paper'
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['owner']),
            models.Index(fields=['updated_at']),
        ]

    def __str__(self): return self.name

    @property
    def file_count(self): return self.files.count()

    @property
    def last_compile(self): return self.compiles.order_by('-created_at').first()


class PaperFile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(PaperProject, on_delete=models.CASCADE, related_name='files')
    filename = models.CharField(max_length=500)
    content = models.TextField(blank=True)
    binary = models.FileField(upload_to=paper_file_upload_to, blank=True, null=True)
    is_binary = models.BooleanField(default=False)
    version = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'paper'
        unique_together = [['project', 'filename']]
        ordering = ['filename']

    def __str__(self): return f"{self.project.name}/{self.filename}"

    @property
    def extension(self): return self.filename.rsplit('.', 1)[-1].lower() if '.' in self.filename else ''

    @property
    def is_tex(self): return self.extension == 'tex'

    @property
    def is_image(self): return self.extension in ('png', 'jpg', 'jpeg', 'pdf', 'eps', 'svg')

    @property
    def is_bib(self): return self.extension == 'bib'


class PaperCompile(models.Model):
    STATUS_CHOICES = [
        ('queued', 'Queued'), ('running', 'Running'), ('completed', 'Completed'),
        ('failed', 'Failed'), ('timed_out', 'Timed Out'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(PaperProject, on_delete=models.CASCADE, related_name='compiles')
    initiated_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    compiler = models.CharField(max_length=20, default='pdflatex')
    draft_mode = models.BooleanField(default=False)
    root_file = models.CharField(max_length=255, default='main.tex')
    pdf = models.FileField(upload_to='compiled_pdfs/', blank=True, null=True)
    log = models.TextField(blank=True)
    error_count = models.IntegerField(default=0)
    warning_count = models.IntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'paper'
        ordering = ['-created_at']

    @property
    def duration(self):
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    def mark_running(self):
        self.status = 'running'
        self.started_at = timezone.now()
        self.save(update_fields=['status', 'started_at'])

    def mark_completed(self, pdf_path, log_text, errors, warnings):
        self.status = 'completed'
        self.finished_at = timezone.now()
        self.error_count = errors
        self.warning_count = warnings
        self.log = log_text
        if pdf_path:
            self.pdf.save(f'{self.id.hex}.pdf', open(pdf_path, 'rb'), save=False)
        self.save()

    def mark_failed(self, log_text, errors, warnings):
        self.status = 'failed'
        self.finished_at = timezone.now()
        self.error_count = errors
        self.warning_count = warnings
        self.log = log_text
        self.save()


class PaperTemplate(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    category = models.CharField(max_length=100, blank=True, default='general')
    files = models.JSONField(default=dict, help_text="Dict of filename -> content")
    is_public = models.BooleanField(default=True)
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    usage_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'paper'
        ordering = ['-usage_count', 'name']

    def __str__(self): return self.name


class PaperShareLink(models.Model):
    """Public share link for a paper project – view-only PDF + source."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(PaperProject, on_delete=models.CASCADE, related_name='share_links')
    token = models.CharField(max_length=64, unique=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    password = models.CharField(max_length=128, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    view_count = models.IntegerField(default=0)
    max_views = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'paper'
        ordering = ['-created_at']

    @property
    def is_expired(self):
        if self.expires_at and timezone.now() > self.expires_at: return True
        if self.max_views and self.view_count >= self.max_views: return True
        return False

    @property
    def public_url(self):
        base = getattr(settings, 'PUBLIC_BASE_URL', 'http://localhost:8000')
        return f"{base}/paper/shared/{self.token}/"


class PaperComment(models.Model):
    """Inline comments on specific files/lines."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(PaperProject, on_delete=models.CASCADE, related_name='comments')
    file = models.ForeignKey(PaperFile, on_delete=models.CASCADE, null=True, blank=True, related_name='comments')
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    line_start = models.IntegerField(null=True, blank=True)
    line_end = models.IntegerField(null=True, blank=True)
    content = models.TextField()
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = 'paper'
        ordering = ['created_at']


class PaperVersion(models.Model):
    """Snapshot of entire project at a point in time."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(PaperProject, on_delete=models.CASCADE, related_name='snapshots')
    version_number = models.IntegerField()
    files_snapshot = models.JSONField(default=dict)  # filename -> content
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    message = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        app_label = 'paper'
        unique_together = [['project', 'version_number']]
        ordering = ['-version_number']


class CollaborationSession(models.Model):
    """An active co‑authoring session on a paper project."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(PaperProject, on_delete=models.CASCADE, related_name='collab_sessions')
    started_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    active_users = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name='active_collab_sessions')
    created_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)

class CursorPosition(models.Model):
    """Real‑time cursor position of a collaborator."""
    session = models.ForeignKey(CollaborationSession, on_delete=models.CASCADE, related_name='cursors')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    file = models.ForeignKey(PaperFile, on_delete=models.CASCADE)
    line = models.IntegerField()
    column = models.IntegerField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta: unique_together = [['session', 'user', 'file']]