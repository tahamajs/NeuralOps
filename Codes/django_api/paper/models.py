import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone


class PaperProject(models.Model):
    """A LaTeX project – analogous to an Overleaf project."""

    COMPILER_CHOICES = [
        ('pdflatex', 'pdfLaTeX'),
        ('xelatex', 'XeLaTeX'),
        ('lualatex', 'LuaLaTeX'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='paper_projects',
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)

    # Which file is the root document for compilation
    root_file = models.CharField(
        max_length=255, default='main.tex',
        help_text="Path to the main .tex file, e.g. 'main.tex'"
    )

    # Compilation preferences
    compiler = models.CharField(
        max_length=20, choices=COMPILER_CHOICES, default='pdflatex'
    )
    draft_mode = models.BooleanField(default=False)
    stop_on_first_error = models.BooleanField(default=False)

    # Versioning
    version = models.PositiveIntegerField(default=1)

    # Sharing (like Overleaf's collaborator system)
    collaborators = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        blank=True,
        related_name='shared_paper_projects',
    )

    # Git integration (store remote URL)
    git_remote = models.URLField(blank=True, default='')
    git_branch = models.CharField(max_length=100, blank=True, default='main')

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_compiled_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['owner']),
            models.Index(fields=['updated_at']),
        ]

    def __str__(self):
        return self.name

    @property
    def file_count(self):
        return self.files.count()

    @property
    def last_compile(self):
        return self.compiles.order_by('-created_at').first()


class PaperFile(models.Model):
    """A single file inside a LaTeX project."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        PaperProject, on_delete=models.CASCADE, related_name='files'
    )
    filename = models.CharField(max_length=500)
    content = models.TextField(blank=True)

    # For binary files (images, etc.)
    binary = models.FileField(upload_to='paper_files/', blank=True, null=True)
    is_binary = models.BooleanField(default=False)

    # Version tracking
    version = models.PositiveIntegerField(default=1)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['project', 'filename']]
        ordering = ['filename']
        indexes = [
            models.Index(fields=['project', 'filename']),
        ]

    def __str__(self):
        return f"{self.project.name}/{self.filename}"

    @property
    def extension(self):
        return self.filename.rsplit('.', 1)[-1].lower() if '.' in self.filename else ''

    @property
    def is_tex(self):
        return self.extension == 'tex'

    @property
    def is_image(self):
        return self.extension in ('png', 'jpg', 'jpeg', 'pdf', 'eps', 'svg')

    @property
    def is_bib(self):
        return self.extension == 'bib'


class PaperCompile(models.Model):
    """A compilation record – stores status, output, and logs."""

    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('timed_out', 'Timed Out'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        PaperProject, on_delete=models.CASCADE, related_name='compiles'
    )
    initiated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')

    # Compilation options at time of compile
    compiler = models.CharField(max_length=20, default='pdflatex')
    draft_mode = models.BooleanField(default=False)

    # Input: snapshot of the root file content
    root_file = models.CharField(max_length=255, default='main.tex')

    # Output
    pdf = models.FileField(upload_to='compiled_pdfs/', blank=True, null=True)
    log = models.TextField(blank=True)
    error_count = models.IntegerField(default=0)
    warning_count = models.IntegerField(default=0)

    # Timing
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['project', '-created_at']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"Compile {self.id.hex[:8]} – {self.status}"

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
    """Pre-built LaTeX templates for quick project creation."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    category = models.CharField(max_length=100, blank=True, default='general')

    # Template files stored as JSON
    files = models.JSONField(
        default=dict,
        help_text="Dict of filename -> content for each template file"
    )

    is_public = models.BooleanField(default=True)
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
    )
    usage_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-usage_count', 'name']

    def __str__(self):
        return self.name