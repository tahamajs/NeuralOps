import uuid

from django.conf import settings
from django.db import models


class CodeGenerationSession(models.Model):
    """
    A single code‑generation session (like a Copilot chat thread).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="code_sessions"
    )
    title = models.CharField(max_length=300, blank=True)
    context = models.JSONField(
        default=dict,
        blank=True,
        help_text="Context for generation: {'model_name':..., 'task_name':..., 'file':...}",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title or f"Session {self.id.hex[:8]}"


class CodeMessage(models.Model):
    """
    A single message / exchange in a code generation session.
    """

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        CodeGenerationSession, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    language = models.CharField(max_length=30, default="python")
    tokens_used = models.PositiveIntegerField(default=0)
    model_used = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:80]}"


class GeneratedCode(models.Model):
    """
    A saved piece of generated code that can be reviewed, edited, and exported.
    """

    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("reviewing", "In Review"),
        ("approved", "Approved"),
        ("exported", "Exported"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        CodeGenerationSession, on_delete=models.CASCADE, related_name="generated_files"
    )
    filename = models.CharField(max_length=300)
    content = models.TextField()
    language = models.CharField(max_length=30, default="python")
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")
    version = models.PositiveIntegerField(default=1)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.filename


class CodeReview(models.Model):
    """
    An AI‑generated code review for a piece of generated code.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    generated_code = models.ForeignKey(
        GeneratedCode, on_delete=models.CASCADE, related_name="reviews"
    )
    issues = models.JSONField(default=list, blank=True)
    suggestions = models.JSONField(default=list, blank=True)
    overall_assessment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review for {self.generated_code.filename}"


class TestGeneration(models.Model):
    """
    Auto‑generated tests for a piece of code.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    generated_code = models.OneToOneField(
        GeneratedCode, on_delete=models.CASCADE, related_name="generated_tests"
    )
    test_content = models.TextField()
    test_framework = models.CharField(max_length=30, default="pytest")
    coverage_estimate = models.FloatField(default=0.0, help_text="Estimated coverage %")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Tests for {self.generated_code.filename}"
