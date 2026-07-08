import uuid

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class AnnotationProject(models.Model):
    """
    A data annotation project – like a Label Studio project.
    """

    TASK_TYPES = [
        ("classification", "Text Classification"),
        ("ner", "Named Entity Recognition"),
        ("qa", "Question Answering"),
        ("summarization", "Summarization"),
        ("sentiment", "Sentiment Analysis"),
        ("custom", "Custom"),
    ]
    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("active", "Active"),
        ("completed", "Completed"),
        ("archived", "Archived"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="annotation_projects",
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    task_type = models.CharField(
        max_length=30, choices=TASK_TYPES, default="classification"
    )

    # Label Studio integration
    label_studio_project_id = models.IntegerField(null=True, blank=True)
    label_studio_url = models.URLField(blank=True)

    # Annotation configuration (Label Studio XML format)
    labeling_config = models.TextField(
        blank=True, help_text="Label Studio XML config for the labeling interface"
    )

    # Labels / classes
    labels = models.JSONField(
        default=list, blank=True, help_text="e.g., ['positive', 'negative', 'neutral']"
    )

    # Stats
    total_tasks = models.PositiveIntegerField(default=0)
    completed_tasks = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")

    # AI-assisted labeling
    ai_model_id = models.CharField(
        max_length=300,
        blank=True,
        help_text="Model ID to use for AI-assisted pre-labeling (e.g., 'bert-base-uncased')",
    )
    ai_confidence_threshold = models.FloatField(
        default=0.8, validators=[MinValueValidator(0), MaxValueValidator(1)]
    )

    # Collaborators
    annotators = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name="annotation_assignments"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class DataTask(models.Model):
    """
    A single data item to be annotated (like a Label Studio task).
    """

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("in_progress", "In Progress"),
        ("completed", "Completed"),
        ("skipped", "Skipped"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        AnnotationProject, on_delete=models.CASCADE, related_name="tasks"
    )
    data = models.JSONField(
        default=dict, help_text="The actual data to label (text, image URL, etc.)"
    )

    # AI pre-labeling
    ai_predictions = models.JSONField(default=list, blank=True)
    ai_confidence = models.FloatField(null=True, blank=True)

    # Human annotation
    annotations = models.JSONField(default=list, blank=True)
    annotated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="annotated_tasks",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")

    # Metadata
    task_index = models.PositiveIntegerField(default=0)
    is_ground_truth = models.BooleanField(default=False)
    agreement_score = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    annotated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["task_index"]
        unique_together = [["project", "task_index"]]

    def __str__(self):
        return f"Task {self.task_index} in {self.project.name}"


class AnnotationReview(models.Model):
    """
    A review of an annotation (quality control).
    """

    STATUS_CHOICES = [
        ("approved", "Approved"),
        ("rejected", "Rejected"),
        ("flagged", "Flagged for Review"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.ForeignKey(DataTask, on_delete=models.CASCADE, related_name="reviews")
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="annotation_reviews",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="flagged")
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Review of {self.task} by {self.reviewer.username}: {self.status}"


class AnnotationGuideline(models.Model):
    """
    Annotation guidelines for a project.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.OneToOneField(
        AnnotationProject, on_delete=models.CASCADE, related_name="guidelines"
    )
    content = models.TextField(
        help_text="Markdown content with annotation instructions"
    )
    examples = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Guidelines for {self.project.name}"


class AIModelFeedback(models.Model):
    """
    Feedback on AI predictions to improve the model (active learning loop).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    task = models.ForeignKey(
        DataTask, on_delete=models.CASCADE, related_name="ai_feedback"
    )
    ai_prediction = models.JSONField(default=dict)
    human_correction = models.JSONField(default=dict)
    is_correct = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"AI Feedback for {self.task} – {'Correct' if self.is_correct else 'Incorrect'}"
