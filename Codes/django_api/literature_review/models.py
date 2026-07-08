# literature_review/models.py
import uuid

from django.conf import settings
from django.db import models


class LiteratureProject(models.Model):
    """A literature review project with step‑by‑step workflow."""

    STEP_CHOICES = [
        (1, "Topic Finder"),
        (2, "Research Notebook"),
        (3, "Deep Review"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="lit_projects"
    )
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    current_step = models.PositiveSmallIntegerField(choices=STEP_CHOICES, default=1)
    progress = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title


class Topic(models.Model):
    """A research topic with citation‑backed subtopics, themes, and gaps."""

    project = models.ForeignKey(
        LiteratureProject, on_delete=models.CASCADE, related_name="topics"
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    subtopics = models.JSONField(default=list, help_text="Citation‑backed subtopics")
    key_themes = models.JSONField(default=list)
    research_gaps = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class ResearchNotebookSection(models.Model):
    """A thematic section of the evolving research notebook."""

    project = models.ForeignKey(
        LiteratureProject, on_delete=models.CASCADE, related_name="notebook_sections"
    )
    title = models.CharField(max_length=300)
    content = models.TextField(blank=True)
    citations = models.JSONField(default=list, help_text="List of paper IDs")
    order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["order"]

    def __str__(self):
        return f"{self.title} ({self.project.title})"


class DeepReview(models.Model):
    """Deep review of 20+ papers, with notes and generated paragraphs."""

    project = models.ForeignKey(
        LiteratureProject, on_delete=models.CASCADE, related_name="deep_reviews"
    )
    title = models.CharField(max_length=300)
    notes = models.TextField(blank=True, help_text="Aggregated notes from all papers")
    generated_paragraphs = models.JSONField(
        default=list, help_text="AI‑generated paragraphs for literature review"
    )
    ethical_ai_guidelines = models.BooleanField(default=False)
    papers_analyzed = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title


class PaperReviewNote(models.Model):
    """Notes for a single paper within a deep review."""

    deep_review = models.ForeignKey(
        DeepReview, on_delete=models.CASCADE, related_name="paper_notes"
    )
    paper = models.ForeignKey("agents.ResearchPaper", on_delete=models.CASCADE)
    notes = models.TextField(blank=True)
    key_points = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["deep_review", "paper"]]


class LiteratureSearch(models.Model):
    """Saved search query with results."""

    project = models.ForeignKey(
        LiteratureProject, on_delete=models.CASCADE, related_name="searches"
    )
    query = models.CharField(max_length=300)
    results = models.JSONField(default=list, help_text="List of paper IDs from search")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Search: {self.query}"


class ResearchGap(models.Model):
    """Identified research gap from topic analysis."""

    project = models.ForeignKey(
        LiteratureProject, on_delete=models.CASCADE, related_name="gaps"
    )
    topic = models.ForeignKey(
        Topic, on_delete=models.CASCADE, related_name="gaps", null=True, blank=True
    )
    description = models.TextField()
    importance = models.FloatField(default=0.5)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.description[:100]
