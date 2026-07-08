# reviews/models.py
"""
Complete Peer Review System — OpenReview-style
Researchers submit papers for review, reviewers respond with structured
feedback and scores, and the community can comment.
"""

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class ReviewRequest(models.Model):
    """A paper / model submitted for peer review."""

    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("submitted", "Submitted"),
        ("in_review", "In Review"),
        ("completed", "Completed"),
        ("withdrawn", "Withdrawn"),
        ("rejected", "Rejected"),
    ]
    VISIBILITY_CHOICES = [
        ("public", "Public"),
        ("private", "Private"),
        ("anonymous", "Anonymous"),
    ]
    SUBMISSION_TYPE = [
        ("paper", "Research Paper"),
        ("model", "Trained Model"),
        ("compression", "Compression Method"),
        ("benchmark", "Benchmark Submission"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="review_requests",
    )
    title = models.CharField(max_length=500)
    abstract = models.TextField(blank=True)
    submission_type = models.CharField(
        max_length=20, choices=SUBMISSION_TYPE, default="paper"
    )
    paper_url = models.URLField(
        blank=True, help_text="arXiv / HuggingFace / external link"
    )
    file = models.FileField(upload_to="review_submissions/", blank=True, null=True)

    # Metadata
    tags = models.JSONField(default=list, blank=True)
    metadata = models.JSONField(
        default=dict,
        blank=True,
        help_text="e.g., {'model_name': '...', 'benchmark': '...'}",
    )

    # Status & visibility
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")
    visibility = models.CharField(
        max_length=20, choices=VISIBILITY_CHOICES, default="public"
    )
    reviewers = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name="assigned_reviews"
    )

    # Deadlines
    reviewers_required = models.PositiveIntegerField(default=3)
    review_deadline = models.DateTimeField(null=True, blank=True)

    # Stats
    view_count = models.PositiveIntegerField(default=0)
    comment_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "-created_at"]),
            models.Index(fields=["author", "-created_at"]),
            models.Index(fields=["visibility"]),
        ]

    def __str__(self):
        return self.title

    @property
    def completed_reviews(self):
        return self.reviews.filter(status="completed")

    @property
    def average_score(self):
        scores = [
            r.overall_score
            for r in self.completed_reviews
            if r.overall_score is not None
        ]
        if not scores:
            return None
        return round(sum(scores) / len(scores), 2)

    @property
    def is_open_for_review(self):
        if self.status != "in_review":
            return False
        if self.review_deadline and timezone.now() > self.review_deadline:
            return False
        return True


class Review(models.Model):
    """A single review response for a submitted paper/model."""

    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("completed", "Completed"),
        ("withdrawn", "Withdrawn"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    review_request = models.ForeignKey(
        ReviewRequest, on_delete=models.CASCADE, related_name="reviews"
    )
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="reviews_given"
    )

    # Structured scores (1-10 scale)
    overall_score = models.PositiveSmallIntegerField(null=True, blank=True)
    originality_score = models.PositiveSmallIntegerField(null=True, blank=True)
    soundness_score = models.PositiveSmallIntegerField(null=True, blank=True)
    presentation_score = models.PositiveSmallIntegerField(null=True, blank=True)
    significance_score = models.PositiveSmallIntegerField(null=True, blank=True)
    reproducibility_score = models.PositiveSmallIntegerField(null=True, blank=True)

    # Text review
    summary = models.TextField(blank=True, help_text="Brief summary of the work")
    strengths = models.TextField(blank=True)
    weaknesses = models.TextField(blank=True)
    suggestions = models.TextField(blank=True)
    questions = models.TextField(blank=True)

    # Meta
    is_anonymous = models.BooleanField(default=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")
    version = models.PositiveIntegerField(default=1)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [["review_request", "reviewer"]]
        ordering = ["-created_at"]

    def __str__(self):
        return f"Review by {self.reviewer.username} on {self.review_request.title[:50]}"


class ReviewComment(models.Model):
    """Discussion thread on a review (author ↔ reviewer dialogue)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    review = models.ForeignKey(
        Review, on_delete=models.CASCADE, related_name="comments"
    )
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    content = models.TextField()
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"Comment by {self.author.username} on {self.review_id}"


class ReviewTemplate(models.Model):
    """Reusable review template for different submission types."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    submission_type = models.CharField(
        max_length=20, choices=ReviewRequest.SUBMISSION_TYPE, default="paper"
    )
    # Structured fields to include
    scoring_criteria = models.JSONField(
        default=list, blank=True, help_text='e.g., [{"name":"Originality","max":10}]'
    )
    template_text = models.TextField(
        blank=True, help_text="Markdown template with placeholders"
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )
    is_default = models.BooleanField(default=False)
    usage_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-usage_count", "name"]

    def __str__(self):
        return self.name


class ReviewBadge(models.Model):
    """Badges for reviewers (like Publons)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=10, default="🏅")
    required_reviews = models.PositiveIntegerField(default=1)
    required_acceptance = models.FloatField(
        default=0.0, help_text="Minimum % of reviews rated as helpful"
    )

    def __str__(self):
        return f"{self.icon} {self.name}"


class ReviewerBadge(models.Model):
    """A badge awarded to a reviewer."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="reviewer_badges",
    )
    badge = models.ForeignKey(ReviewBadge, on_delete=models.CASCADE)
    awarded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["reviewer", "badge"]]


class ReviewVote(models.Model):
    """Upvote/downvote on whether a review was helpful."""

    DIRECTION = [("up", "Helpful"), ("down", "Not Helpful")]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    review = models.ForeignKey(
        Review, on_delete=models.CASCADE, related_name="helpfulness_votes"
    )
    voter = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    direction = models.CharField(max_length=4, choices=DIRECTION)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["review", "voter"]]


class AIReviewAssistant:
    @staticmethod
    def generate_review(paper_content: str, model_id: str = "gpt-4o") -> dict:
        prompt = f"""You are a peer reviewer for a machine learning conference. Review the following paper and provide:
1. Summary (2-3 sentences)
2. Strengths (bullet points)
3. Weaknesses (bullet points)
4. Missing citations (list of papers that should be cited)
5. Soundness score (1-10)
6. Presentation score (1-10)
7. Overall recommendation (accept / weak accept / weak reject / reject)

Paper:
{paper_content[:8000]}

Return JSON with keys: summary, strengths, weaknesses, missing_citations, soundness_score, presentation_score, recommendation"""

        result = LLMGateway.complete(
            model_id, [{"role": "user", "content": prompt}], temperature=0.2
        )
        try:
            return json.loads(result.get("content", "{}"))
        except:
            return {"error": "Failed to parse AI review"}

    @staticmethod
    def check_argument_soundness(paper_content: str) -> list:
        """Identify logical fallacies or unsupported claims."""
        prompt = f"""Analyze the following paper for logical fallacies, unsupported claims, or argument weaknesses. Return a JSON list of issues found.

Paper:
{paper_content[:5000]}

Return: [{{"issue": "...", "location": "...", "severity": "high/medium/low"}}]"""
        result = LLMGateway.complete(
            "gpt-4o", [{"role": "user", "content": prompt}], temperature=0.1
        )
        try:
            return json.loads(result.get("content", "[]"))
        except:
            return []
