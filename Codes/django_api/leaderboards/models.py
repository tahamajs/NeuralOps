# leaderboards/models.py
"""
Complete Leaderboard & Benchmark System
Inspired by: PapersWithCode taxonomy, HuggingFace Open LLM Leaderboard, Kaggle competitions.

Taxonomy:  Task → Dataset → Metric → Submission → Score
"""

import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


# ═══════════════════════════════════════════════════════════════
# 1. TASK — the high-level NLP/CV task (e.g. "Sentiment Analysis")
# ═══════════════════════════════════════════════════════════════
class BenchmarkTask(models.Model):
    """A high-level ML task, e.g. Sentiment Analysis, NER, Summarization."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    category = models.CharField(
        max_length=100,
        blank=True,
        default="NLP",
        choices=[
            ("NLP", "NLP"),
            ("CV", "Computer Vision"),
            ("Audio", "Audio"),
            ("Multimodal", "Multimodal"),
            ("RL", "Reinforcement Learning"),
        ],
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["category", "name"]

    def __str__(self):
        return self.name


# ═══════════════════════════════════════════════════════════════
# 2. DATASET — a specific evaluation dataset
# ═══════════════════════════════════════════════════════════════
class BenchmarkDataset(models.Model):
    """A specific evaluation dataset, e.g. SST-2, MNLI, GLUE."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    slug = models.SlugField(max_length=200, unique=True)
    task = models.ForeignKey(
        BenchmarkTask, on_delete=models.CASCADE, related_name="datasets"
    )
    description = models.TextField(blank=True)
    paper_url = models.URLField(blank=True)
    train_size = models.PositiveIntegerField(null=True, blank=True)
    test_size = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["task", "name"]

    def __str__(self):
        return f"{self.name} ({self.task.name})"


# ═══════════════════════════════════════════════════════════════
# 3. METRIC — a specific evaluation metric
# ═══════════════════════════════════════════════════════════════
class Metric(models.Model):
    """A specific evaluation metric, e.g. Accuracy, F1, BLEU, ROUGE."""

    METRIC_DIRECTIONS = [("maximize", "Maximize"), ("minimize", "Minimize")]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True)
    unit = models.CharField(max_length=30, blank=True, default="")
    direction = models.CharField(
        max_length=10, choices=METRIC_DIRECTIONS, default="maximize"
    )
    is_primary = models.BooleanField(
        default=False, help_text="Primary metric for ranking"
    )

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


# ═══════════════════════════════════════════════════════════════
# 4. BENCHMARK — the competition / leaderboard
# ═══════════════════════════════════════════════════════════════
class Benchmark(models.Model):
    """A complete benchmark (competition) that users submit results to."""

    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("active", "Active"),
        ("paused", "Paused"),
        ("completed", "Completed"),
        ("archived", "Archived"),
    ]
    VISIBILITY_CHOICES = [
        ("public", "Public"),
        ("private", "Private"),
        ("unlisted", "Unlisted"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=300)
    slug = models.SlugField(max_length=300, unique=True)
    description = models.TextField(blank=True)

    # Organiser
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="owned_benchmarks",
    )
    collaborators = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name="collab_benchmarks"
    )

    # Scope
    tasks = models.ManyToManyField(BenchmarkTask, through="BenchmarkTaskMembership")
    datasets = models.ManyToManyField(
        BenchmarkDataset, through="BenchmarkDatasetMembership"
    )
    metrics = models.ManyToManyField(Metric, through="BenchmarkMetricMembership")
    primary_metric = models.ForeignKey(
        Metric,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="primary_for_benchmarks",
    )

    # Competition settings (Kaggle-style)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")
    visibility = models.CharField(
        max_length=20, choices=VISIBILITY_CHOICES, default="public"
    )
    submission_limit_per_user = models.PositiveIntegerField(default=5)
    public_split_ratio = models.FloatField(
        default=0.4, help_text="Fraction of test data visible on public LB"
    )
    opens_at = models.DateTimeField(null=True, blank=True)
    closes_at = models.DateTimeField(null=True, blank=True)
    final_results_published = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    @property
    def is_open(self):
        now = timezone.now()
        if self.status != "active":
            return False
        if self.opens_at and now < self.opens_at:
            return False
        if self.closes_at and now > self.closes_at:
            return False
        return True

    @property
    def submission_count(self):
        return self.submissions.count()


class BenchmarkTaskMembership(models.Model):
    benchmark = models.ForeignKey(Benchmark, on_delete=models.CASCADE)
    task = models.ForeignKey(BenchmarkTask, on_delete=models.CASCADE)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order"]


class BenchmarkDatasetMembership(models.Model):
    benchmark = models.ForeignKey(Benchmark, on_delete=models.CASCADE)
    dataset = models.ForeignKey(BenchmarkDataset, on_delete=models.CASCADE)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order"]


class BenchmarkMetricMembership(models.Model):
    benchmark = models.ForeignKey(Benchmark, on_delete=models.CASCADE)
    metric = models.ForeignKey(Metric, on_delete=models.CASCADE)
    weight = models.FloatField(
        default=1.0, help_text="Weight of this metric in the composite score"
    )
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order"]


# ═══════════════════════════════════════════════════════════════
# 5. SUBMISSION — a single model submission to a benchmark
# ═══════════════════════════════════════════════════════════════
class Submission(models.Model):
    """One model submission to a benchmark."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("evaluating", "Evaluating"),
        ("completed", "Completed"),
        ("failed", "Failed"),
        ("withdrawn", "Withdrawn"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    benchmark = models.ForeignKey(
        Benchmark, on_delete=models.CASCADE, related_name="submissions"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="submissions"
    )
    team_name = models.CharField(max_length=200, blank=True)
    model_name = models.CharField(max_length=300)
    model_description = models.TextField(blank=True)
    training_run = models.ForeignKey(
        "jobscheduler.Job",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="benchmark_submissions",
    )

    # Metadata
    architecture = models.CharField(max_length=300, blank=True)
    parameters_million = models.FloatField(null=True, blank=True)
    hardware_used = models.CharField(max_length=300, blank=True)
    code_url = models.URLField(blank=True)
    paper_url = models.URLField(blank=True)
    huggingface_model_id = models.CharField(max_length=300, blank=True)

    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    error_message = models.TextField(blank=True)

    # Kaggle-style: keep best score
    public_score = models.FloatField(null=True, blank=True)
    private_score = models.FloatField(null=True, blank=True)
    composite_score = models.FloatField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Weighted composite score across all metrics",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-composite_score"]
        indexes = [
            models.Index(fields=["benchmark", "user", "-created_at"]),
            models.Index(fields=["benchmark", "-composite_score"]),
        ]

    def __str__(self):
        return f"{self.model_name} @ {self.benchmark.name}"

    @property
    def rank(self):
        """Compute rank from composite_score using Django ORM Window functions."""
        from django.db.models import F, Window
        from django.db.models.functions import Rank

        qs = (
            Submission.objects.filter(benchmark=self.benchmark, status="completed")
            .annotate(
                rank=Window(expression=Rank(), order_by=F("composite_score").desc())
            )
            .values("id", "rank")
        )
        for row in qs:
            if row["id"] == self.id:
                return row["rank"]
        return None


# ═══════════════════════════════════════════════════════════════
# 6. SCORE — per-metric scores for a submission
# ═══════════════════════════════════════════════════════════════
class Score(models.Model):
    """A single metric score for a submission."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    submission = models.ForeignKey(
        Submission, on_delete=models.CASCADE, related_name="scores"
    )
    metric = models.ForeignKey(Metric, on_delete=models.CASCADE)
    dataset = models.ForeignKey(BenchmarkDataset, on_delete=models.CASCADE)
    value = models.FloatField()
    is_public = models.BooleanField(
        default=True, help_text="Public or private split (Kaggle-style)"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["submission", "metric", "dataset", "is_public"]]

    def __str__(self):
        return f"{self.submission.model_name} | {self.metric.name}: {self.value}"


# ═══════════════════════════════════════════════════════════════
# 7. VOTE — community voting (like HuggingFace Open LLM LB)
# ═══════════════════════════════════════════════════════════════
class Vote(models.Model):
    """Community vote on a submission."""

    DIRECTION_CHOICES = [("up", "Upvote"), ("down", "Downvote")]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    submission = models.ForeignKey(
        Submission, on_delete=models.CASCADE, related_name="votes"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="leaderboard_votes",
    )
    direction = models.CharField(max_length=4, choices=DIRECTION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["submission", "user"]]

    def __str__(self):
        return f"{self.user.username} {self.direction} on {self.submission_id}"


# ═══════════════════════════════════════════════════════════════
# 8. BADGE — rewards for leaderboard achievements
# ═══════════════════════════════════════════════════════════════
class Badge(models.Model):
    """Achievement badges awarded to users."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=10, default="🏆")
    benchmark = models.ForeignKey(
        Benchmark,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="badges",
    )
    condition = models.JSONField(
        default=dict, help_text="e.g. {'rank': 1, 'submission_count': 5}"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.icon} {self.name}"


class UserBadge(models.Model):
    """A badge awarded to a specific user."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="leaderboard_badges",
    )
    badge = models.ForeignKey(Badge, on_delete=models.CASCADE)
    submission = models.ForeignKey(
        Submission, on_delete=models.CASCADE, null=True, blank=True
    )
    awarded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "badge", "submission"]]

    def __str__(self):
        return f"{self.badge.name} → {self.user.username}"


# ═══════════════════════════════════════════════════════════════
# 9. FOLLOW — watch a benchmark for updates
# ═══════════════════════════════════════════════════════════════
class BenchmarkFollow(models.Model):
    """User follows a benchmark to receive notifications."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="followed_benchmarks",
    )
    benchmark = models.ForeignKey(
        Benchmark, on_delete=models.CASCADE, related_name="followers"
    )
    notify_on_new_submission = models.BooleanField(default=True)
    notify_on_rank_change = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "benchmark"]]
