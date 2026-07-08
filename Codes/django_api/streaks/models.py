import uuid

from django.conf import settings
from django.db import models
from django.utils import timezone


class DailyActivity(models.Model):
    """
    Counts of research activities for a user on a specific date.
    One row per user per date.  Activity types are stored as integer fields.
    """

    ACTIVITY_TYPES = [
        ("paper_read", "Paper Read / Saved"),
        ("experiment_run", "Experiment Run"),
        ("note_created", "Note Created"),
        ("paper_written", "Paper / LaTeX Edited"),
        ("code_commit", "Code Commit"),
        ("benchmark_submission", "Benchmark Submission"),
        ("discussion_post", "Discussion / Reply"),
    ]
    # We'll store each type as a separate integer field for simplicity.
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="daily_activities",
    )
    date = models.DateField(db_index=True)

    paper_reads = models.PositiveIntegerField(default=0)
    experiments_run = models.PositiveIntegerField(default=0)
    notes_created = models.PositiveIntegerField(default=0)
    papers_written = models.PositiveIntegerField(default=0)
    code_commits = models.PositiveIntegerField(default=0)
    benchmark_submissions = models.PositiveIntegerField(default=0)
    discussion_posts = models.PositiveIntegerField(default=0)

    # Total count for heatmap coloring
    total_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [["user", "date"]]
        ordering = ["-date"]
        indexes = [
            models.Index(fields=["user", "-date"]),
        ]

    def __str__(self):
        return f"{self.user.username} on {self.date}: {self.total_count} activities"

    def recalculate_total(self):
        self.total_count = (
            self.paper_reads
            + self.experiments_run
            + self.notes_created
            + self.papers_written
            + self.code_commits
            + self.benchmark_submissions
            + self.discussion_posts
        )
        self.save(update_fields=["total_count"])

    @staticmethod
    def record(user, activity_type: str, increment: int = 1):
        """Increment the given activity type for today."""
        today = timezone.now().date()
        obj, _ = DailyActivity.objects.get_or_create(user=user, date=today)
        field_map = {
            "paper_read": "paper_reads",
            "experiment_run": "experiments_run",
            "note_created": "notes_created",
            "paper_written": "papers_written",
            "code_commit": "code_commits",
            "benchmark_submission": "benchmark_submissions",
            "discussion_post": "discussion_posts",
        }
        field = field_map.get(activity_type)
        if field:
            setattr(obj, field, getattr(obj, field) + increment)
            obj.recalculate_total()
            # Update streak
            streak, _ = Streak.objects.get_or_create(user=user)
            streak.update_streak(today)


class Streak(models.Model):
    """
    Tracks the current and longest streak for a user.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="streak"
    )
    current_streak = models.PositiveIntegerField(default=0)
    longest_streak = models.PositiveIntegerField(default=0)
    last_active_date = models.DateField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}: {self.current_streak} days (best: {self.longest_streak})"

    def update_streak(self, activity_date):
        """Update streak after an activity on activity_date."""

        if self.last_active_date is None:
            # First activity ever
            self.current_streak = 1
            self.longest_streak = 1
        else:
            delta = (activity_date - self.last_active_date).days
            if delta == 0:
                # Same day, nothing to change
                return
            elif delta == 1:
                # Consecutive day
                self.current_streak += 1
            else:
                # Break in streak
                self.current_streak = 1

        self.last_active_date = activity_date
        if self.current_streak > self.longest_streak:
            self.longest_streak = self.current_streak
        self.save()


class WeeklySummary(models.Model):
    """
    A snapshot of a user's activity for a given week.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="weekly_summaries",
    )
    week_start = models.DateField()
    total_activities = models.PositiveIntegerField(default=0)
    paper_reads = models.PositiveIntegerField(default=0)
    experiments_run = models.PositiveIntegerField(default=0)
    notes_created = models.PositiveIntegerField(default=0)
    papers_written = models.PositiveIntegerField(default=0)
    most_active_day = models.DateField(null=True, blank=True)
    summary_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "week_start"]]
        ordering = ["-week_start"]

    def __str__(self):
        return f"Week {self.week_start} for {self.user.username}"
