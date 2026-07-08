import uuid

from django.conf import settings
from django.db import models


class Badge(models.Model):
    """
    A badge definition. Each badge has a name, description, icon, tier, and a set of criteria.
    """

    TIER_CHOICES = [
        ("bronze", "Bronze"),
        ("silver", "Silver"),
        ("gold", "Gold"),
        ("platinum", "Platinum"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=10, default="🏅")
    tier = models.CharField(max_length=20, choices=TIER_CHOICES, default="bronze")
    category = models.CharField(
        max_length=50,
        default="general",
        choices=[
            ("publication", "Publication"),
            ("contribution", "Contribution"),
            ("streak", "Streak"),
            ("social", "Social"),
            ("milestone", "Milestone"),
        ],
    )
    criteria = models.JSONField(
        default=dict,
        blank=True,
        help_text='e.g., {"papers_saved": 10, "citations": 100, "days_streak": 7}',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["tier", "name"]

    def __str__(self):
        return f"{self.icon} {self.name} ({self.tier})"


class UserBadge(models.Model):
    """
    A badge awarded to a specific user.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="gamification_badges",
    )
    badge = models.ForeignKey(Badge, on_delete=models.CASCADE)
    awarded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "badge"]]

    def __str__(self):
        return f"{self.badge.name} → {self.user.username}"


class ContributionEvent(models.Model):
    """
    A single contribution event that earns points.
    """

    EVENT_TYPES = [
        ("save_paper", "Save Paper"),
        ("read_paper", "Read Paper"),
        ("comment", "Comment"),
        ("reply", "Reply"),
        ("review", "Review"),
        ("benchmark_submit", "Benchmark Submission"),
        ("note_create", "Note Created"),
        ("note_share", "Note Shared"),
        ("discussion_start", "Discussion Started"),
        ("upvote_received", "Upvote Received"),
        ("answer_accepted", "Answer Accepted"),
        ("follow_gained", "Follower Gained"),
        ("streak_bonus", "Streak Bonus"),
        ("daily_login", "Daily Login"),
    ]
    POINT_MAP = {
        "save_paper": 5,
        "read_paper": 2,
        "comment": 3,
        "reply": 3,
        "review": 15,
        "benchmark_submit": 10,
        "note_create": 5,
        "note_share": 3,
        "discussion_start": 5,
        "upvote_received": 2,
        "answer_accepted": 10,
        "follow_gained": 1,
        "streak_bonus": 5,
        "daily_login": 1,
    }
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="contribution_events",
    )
    event_type = models.CharField(max_length=30, choices=EVENT_TYPES)
    points = models.PositiveIntegerField(default=0)
    description = models.CharField(max_length=300, blank=True)
    related_object_id = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["event_type"]),
        ]

    def __str__(self):
        return f"{self.user.username} +{self.points} ({self.event_type})"

    def save(self, *args, **kwargs):
        if self.points == 0:
            self.points = self.POINT_MAP.get(self.event_type, 0)
        super().save(*args, **kwargs)


class UserStreak(models.Model):
    """
    Tracks daily activity streaks for a user.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="streak"
    )
    current_streak = models.PositiveIntegerField(default=0)
    longest_streak = models.PositiveIntegerField(default=0)
    last_active_date = models.DateField(null=True, blank=True)
    total_points = models.PositiveIntegerField(default=0)
    weekly_points = models.PositiveIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}: {self.current_streak} day streak ({self.total_points} pts)"


class LeaderboardSnapshot(models.Model):
    """
    A periodic snapshot of the leaderboard for historical tracking.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    date = models.DateField(auto_now_add=True)
    field = models.CharField(
        max_length=100, blank=True, help_text="Research field filter"
    )
    institution = models.CharField(
        max_length=200, blank=True, help_text="Institution filter"
    )
    rankings = models.JSONField(
        default=list, help_text="List of {user_id, username, points, rank}"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date"]

    def __str__(self):
        return f"Leaderboard snapshot ({self.date})"


class AchievementUnlock(models.Model):
    """
    A premium feature or capability unlocked by reaching a contribution threshold.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="unlocked_achievements",
    )
    achievement = models.CharField(max_length=200)
    required_points = models.PositiveIntegerField(default=0)
    unlocked_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "achievement"]]

    def __str__(self):
        return f"{self.user.username} unlocked {self.achievement}"


class PremiumFeature(models.Model):
    """
    Definition of a premium feature that can be unlocked via achievements.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    required_points = models.PositiveIntegerField(default=0)
    required_badges = models.ManyToManyField(Badge, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name
