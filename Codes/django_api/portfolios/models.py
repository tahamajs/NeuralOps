import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone

class Portfolio(models.Model):
    """One‑to‑one public showcase for a researcher."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                related_name='portfolio')
    headline = models.CharField(max_length=300, blank=True, help_text="e.g. 'ML Researcher at XYZ'")
    bio = models.TextField(max_length=800, blank=True)
    is_public = models.BooleanField(default=True)
    theme_color = models.CharField(max_length=20, default='#667eea')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['user__username']

    def __str__(self):
        return f"Portfolio of {self.user.username}"


class ShowcaseItem(models.Model):
    """An item displayed on a user's portfolio (model, paper, badge, etc.)."""
    ITEM_TYPES = [
        ('model', 'Trained Model'),
        ('paper', 'Research Paper'),
        ('compression', 'Compression Experiment'),
        ('leaderboard', 'Leaderboard Rank'),
        ('badge', 'Badge'),
        ('repository', 'Code Repository'),
        ('talk', 'Talk / Presentation'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='items')
    item_type = models.CharField(max_length=20, choices=ITEM_TYPES)
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    url = models.URLField(blank=True)
    metadata = models.JSONField(default=dict, blank=True,
                                help_text="e.g., {'run_id': ..., 'paper_id': ..., 'rank': 3}")
    is_pinned = models.BooleanField(default=False)
    sort_order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-is_pinned', 'sort_order', '-created_at']

    def __str__(self):
        return f"{self.item_type}: {self.title}"


class Endorsement(models.Model):
    """Another user endorses a skill of a researcher."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='endorsements')
    endorser = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                                 related_name='given_endorsements')
    skill = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['portfolio', 'endorser', 'skill']]

    def __str__(self):
        return f"{self.endorser.username} endorsed {self.skill} for {self.portfolio.user.username}"


class ActivityEvent(models.Model):
    """Daily activity log for the GitHub‑style contribution chart."""
    EVENT_TYPES = [
        ('training', 'Training Completed'),
        ('compression', 'Compression Applied'),
        ('paper_saved', 'Paper Saved'),
        ('review', 'Review Written'),
        ('submission', 'Leaderboard Submission'),
        ('benchmark_created', 'Benchmark Created'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name='activity_events')
    event_type = models.CharField(max_length=30, choices=EVENT_TYPES)
    date = models.DateField(db_index=True)
    count = models.PositiveIntegerField(default=1)

    class Meta:
        unique_together = [['user', 'date', 'event_type']]
        ordering = ['-date']

    def __str__(self):
        return f"{self.user.username}: {self.event_type} on {self.date}"


class WidgetEmbed(models.Model):
    """Custom embeddable widget for personal websites."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name='widgets')
    theme = models.CharField(max_length=20, default='light')
    show_models = models.BooleanField(default=True)
    show_papers = models.BooleanField(default=True)
    show_badges = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Widget for {self.portfolio.user.username}"