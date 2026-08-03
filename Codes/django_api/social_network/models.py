import uuid

from django.conf import settings
from django.db import models


class Follow(models.Model):
    """
    A user follows another researcher to stay updated.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    follower = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="following"
    )
    following = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="followers"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["follower", "following"]]
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["follower"]),
            models.Index(fields=["following"]),
        ]

    def __str__(self):
        return f"{self.follower.username} → {self.following.username}"


class ResearchGroup(models.Model):
    """
    A research group / lab / interest group.
    """

    VISIBILITY_CHOICES = [
        ("public", "Public"),
        ("private", "Private"),
        ("hidden", "Hidden"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    cover_image = models.ImageField(upload_to="group_covers/", blank=True, null=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="created_groups_network",
    )
    members = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        through="GroupMembership",
        related_name="research_groups_network",
    )
    visibility = models.CharField(
        max_length=20, choices=VISIBILITY_CHOICES, default="public"
    )
    tags = models.JSONField(default=list, blank=True)
    member_count = models.PositiveIntegerField(default=0)
    discussion_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-member_count", "-created_at"]

    def __str__(self):
        return self.name


class GroupMembership(models.Model):
    """
    Membership of a user in a research group.
    """

    ROLE_CHOICES = [
        ("admin", "Admin"),
        ("moderator", "Moderator"),
        ("member", "Member"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="groupmembership_network",
    )
    group = models.ForeignKey(ResearchGroup, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="member")
    joined_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "group"]]

    def __str__(self):
        return f"{self.user.username} in {self.group.name} ({self.role})"


class Discussion(models.Model):
    """
    A discussion thread within a research group.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    group = models.ForeignKey(
        ResearchGroup, on_delete=models.CASCADE, related_name="discussions"
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="discussions_network",
    )
    title = models.CharField(max_length=300)
    content = models.TextField(blank=True)
    is_pinned = models.BooleanField(default=False)
    is_closed = models.BooleanField(default=False)
    view_count = models.PositiveIntegerField(default=0)
    reply_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_pinned", "-updated_at"]

    def __str__(self):
        return self.title


class DiscussionReply(models.Model):
    """
    A reply to a discussion thread.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    discussion = models.ForeignKey(
        Discussion, on_delete=models.CASCADE, related_name="replies"
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="discussion_replies_network",
    )
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"Reply by {self.author.username}"


class ActivityFeed(models.Model):
    """
    An activity event visible in the social feed.
    Aggregates publications, experiments, notes, and other actions.
    """

    ACTIVITY_TYPES = [
        ("paper_saved", "Paper Saved"),
        ("paper_published", "Paper Published"),
        ("experiment_completed", "Experiment Completed"),
        ("note_created", "Note Created"),
        ("note_updated", "Note Updated"),
        ("benchmark_submitted", "Benchmark Submitted"),
        ("group_joined", "Group Joined"),
        ("discussion_started", "Discussion Started"),
        ("reputation_milestone", "Reputation Milestone"),
        ("followed", "Followed Someone"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="activities_network",
    )
    activity_type = models.CharField(max_length=30, choices=ACTIVITY_TYPES)
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    link = models.URLField(blank=True, max_length=500)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "-created_at"]),
            models.Index(fields=["activity_type", "-created_at"]),
        ]

    def __str__(self):
        return f"{self.user.username}: {self.title}"


class PublicationAlert(models.Model):
    """
    Alert sent to followers when a researcher publishes something.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="publication_alerts_network",
    )
    follower = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="received_alerts_network",
    )
    alert_type = models.CharField(max_length=30, choices=ActivityFeed.ACTIVITY_TYPES)
    content = models.TextField()
    link = models.URLField(blank=True, max_length=500)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["follower", "is_read"]),
        ]

    def __str__(self):
        return f"Alert for {self.follower.username}: {self.user.username} {self.alert_type}"
