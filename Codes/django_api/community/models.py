# community/models.py
"""
Complete Research Q&A Community — Stack Overflow + ResearchGate + Discourse
Reputation system · Badges (bronze/silver/gold) · Trust levels · Bounties ·
Privileges · Weekly digest · Skills-based routing
"""

import uuid

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


# ═══════════════════════════════════════════════════════════════
# 1. TAG
# ═══════════════════════════════════════════════════════════════
class Tag(models.Model):
    """Research topic tag — e.g. 'nlp', 'compression', 'attention-mechanisms'."""

    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=120, unique=True)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=10, blank=True, default="🏷️")
    is_official = models.BooleanField(
        default=False, help_text="Set by staff for canonical tags"
    )
    parent = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="children",
    )
    question_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-question_count", "name"]

    def __str__(self):
        return self.name


# ═══════════════════════════════════════════════════════════════
# 2. QUESTION
# ═══════════════════════════════════════════════════════════════
class Question(models.Model):
    """A research question posted by a user."""

    STATUS_CHOICES = [
        ("open", "Open"),
        ("closed", "Closed"),
        ("duplicate", "Duplicate"),
        ("migrated", "Migrated"),
        ("deleted", "Deleted"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="community_questions",
    )
    title = models.CharField(max_length=300)
    body = models.TextField(help_text="Markdown content with LaTeX support")
    tags = models.ManyToManyField(Tag, blank=True, related_name="questions")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="open")
    close_reason = models.TextField(blank=True)

    # Scores
    score = models.IntegerField(default=0, db_index=True)
    upvotes = models.PositiveIntegerField(default=0)
    downvotes = models.PositiveIntegerField(default=0)
    view_count = models.PositiveIntegerField(default=0)
    answer_count = models.PositiveIntegerField(default=0)

    # Answers
    accepted_answer = models.ForeignKey(
        "Answer",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="accepted_for",
    )
    has_bounty = models.BooleanField(default=False)

    # Metadata
    linked_paper = models.ForeignKey(
        "agents.ResearchPaper",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="Linked research paper",
    )
    linked_run = models.ForeignKey(
        "jobscheduler.Job",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="Linked training run",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_activity_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-last_activity_at"]
        indexes = [
            models.Index(fields=["author", "-created_at"]),
            models.Index(fields=["score"]),
            models.Index(fields=["-last_activity_at"]),
        ]

    def __str__(self):
        return self.title[:80]

    def update_score(self):
        self.upvotes = self.votes.filter(direction="up").count()
        self.downvotes = self.votes.filter(direction="down").count()
        self.score = self.upvotes - self.downvotes
        self.save(update_fields=["score", "upvotes", "downvotes"])


# ═══════════════════════════════════════════════════════════════
# 3. ANSWER
# ═══════════════════════════════════════════════════════════════
class Answer(models.Model):
    """An answer to a question."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.ForeignKey(
        Question, on_delete=models.CASCADE, related_name="answers"
    )
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="community_answers",
    )
    body = models.TextField(help_text="Markdown content with LaTeX support")
    is_accepted = models.BooleanField(default=False)

    score = models.IntegerField(default=0, db_index=True)
    upvotes = models.PositiveIntegerField(default=0)
    downvotes = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_accepted", "-score", "-created_at"]
        indexes = [
            models.Index(fields=["question", "-score"]),
            models.Index(fields=["author", "-created_at"]),
        ]

    def __str__(self):
        return f"Answer by {self.author.username} on {self.question.title[:40]}"

    def update_score(self):
        self.upvotes = self.votes.filter(direction="up").count()
        self.downvotes = self.votes.filter(direction="down").count()
        self.score = self.upvotes - self.downvotes
        self.save(update_fields=["score", "upvotes", "downvotes"])


# ═══════════════════════════════════════════════════════════════
# 4. COMMENT (on questions/answers)
# ═══════════════════════════════════════════════════════════════
class Comment(models.Model):
    """A comment on a question or answer."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.ForeignKey(
        Question,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="comments",
    )
    answer = models.ForeignKey(
        Answer, on_delete=models.CASCADE, null=True, blank=True, related_name="comments"
    )
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    body = models.TextField(max_length=600)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"Comment by {self.author.username}"


# ═══════════════════════════════════════════════════════════════
# 5. VOTE
# ═══════════════════════════════════════════════════════════════
class Vote(models.Model):
    """A vote on a question or answer."""

    DIRECTION = [("up", "Upvote"), ("down", "Downvote")]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    voter = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="community_votes",
    )
    question = models.ForeignKey(
        Question, on_delete=models.CASCADE, null=True, blank=True, related_name="votes"
    )
    answer = models.ForeignKey(
        Answer, on_delete=models.CASCADE, null=True, blank=True, related_name="votes"
    )
    direction = models.CharField(max_length=4, choices=DIRECTION)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["voter", "question"], ["voter", "answer"]]


# ═══════════════════════════════════════════════════════════════
# 6. BOOKMARK / SAVED
# ═══════════════════════════════════════════════════════════════
class Bookmark(models.Model):
    """User bookmarks a question for later."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="question_bookmarks",
    )
    question = models.ForeignKey(
        Question, on_delete=models.CASCADE, related_name="bookmarks"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "question"]]


# ═══════════════════════════════════════════════════════════════
# 7. BOUNTY
# ═══════════════════════════════════════════════════════════════
class Bounty(models.Model):
    """A reputation bounty placed on a question."""

    REASONS = [
        ("authoritative", "Authoritative reference needed"),
        ("canonical", "Canonical answer required"),
        ("draw_attention", "Draw attention"),
        ("improve_details", "Improve details"),
        ("reward_existing", "Reward existing answer"),
        ("current_answers", "Current answers are outdated"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.ForeignKey(
        Question, on_delete=models.CASCADE, related_name="bounties"
    )
    offerer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="offered_bounties",
    )
    amount = models.PositiveIntegerField(
        validators=[MinValueValidator(50), MaxValueValidator(500)]
    )
    reason = models.CharField(max_length=30, choices=REASONS, default="draw_attention")
    notice = models.TextField(blank=True)
    awarded_to = models.ForeignKey(
        Answer, on_delete=models.SET_NULL, null=True, blank=True
    )
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def is_active(self):
        return not self.awarded_to and timezone.now() < self.expires_at


# ═══════════════════════════════════════════════════════════════
# 8. REPUTATION EVENT
# ═══════════════════════════════════════════════════════════════
class ReputationEvent(models.Model):
    """A single reputation change event for audit trail."""

    EVENT_TYPES = [
        ("question_upvoted", "Question Upvoted"),
        ("question_downvoted", "Question Downvoted"),
        ("answer_upvoted", "Answer Upvoted"),
        ("answer_downvoted", "Answer Downvoted"),
        ("answer_accepted", "Answer Accepted"),
        ("answer_accept_given", "Answer Accepted (Acceptor)"),
        ("bounty_awarded", "Bounty Awarded"),
        ("bounty_offered", "Bounty Offered"),
        ("bounty_expired", "Bounty Expired"),
        ("association_bonus", "Association Bonus"),
        ("downvoted_answer", "Downvoted an Answer"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="community_reputation_events",
    )
    event_type = models.CharField(max_length=30, choices=EVENT_TYPES)
    amount = models.IntegerField()
    question = models.ForeignKey(
        Question, on_delete=models.SET_NULL, null=True, blank=True
    )
    answer = models.ForeignKey(Answer, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


# ═══════════════════════════════════════════════════════════════
# 9. TRUST LEVEL (Discourse-style)
# ═══════════════════════════════════════════════════════════════
class TrustLevel(models.Model):
    """Discourse-inspired trust level assigned to a user."""

    LEVELS = [
        (0, "New User"),
        (1, "Basic User"),
        (2, "Member"),
        (3, "Regular"),
        (4, "Expert"),
    ]
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="trust_level"
    )
    level = models.PositiveSmallIntegerField(default=0, choices=LEVELS)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}: TL{self.level}"


# ═══════════════════════════════════════════════════════════════
# 10. BADGE
# ═══════════════════════════════════════════════════════════════
class Badge(models.Model):
    """A badge definition — like Stack Overflow badges."""

    TIERS = [("bronze", "Bronze"), ("silver", "Silver"), ("gold", "Gold")]
    CATEGORIES = [
        ("participation", "Participation"),
        ("quality", "Quality"),
        ("moderation", "Moderation"),
        ("tag", "Tag Badge"),
        ("bounty", "Bounty"),
        ("special", "Special"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    tier = models.CharField(max_length=10, choices=TIERS, default="bronze")
    category = models.CharField(
        max_length=20, choices=CATEGORIES, default="participation"
    )
    icon = models.CharField(max_length=10, default="🏅")
    tag = models.ForeignKey(
        Tag,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="For tag badges",
    )
    condition = models.JSONField(
        default=dict, help_text='e.g., {"score": 100, "answers": 20}'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["tier", "name"]

    def __str__(self):
        return f"{self.tier}: {self.name}"


class UserBadge(models.Model):
    """A badge awarded to a user."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="community_badges",
    )
    badge = models.ForeignKey(Badge, on_delete=models.CASCADE)
    question = models.ForeignKey(
        Question, on_delete=models.SET_NULL, null=True, blank=True
    )
    answer = models.ForeignKey(Answer, on_delete=models.SET_NULL, null=True, blank=True)
    awarded_at = models.DateTimeField(auto_now_add=True)
    count = models.PositiveIntegerField(
        default=1, help_text="How many times awarded (for tag badges)"
    )

    class Meta:
        unique_together = [["user", "badge", "question"], ["user", "badge", "answer"]]


# ═══════════════════════════════════════════════════════════════
# 11. PRIVILEGE
# ═══════════════════════════════════════════════════════════════
class Privilege(models.Model):
    """Reputation-based privilege definition."""

    name = models.CharField(max_length=200, unique=True)
    slug = models.SlugField(max_length=200, unique=True)
    description = models.TextField(blank=True)
    reputation_required = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["reputation_required"]

    def __str__(self):
        return f"{self.name} ({self.reputation_required} rep)"


# ═══════════════════════════════════════════════════════════════
# 12. ACTIVITY LOG (GitHub-style contribution graph)
# ═══════════════════════════════════════════════════════════════
class ActivityLog(models.Model):
    """Daily activity log for the contribution graph."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="community_activity",
    )
    date = models.DateField(db_index=True)
    questions_asked = models.PositiveIntegerField(default=0)
    answers_posted = models.PositiveIntegerField(default=0)
    votes_cast = models.PositiveIntegerField(default=0)
    comments_posted = models.PositiveIntegerField(default=0)
    accepted_answers = models.PositiveIntegerField(default=0)

    class Meta:
        unique_together = [["user", "date"]]
        ordering = ["-date"]


# ═══════════════════════════════════════════════════════════════
# 13. DIGEST SUBSCRIPTION
# ═══════════════════════════════════════════════════════════════
class DigestSubscription(models.Model):
    """User subscription to weekly digest emails."""

    FREQUENCIES = [("daily", "Daily"), ("weekly", "Weekly"), ("monthly", "Monthly")]
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="digest_subscription",
    )
    frequency = models.CharField(max_length=10, choices=FREQUENCIES, default="weekly")
    include_trending = models.BooleanField(default=True)
    include_my_tags = models.BooleanField(default=True)
    last_sent_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Digest for {self.user.username}"
