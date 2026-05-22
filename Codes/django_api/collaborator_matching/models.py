import uuid, json
from django.db import models
from django.conf import settings


class ResearcherInterestProfile(models.Model):
    """
    A user's dynamic interest profile.
    Stored as a JSON list of floats representing the embedding.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='interest_profile'
    )
    embedding = models.JSONField(default=list)
    topics = models.JSONField(default=list, help_text="Top keywords/topics derived from papers")
    last_generated = models.DateTimeField(auto_now=True)
    paper_count = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"Interest profile of {self.user.username}"


class CollaboratorMatch(models.Model):
    """
    A suggested collaborator match for a user.
    """
    STATUS_CHOICES = [
        ('suggested', 'Suggested'),
        ('accepted', 'Accepted'),
        ('dismissed', 'Dismissed'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='incoming_matches'
    )
    matched_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='outgoing_matches'
    )
    score = models.FloatField(default=0.0, help_text="Cosine similarity score (0–1)")
    common_interests = models.JSONField(default=list, blank=True)
    explanation = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='suggested')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [['user', 'matched_user']]
        ordering = ['-score']

    def __str__(self):
        return f"Match: {self.user.username} ↔ {self.matched_user.username} ({self.score:.2f})"