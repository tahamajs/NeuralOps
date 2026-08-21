import uuid

from django.conf import settings
from django.db import models

User = settings.AUTH_USER_MODEL


class SupportTicket(models.Model):
    """A research problem submitted by a user."""

    STATUS_CHOICES = [
        ("open", "Open"),
        ("analyzing", "Analyzing"),
        ("solved", "Solved"),
        ("closed", "Closed"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="support_tickets"
    )
    title = models.CharField(max_length=300)
    description = models.TextField()
    code_snippet = models.TextField(blank=True)
    log_attachment = models.FileField(upload_to="support_logs/", blank=True, null=True)
    tags = models.JSONField(default=list, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="open")
    # AI response metadata
    ai_summary = models.TextField(blank=True)
    ai_steps = models.JSONField(default=list, blank=True)
    web_sources = models.JSONField(default=list, blank=True)
    # Embedding for semantic search (optional, can be JSON field)
    embedding = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class TicketMessage(models.Model):
    """A message in a support ticket conversation."""

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    ticket = models.ForeignKey(
        SupportTicket, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"


class Solution(models.Model):
    """
    A stored solution that can be re‑used via semantic search.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    ticket = models.ForeignKey(
        SupportTicket, on_delete=models.CASCADE, related_name="solutions"
    )
    content = models.TextField()
    embedding = models.JSONField(default=list, blank=True)
    upvotes = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-upvotes"]

    def __str__(self):
        return f"Solution for {self.ticket.title}"


class Feedback(models.Model):
    """User feedback on an AI response."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    ticket = models.ForeignKey(
        SupportTicket, on_delete=models.CASCADE, related_name="feedback"
    )
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="support_feedback"
    )
    rating = models.PositiveSmallIntegerField(
        choices=[
            (1, "Poor"),
            (2, "Fair"),
            (3, "Good"),
            (4, "Very Good"),
            (5, "Excellent"),
        ]
    )
    comment = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback for {self.ticket.title} by {self.user.username}"

class SupportSolution(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    description = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
