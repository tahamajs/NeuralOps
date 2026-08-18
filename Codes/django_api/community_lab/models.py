import uuid

from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class ResearchLab(models.Model):
    """
    An AI + Human research lab created from a single prompt.
    Contains customizable agent roles, human collaborators, and resources.
    """

    STATUS_CHOICES = [
        ("forming", "Forming"),
        ("active", "Active"),
        ("paused", "Paused"),
        ("completed", "Completed"),
        ("archived", "Archived"),
    ]
    VISIBILITY_CHOICES = [
        ("private", "Private"),
        ("invite_only", "Invite Only"),
        ("public", "Public"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    creator = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="created_labs"
    )
    prompt = models.TextField(
        blank=True,
        default="",
        help_text="The original prompt that created this lab, e.g. 'Build a lab to improve MRPC accuracy above 0.9'",
    )
    objective = models.TextField(blank=True)

    # Status & Visibility
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="forming")
    visibility = models.CharField(
        max_length=20, choices=VISIBILITY_CHOICES, default="private"
    )

    # Members
    human_collaborators = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name="joined_labs"
    )

    # Resources
    max_cycles = models.PositiveIntegerField(
        default=100, help_text="Maximum agent research cycles"
    )
    budget_limit_cents = models.PositiveIntegerField(
        default=5000, help_text="Budget in cents ($50.00 default)"
    )
    spent_cents = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    @property
    def member_count(self):
        return self.human_collaborators.count() + self.agents.count()


class AgentRole(models.Model):
    """
    A customizable AI agent role within a research lab.
    Examples: "Literature Reviewer", "Experiment Designer", "Code Optimizer", "Critic"
    """

    ROLE_TYPES = [
        ("researcher", "Researcher"),
        ("reviewer", "Reviewer"),
        ("experimenter", "Experimenter"),
        ("critic", "Critic"),
        ("writer", "Writer"),
        ("coordinator", "Coordinator"),
        ("custom", "Custom"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lab = models.ForeignKey(
        ResearchLab, on_delete=models.CASCADE, related_name="agent_roles"
    )
    name = models.CharField(max_length=200)
    role_type = models.CharField(max_length=20, choices=ROLE_TYPES, default="custom")
    description = models.TextField(blank=True)
    system_prompt = models.TextField(
        help_text="Custom system prompt that defines this agent's behavior"
    )
    model_name = models.CharField(
        max_length=100,
        default="claude-sonnet-4-20250514",
        help_text="LLM model used for this agent",
    )
    temperature = models.FloatField(default=0.7)
    tools_allowed = models.JSONField(default=list, blank=True)
    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["lab", "role_type", "name"]

    def __str__(self):
        return f"{self.name} ({self.role_type}) @ {self.lab.name}"


class AgentAction(models.Model):
    """
    A single action taken by an agent in a research lab.
    Logs tool calls, decisions, hypotheses, and results.
    """

    ACTION_TYPES = [
        ("tool_call", "Tool Call"),
        ("hypothesis", "Hypothesis Generated"),
        ("experiment", "Experiment Launched"),
        ("analysis", "Analysis Completed"),
        ("recommendation", "Recommendation Made"),
        ("error", "Error"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lab = models.ForeignKey(
        ResearchLab, on_delete=models.CASCADE, related_name="agent_actions"
    )
    agent_role = models.ForeignKey(
        AgentRole, on_delete=models.CASCADE, related_name="actions"
    )
    action_type = models.CharField(max_length=20, choices=ACTION_TYPES)
    input_data = models.JSONField(default=dict, blank=True)
    output_summary = models.TextField(blank=True)
    tokens_used = models.PositiveIntegerField(default=0)
    cost_cents = models.PositiveIntegerField(default=0)
    success = models.BooleanField(default=True)
    cycle_number = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"[Cycle {self.cycle_number}] {self.agent_role.name}: {self.action_type}"


class ChatRoom(models.Model):
    """
    A real-time collaborative chatroom for a research lab.
    Supports human-to-human, human-to-agent, and agent-to-agent communication.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lab = models.ForeignKey(
        ResearchLab, on_delete=models.CASCADE, related_name="chatrooms"
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    is_private = models.BooleanField(default=False)

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="created_chatrooms",
    )
    members = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name="chatroom_memberships"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} @ {self.lab.name}"


class ChatMessage(models.Model):
    """
    A single message in a chatroom.
    Can be from a human or an agent.
    """

    SENDER_TYPES = [
        ("human", "Human"),
        ("agent", "Agent"),
        ("system", "System"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    room = models.ForeignKey(
        ChatRoom, on_delete=models.CASCADE, related_name="messages"
    )
    sender_type = models.CharField(max_length=10, choices=SENDER_TYPES)
    sender_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="chat_messages",
    )
    sender_agent = models.ForeignKey(
        AgentRole,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="chat_messages",
    )
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        sender = (
            self.sender_user.username if self.sender_user else self.sender_agent.name
        )
        return f"[{sender}]: {self.content[:80]}"


class WorkflowGraph(models.Model):
    """
    A visual workflow designer (like Tangle) for experiment pipelines.
    Stores nodes and edges as JSON, renderable with React Flow or similar.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lab = models.ForeignKey(
        ResearchLab, on_delete=models.CASCADE, related_name="workflows"
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    creator = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="workflows"
    )

    # Graph data (React Flow compatible)
    nodes = models.JSONField(
        default=list, help_text="Array of {id, type, position, data} objects"
    )
    edges = models.JSONField(
        default=list, help_text="Array of {id, source, target} objects"
    )

    is_template = models.BooleanField(default=False)
    usage_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.name


class DiscoverySubmission(models.Model):
    """
    A scientific discovery submitted by a lab for community peer review.
    """

    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("submitted", "Submitted"),
        ("in_review", "In Review"),
        ("accepted", "Accepted"),
        ("rejected", "Rejected"),
        ("featured", "Featured"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lab = models.ForeignKey(
        ResearchLab, on_delete=models.CASCADE, related_name="discoveries"
    )
    title = models.CharField(max_length=500)
    abstract = models.TextField()
    methodology = models.TextField(blank=True)
    results = models.JSONField(default=dict, blank=True)
    reproducibility_score = models.FloatField(null=True, blank=True)

    # Submission metadata
    submitted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="submitted_discoveries",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")

    # Community engagement
    upvotes = models.PositiveIntegerField(default=0)
    downvotes = models.PositiveIntegerField(default=0)
    comment_count = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class DiscoveryReview(models.Model):
    """
    A peer review of a discovery submission.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    submission = models.ForeignKey(
        DiscoverySubmission, on_delete=models.CASCADE, related_name="reviews"
    )
    reviewer = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="discovery_reviews",
    )
    overall_score = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)]
    )
    methodology_score = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        null=True,
        blank=True,
    )
    reproducibility_score = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10)],
        null=True,
        blank=True,
    )
    comments = models.TextField(blank=True)
    is_public = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["submission", "reviewer"]]

    def __str__(self):
        return f"Review by {self.reviewer.username}: {self.overall_score}/10"


class LabInvitation(models.Model):
    """
    Invitation to join a research lab.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lab = models.ForeignKey(
        ResearchLab, on_delete=models.CASCADE, related_name="invitations"
    )
    email = models.EmailField()
    invited_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="sent_lab_invites",
    )
    token = models.CharField(max_length=64, unique=True)
    status = models.CharField(
        max_length=20,
        default="pending",
        choices=[
            ("pending", "Pending"),
            ("accepted", "Accepted"),
            ("expired", "Expired"),
        ],
    )
    expires_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
