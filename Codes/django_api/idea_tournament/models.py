import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone


class IdeaTournament(models.Model):
    """
    A tournament that generates and refines research hypotheses.
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='idea_tournaments'
    )
    research_domain = models.CharField(max_length=300, help_text="e.g., 'NLP model compression'")
    research_question = models.TextField(help_text="Open-ended question to explore")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    # Tournament config
    num_hypotheses = models.PositiveIntegerField(default=10, help_text="Initial hypotheses to generate")
    num_rounds = models.PositiveIntegerField(default=3, help_text="Number of debate/refinement rounds")
    top_k = models.PositiveIntegerField(default=3, help_text="Top hypotheses to retain after each round")

    # Stats
    total_hypotheses_generated = models.PositiveIntegerField(default=0)
    current_round = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Tournament: {self.research_domain} ({self.status})"


class Hypothesis(models.Model):
    """
    A single research hypothesis within a tournament.
    """
    STATUS_CHOICES = [
        ('proposed', 'Proposed'),
        ('shortlisted', 'Shortlisted'),
        ('eliminated', 'Eliminated'),
        ('winner', 'Winner'),
    ]
    ROUND_CHOICES = [(i, f'Round {i}') for i in range(1, 6)]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tournament = models.ForeignKey(
        IdeaTournament,
        on_delete=models.CASCADE,
        related_name='hypotheses'
    )
    title = models.CharField(max_length=500)
    content = models.TextField(help_text="Full hypothesis description")
    round_created = models.PositiveIntegerField(default=1, choices=ROUND_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='proposed')
    score = models.FloatField(default=0.0, help_text="Overall score from reviews and ranking")
    rank = models.PositiveIntegerField(null=True, blank=True)

    # Source agents
    generated_by = models.CharField(
        max_length=50,
        choices=[
            ('generation', 'Generation Agent'),
            ('evolution', 'Evolution Agent'),
            ('reflection', 'Reflection Agent'),
        ],
        default='generation'
    )
    parent_hypothesis = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='evolved_children'
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-score', 'created_at']

    def __str__(self):
        return self.title[:80]


class HypothesisReview(models.Model):
    """
    A review / critique of a hypothesis by an AI agent.
    """
    AGENT_CHOICES = [
        ('proximity', 'Proximity Agent'),
        ('reflection', 'Reflection Agent'),
        ('ranking', 'Ranking Agent'),
        ('meta_review', 'Meta-Review Agent'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    hypothesis = models.ForeignKey(
        Hypothesis,
        on_delete=models.CASCADE,
        related_name='reviews'
    )
    agent = models.CharField(max_length=30, choices=AGENT_CHOICES)
    round_number = models.PositiveIntegerField(default=1)

    # Review content
    score = models.FloatField(default=0.0, help_text="Score assigned by this agent (0–10)")
    strengths = models.TextField(blank=True)
    weaknesses = models.TextField(blank=True)
    suggestions = models.TextField(blank=True)
    overall_assessment = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['hypothesis', 'round_number']

    def __str__(self):
        return f"Review by {self.agent} on {self.hypothesis.title[:40]}"


class LiteratureCitation(models.Model):
    """
    A scientific paper cited in support of a hypothesis.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    hypothesis = models.ForeignKey(
        Hypothesis,
        on_delete=models.CASCADE,
        related_name='citations'
    )
    paper = models.ForeignKey(
        'agents.ResearchPaper',
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='hypothesis_citations'
    )
    title = models.CharField(max_length=500)
    authors = models.JSONField(default=list, blank=True)
    year = models.PositiveIntegerField(null=True, blank=True)
    url = models.URLField(blank=True, max_length=500)
    snippet = models.TextField(blank=True, help_text="Relevant excerpt from the paper")
    relevance_score = models.FloatField(default=1.0)

    class Meta:
        ordering = ['-relevance_score']

    def __str__(self):
        return self.title[:100]


class TournamentLog(models.Model):
    """
    A log entry recording each step in the tournament process.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    tournament = models.ForeignKey(
        IdeaTournament,
        on_delete=models.CASCADE,
        related_name='logs'
    )
    round_number = models.PositiveIntegerField(default=1)
    agent = models.CharField(max_length=50)
    action = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['round_number', 'created_at']

    def __str__(self):
        return f"[Round {self.round_number}] {self.agent}: {self.action}"