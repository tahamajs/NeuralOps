import uuid
from django.db import models
from django.conf import settings


class ResearchSession(models.Model):
    """A single research workflow session."""
    STATUS_CHOICES = [
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='research_sessions'
    )
    research_goal = models.TextField()
    model_name = models.CharField(max_length=100, default='gpt-4o-mini')
    max_iterations = models.PositiveIntegerField(default=3)
    hypotheses_per_generation = models.PositiveIntegerField(default=10)
    tournament_size = models.PositiveIntegerField(default=8)
    evolution_top_k = models.PositiveIntegerField(default=3)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    final_hypothesis_id = models.UUIDField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Research: {self.research_goal[:80]}"


class Hypothesis(models.Model):
    """A single hypothesis generated during the research workflow."""
    STATUS_CHOICES = [
        ('proposed', 'Proposed'),
        ('refined', 'Refined'),
        ('shortlisted', 'Shortlisted'),
        ('finalist', 'Finalist'),
        ('rejected', 'Rejected'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        ResearchSession, on_delete=models.CASCADE, related_name='hypotheses'
    )
    generation = models.PositiveIntegerField(default=1, help_text="Which iteration produced this")
    title = models.CharField(max_length=300)
    content = models.TextField()
    # Elo rating
    elo_score = models.FloatField(default=1000.0)
    # How many pairwise comparisons it has participated in
    comparisons_count = models.PositiveIntegerField(default=0)
    # Whether it survived to the next generation
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='proposed')
    parent_hypothesis = models.ForeignKey(
        'self', on_delete=models.SET_NULL, null=True, blank=True,
        related_name='children'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-elo_score']

    def __str__(self):
        return self.title[:80]


class HypothesisReview(models.Model):
    """A peer review of a hypothesis."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    hypothesis = models.ForeignKey(
        Hypothesis, on_delete=models.CASCADE, related_name='reviews'
    )
    agent = models.CharField(max_length=50, help_text="Which agent wrote this review")
    strengths = models.TextField(blank=True)
    weaknesses = models.TextField(blank=True)
    score = models.PositiveIntegerField(default=5, help_text="1–10")
    suggestions = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Review by {self.agent} on {self.hypothesis.title[:50]}"


class MetaReview(models.Model):
    """Synthesis of all reviews for a hypothesis."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    hypothesis = models.OneToOneField(
        Hypothesis, on_delete=models.CASCADE, related_name='meta_review'
    )
    synthesis = models.TextField()
    final_score = models.FloatField()
    recommendation = models.CharField(max_length=50, choices=[
        ('accept', 'Accept'),
        ('revise', 'Revise'),
        ('reject', 'Reject'),
    ])
    created_at = models.DateTimeField(auto_now_add=True)


class TournamentMatch(models.Model):
    """A pairwise comparison between two hypotheses."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        ResearchSession, on_delete=models.CASCADE, related_name='tournament_matches'
    )
    generation = models.PositiveIntegerField()
    hypothesis_a = models.ForeignKey(
        Hypothesis, on_delete=models.CASCADE, related_name='matches_as_a'
    )
    hypothesis_b = models.ForeignKey(
        Hypothesis, on_delete=models.CASCADE, related_name='matches_as_b'
    )
    winner = models.ForeignKey(
        Hypothesis, on_delete=models.CASCADE, related_name='matches_won', null=True
    )
    reasoning = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Match {self.hypothesis_a.title[:30]} vs {self.hypothesis_b.title[:30]}"