import uuid
from django.db import models
from django.conf import settings


class PaperGraphCache(models.Model):
    """
    Cached citation relationships for fast graph traversal.
    """
    paper = models.OneToOneField(
        'agents.ResearchPaper',
        on_delete=models.CASCADE,
        related_name='citation_graph_cache'
    )
    # JSON fields storing lists of paper IDs
    citations = models.JSONField(default=list)      # papers this paper cites
    references = models.JSONField(default=list)     # papers that cite this paper
    co_cited = models.JSONField(default=list)       # papers co-cited with this one
    embedding = models.JSONField(default=list, null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Graph cache for {self.paper.title[:60]}"


class CitationPath(models.Model):
    """
    A saved citation path between two papers.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source_paper = models.ForeignKey(
        'agents.ResearchPaper',
        on_delete=models.CASCADE,
        related_name='paths_from'
    )
    target_paper = models.ForeignKey(
        'agents.ResearchPaper',
        on_delete=models.CASCADE,
        related_name='paths_to'
    )
    # JSON list of paper IDs forming the path
    path = models.JSONField(default=list)
    length = models.PositiveIntegerField(default=0)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Path: {self.source_paper.title[:30]} → {self.target_paper.title[:30]} ({self.length} steps)"


class PaperRecommendation(models.Model):
    """
    A paper recommended to a user.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='paper_recommendations'
    )
    paper = models.ForeignKey(
        'agents.ResearchPaper',
        on_delete=models.CASCADE,
        related_name='recommendations'
    )
    score = models.FloatField(default=0.0)
    reason = models.TextField(blank=True)
    source_type = models.CharField(
        max_length=30,
        choices=[
            ('citation', 'Citation-Based'),
            ('semantic', 'Semantic Similarity'),
            ('co_author', 'Co-Author Network'),
            ('topic', 'Topic Modelling'),
        ],
        default='citation'
    )
    is_dismissed = models.BooleanField(default=False)
    is_saved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['user', 'paper', 'source_type']]
        ordering = ['-score', '-created_at']

    def __str__(self):
        return f"Recommend {self.paper.title[:50]} to {self.user.username}"


class ResearchLandscape(models.Model):
    """
    A saved research landscape (cluster/topic map) for a user.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='research_landscapes'
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    # JSON: { "nodes": [...], "edges": [...] }
    graph_data = models.JSONField(default=dict)
    seed_papers = models.JSONField(default=list)  # IDs of papers used to generate
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Landscape: {self.name} by {self.user.username}"


class PaperFeedItem(models.Model):
    """
    A feed item for the continuous paper feed (Spotify-style).
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='paper_feed'
    )
    paper = models.ForeignKey(
        'agents.ResearchPaper',
        on_delete=models.CASCADE,
        related_name='feed_items'
    )
    seen = models.BooleanField(default=False)
    clicked = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [['user', 'paper']]
        ordering = ['-created_at']

    def __str__(self):
        return f"Feed: {self.paper.title[:50]} → {self.user.username}"