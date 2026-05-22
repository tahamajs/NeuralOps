import uuid
from django.db import models
from django.conf import settings

class ChatSession(models.Model):
    """A single conversation between user and agent."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='agent_chats')
    title = models.CharField(max_length=300, blank=True)
    status = models.CharField(max_length=20, default='active', choices=[
        ('active', 'Active'), ('completed', 'Completed'), ('archived', 'Archived')
    ])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta: ordering = ['-updated_at']

class ChatMessage(models.Model):
    """A single exchange in the chat."""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('tool', 'Tool Result'),
        ('system', 'System'),
        ('experiment_card', 'Experiment Card'),  # rich UI card
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=30, choices=ROLE_CHOICES)
    content = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta: ordering = ['created_at']


import uuid
from django.db import models
from django.conf import settings

class AgentSession(models.Model):
    """A long‑running autonomous agent session."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='agent_sessions')
    name = models.CharField(max_length=300)
    objective = models.TextField(help_text="Natural language research goal, e.g. 'Improve MRPC accuracy above 0.89'")
    
    # Agent configuration
    agent_type = models.CharField(max_length=50, default='deep_researcher',
        choices=[('deep_researcher', 'Deep Researcher'), ('hpo_agent', 'HPO Agent'), ('code_agent', 'Code Optimizer')])
    max_cycles = models.IntegerField(default=100)
    max_llm_cost_cents = models.IntegerField(default=500)  # 5 $ budget
    allowed_tools = models.JSONField(default=list, blank=True)
    
    # Status
    status = models.CharField(max_length=20, default='active',
        choices=[('active', 'Active'), ('paused', 'Paused'), ('review', 'Awaiting Review'), ('completed', 'Completed'), ('error', 'Error')])
    
    # Project context
    project = models.ForeignKey('experiment.Experiment', on_delete=models.SET_NULL, null=True, blank=True)
    codebase_files = models.JSONField(default=list, blank=True, help_text="List of files the agent can read/modify")
    metric_name = models.CharField(max_length=100, default='eval_accuracy')
    metric_direction = models.CharField(max_length=10, default='maximize')
    
    # Stats
    cycles_completed = models.IntegerField(default=0)
    experiments_run = models.IntegerField(default=0)
    best_metric_value = models.FloatField(null=True, blank=True)
    total_llm_cost_cents = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Agent: {self.name} ({self.status})"


class AgentMemory(models.Model):
    """Two‑tier constant‑size memory (~5K characters cap)"""
    session = models.OneToOneField(AgentSession, on_delete=models.CASCADE, related_name='memory')
    
    # Tier 1: Short‑term working memory (current cycle context)
    working_memory = models.TextField(blank=True, max_length=3000)
    
    # Tier 2: Long‑term summary (compressed knowledge across cycles)
    long_term_summary = models.TextField(blank=True, max_length=2000)
    key_findings = models.JSONField(default=list, blank=True)
    
    updated_at = models.DateTimeField(auto_now=True)


class AgentAction(models.Model):
    """Each tool call or decision made by the agent."""
    session = models.ForeignKey(AgentSession, on_delete=models.CASCADE, related_name='actions')
    cycle = models.IntegerField()
    action_type = models.CharField(max_length=100)
    tool_name = models.CharField(max_length=100, blank=True)
    input_data = models.JSONField(default=dict, blank=True)
    output_summary = models.TextField(blank=True)
    llm_tokens_used = models.IntegerField(default=0)
    llm_cost_cents = models.FloatField(default=0)
    success = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']


class ResearchPaper(models.Model):
    """A saved research paper discovered by the agent."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=500)
    authors = models.JSONField(default=list, blank=True)
    abstract = models.TextField(blank=True)
    url = models.URLField(max_length=1000, blank=True)
    source = models.CharField(max_length=50, default='arxiv')  # arxiv, semantic_scholar, manual
    source_id = models.CharField(max_length=100, blank=True)
    year = models.IntegerField(null=True, blank=True)
    citation_count = models.IntegerField(default=0)
    pdf_url = models.URLField(max_length=1000, blank=True)
    keywords = models.JSONField(default=list, blank=True)
    notes = models.TextField(blank=True)
    tags = models.JSONField(default=list, blank=True)
    saved_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='saved_papers')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = [['source', 'source_id']]


class CitationEdge(models.Model):
    """Edge in the citation graph."""
    source_paper = models.ForeignKey(ResearchPaper, on_delete=models.CASCADE, related_name='citations_out')
    target_paper = models.ForeignKey(ResearchPaper, on_delete=models.CASCADE, related_name='citations_in')
    relationship = models.CharField(max_length=50, default='cites')  # cites, references, related
    
    class Meta:
        unique_together = [['source_paper', 'target_paper']]


class TrainingAnalysis(models.Model):
    """Analysis result for a training run."""
    run = models.OneToOneField('experiment.TrainingRun', on_delete=models.CASCADE, related_name='analysis')
    status = models.CharField(max_length=50)  # healthy, overfitting, underfitting, plateau, diverging
    confidence = models.FloatField(default=0.0)
    suggestions = models.JSONField(default=list, blank=True)
    metrics_summary = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)



# agents/models.py — add these

class ResearchPlan(models.Model):
    """A structured research plan with objectives and steps."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=300)
    objective = models.TextField()
    steps = models.JSONField(default=list)  # list of {"task": "...", "status": "...", "findings": "..."}
    status = models.CharField(max_length=20, default='draft',
        choices=[('draft','Draft'),('in_progress','In Progress'),('completed','Completed')])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class ResearchFinding(models.Model):
    """A sourced finding with credibility score."""
    plan = models.ForeignKey(ResearchPlan, on_delete=models.CASCADE, related_name='findings')
    source_url = models.URLField()
    source_title = models.CharField(max_length=300)
    source_type = models.CharField(max_length=50)  # 'arxiv', 'semantic_scholar', 'web'
    credibility_score = models.FloatField(default=0.5)  # 0-1
    summary = models.TextField()
    relevance_score = models.FloatField(default=0.5)
    created_at = models.DateTimeField(auto_now_add=True)

class ResearchReport(models.Model):
    """A synthesized research report."""
    plan = models.OneToOneField(ResearchPlan, on_delete=models.CASCADE)
    content_markdown = models.TextField()
    references_json = models.JSONField(default=list)
    generated_at = models.DateTimeField(auto_now_add=True)