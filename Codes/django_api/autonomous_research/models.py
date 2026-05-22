import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone


class ResearchPipeline(models.Model):
    """
    A complete autonomous research pipeline from idea to publication.
    Tracks the entire lifecycle: literature review → hypothesis → experiment → manuscript.
    """
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('literature_review', 'Literature Review'),
        ('hypothesis_generated', 'Hypothesis Generated'),
        ('experimenting', 'Experimenting'),
        ('manuscript_draft', 'Manuscript Draft'),
        ('completed', 'Completed'),
        ('paused', 'Paused'),
        ('failed', 'Failed'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='research_pipelines'
    )
    name = models.CharField(max_length=300)
    description = models.TextField(blank=True)
    research_question = models.TextField(
        help_text="The core research question to investigate"
    )
    status = models.CharField(max_length=30, choices=STATUS_CHOICES, default='draft')
    progress = models.FloatField(default=0.0, help_text="0–100%")
    auto_mode = models.BooleanField(
        default=True,
        help_text="If True, the pipeline runs autonomously without human intervention"
    )

    # Budget & constraints
    max_compute_hours = models.PositiveIntegerField(default=100)
    max_llm_cost_cents = models.PositiveIntegerField(default=500)
    compute_hours_used = models.FloatField(default=0.0)
    llm_cost_cents_used = models.FloatField(default=0.0)

    # Results
    final_hypothesis = models.TextField(blank=True)
    final_manuscript = models.TextField(blank=True)
    final_metrics = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.status})"


class PipelineStep(models.Model):
    """
    A single step in the research pipeline.
    Each step is an action the AI takes: search papers, generate hypothesis, run experiment, etc.
    """
    STEP_TYPES = [
        ('literature_search', 'Literature Search'),
        ('paper_analysis', 'Paper Analysis'),
        ('hypothesis_generation', 'Hypothesis Generation'),
        ('experiment_design', 'Experiment Design'),
        ('experiment_execution', 'Experiment Execution'),
        ('result_analysis', 'Result Analysis'),
        ('manuscript_section', 'Manuscript Section'),
        ('manuscript_revision', 'Manuscript Revision'),
        ('human_review', 'Human Review'),
        ('decision', 'Decision Point'),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('skipped', 'Skipped'),
        ('awaiting_human', 'Awaiting Human'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.ForeignKey(
        ResearchPipeline,
        on_delete=models.CASCADE,
        related_name='steps'
    )
    step_type = models.CharField(max_length=30, choices=STEP_TYPES)
    order = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    title = models.CharField(max_length=300)
    description = models.TextField(blank=True)

    # Input / Output
    input_data = models.JSONField(default=dict, blank=True)
    output_data = models.JSONField(default=dict, blank=True)

    # LLM interaction
    llm_prompt = models.TextField(blank=True)
    llm_response = models.TextField(blank=True)
    llm_model = models.CharField(max_length=100, blank=True)
    llm_tokens_used = models.PositiveIntegerField(default=0)

    # Execution
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['pipeline', 'order']

    def __str__(self):
        return f"Step {self.order}: {self.title} ({self.status})"


class AutonomousDecision(models.Model):
    """
    A decision made by the AI during the research pipeline.
    Tracks why the AI chose a particular path.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.ForeignKey(
        ResearchPipeline,
        on_delete=models.CASCADE,
        related_name='decisions'
    )
    step = models.ForeignKey(
        PipelineStep,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='decisions'
    )
    decision_type = models.CharField(max_length=100)
    reasoning = models.TextField()
    alternatives = models.JSONField(default=list, blank=True)
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Decision: {self.decision_type} (confidence: {self.confidence:.2f})"


class PipelineArtifact(models.Model):
    """
    An artifact generated by the pipeline: paper summary, hypothesis, experiment config, etc.
    """
    ARTIFACT_TYPES = [
        ('literature_summary', 'Literature Summary'),
        ('hypothesis', 'Hypothesis'),
        ('experiment_config', 'Experiment Configuration'),
        ('experiment_results', 'Experiment Results'),
        ('analysis_report', 'Analysis Report'),
        ('manuscript_section', 'Manuscript Section'),
        ('full_manuscript', 'Full Manuscript'),
        ('code', 'Code / Script'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    pipeline = models.ForeignKey(
        ResearchPipeline,
        on_delete=models.CASCADE,
        related_name='artifacts'
    )
    step = models.ForeignKey(
        PipelineStep,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='artifacts'
    )
    artifact_type = models.CharField(max_length=30, choices=ARTIFACT_TYPES)
    title = models.CharField(max_length=300)
    content = models.TextField(blank=True)
    file = models.FileField(upload_to='pipeline_artifacts/', blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.artifact_type}: {self.title}"


class PipelineTemplate(models.Model):
    """
    A reusable pipeline template for common research workflows.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    category = models.CharField(max_length=100, blank=True)
    steps_config = models.JSONField(
        default=list,
        help_text="List of step definitions: [{'step_type': '...', 'title': '...', 'description': '...'}, ...]"
    )
    is_public = models.BooleanField(default=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        related_name='pipeline_templates'
    )
    usage_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-usage_count']

    def __str__(self):
        return self.name