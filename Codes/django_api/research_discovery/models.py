import uuid

from django.conf import settings
from django.db import models


# ═══════════════════════════════════════════════════════════════
# 1. LITERATURE INSIGHTS & SYNTHESIS
# ═══════════════════════════════════════════════════════════════
class LiteratureReview(models.Model):
    """An AI-generated structured literature review."""

    OUTPUT_TYPES = [
        ("report", "Structured Report"),
        ("slides", "Slide Deck"),
        ("infographic", "Infographic"),
        ("audio_summary", "Audio Summary"),
        ("interactive", "Interactive Dashboard"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="literature_reviews",
    )
    title = models.CharField(max_length=300)
    query = models.TextField(help_text="Original search query or research question")
    papers_analyzed = models.PositiveIntegerField(default=0)
    output_type = models.CharField(
        max_length=30, choices=OUTPUT_TYPES, default="report"
    )
    content_markdown = models.TextField(blank=True)
    slide_deck_json = models.JSONField(
        default=dict, blank=True, help_text="Reveal.js / PPTX structure"
    )
    infographic_url = models.URLField(blank=True)
    audio_summary_url = models.URLField(blank=True)
    confidence_score = models.FloatField(default=0.0)
    source_papers = models.JSONField(
        default=list, blank=True, help_text="List of paper IDs and metadata"
    )
    status = models.CharField(
        max_length=20,
        default="queued",
        choices=[
            ("queued", "Queued"),
            ("processing", "Processing"),
            ("completed", "Completed"),
            ("failed", "Failed"),
        ],
    )
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]


class LiteratureSection(models.Model):
    """A section within a literature review."""

    review = models.ForeignKey(
        LiteratureReview, on_delete=models.CASCADE, related_name="sections"
    )
    heading = models.CharField(max_length=300)
    content = models.TextField()
    citations = models.JSONField(default=list, blank=True)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order"]


# ═══════════════════════════════════════════════════════════════
# 2. HYPOTHESIS GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════
class GeneratedHypothesis(models.Model):
    """An AI-generated research hypothesis."""

    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("under_review", "Under Review"),
        ("validated", "Validated"),
        ("rejected", "Rejected"),
        ("published", "Published as Experiment"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="generated_hypotheses",
    )
    title = models.CharField(max_length=500)
    domain = models.CharField(max_length=200, blank=True)
    hypothesis_statement = models.TextField(help_text="The core hypothesis statement")
    supporting_evidence = models.JSONField(
        default=list, help_text="List of papers supporting this hypothesis"
    )
    contradictory_evidence = models.JSONField(default=list)
    novelty_score = models.FloatField(
        default=0.5, help_text="0–1: how novel is this hypothesis?"
    )
    feasibility_score = models.FloatField(default=0.5)
    testability_score = models.FloatField(default=0.5)
    suggested_experiment = models.TextField(
        blank=True, help_text="Suggested experiment to test hypothesis"
    )
    papers_analyzed_count = models.PositiveIntegerField(default=0)
    generated_by_model = models.CharField(max_length=100, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")
    linked_experiment = models.ForeignKey(
        "experiment.TrainingRun", on_delete=models.SET_NULL, null=True, blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-novelty_score", "-created_at"]


class HypothesisSource(models.Model):
    """A source paper that contributed to a hypothesis."""

    hypothesis = models.ForeignKey(
        GeneratedHypothesis, on_delete=models.CASCADE, related_name="sources"
    )
    paper_id = models.CharField(max_length=100)
    paper_title = models.CharField(max_length=500)
    contribution_type = models.CharField(
        max_length=50,
        choices=[
            ("direct_support", "Direct Support"),
            ("partial_support", "Partial Support"),
            ("inspiration", "Inspiration"),
            ("methodology", "Methodology"),
        ],
    )
    relevance_score = models.FloatField(default=0.5)
    citation_snippet = models.TextField(blank=True)


# ═══════════════════════════════════════════════════════════════
# 3. COMPUTATIONAL DISCOVERY & CODE EXPERIMENTATION
# ═══════════════════════════════════════════════════════════════
class CodeExperiment(models.Model):
    """An AI-generated code experiment."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="code_experiments",
    )
    problem_description = models.TextField()
    domain = models.CharField(max_length=200, blank=True)
    generated_code = models.TextField(blank=True)
    language = models.CharField(max_length=30, default="python")
    execution_output = models.TextField(blank=True)
    execution_success = models.BooleanField(null=True)
    iteration_count = models.PositiveIntegerField(default=0)
    max_iterations = models.PositiveIntegerField(default=5)
    error_log = models.JSONField(default=list, blank=True)
    status = models.CharField(max_length=20, default="queued")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]


class CodeIteration(models.Model):
    """A single iteration of code generation."""

    experiment = models.ForeignKey(
        CodeExperiment, on_delete=models.CASCADE, related_name="iterations"
    )
    code = models.TextField()
    output = models.TextField(blank=True)
    error = models.TextField(blank=True)
    success = models.BooleanField(default=False)
    iteration_number = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)


# ═══════════════════════════════════════════════════════════════
# 4. INTEGRATED RESEARCH DATABASE HUB
# ═══════════════════════════════════════════════════════════════
class LifeScienceDatabase(models.Model):
    """A registered life science database."""

    CATEGORIES = [
        ("protein", "Protein Structure & Sequence"),
        ("gene", "Genomics & Genetics"),
        ("pathway", "Pathways & Networks"),
        ("drug", "Drug & Compound"),
        ("disease", "Disease & Phenotype"),
        ("literature", "Literature & Citations"),
        ("clinical", "Clinical Trials"),
        ("ontology", "Ontology & Terminology"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    abbreviation = models.CharField(max_length=20, blank=True)
    category = models.CharField(max_length=30, choices=CATEGORIES)
    base_url = models.URLField()
    api_endpoint = models.URLField(blank=True)
    api_documentation_url = models.URLField(blank=True)
    requires_api_key = models.BooleanField(default=False)
    api_key_env = models.CharField(max_length=100, blank=True)
    rate_limit_per_minute = models.PositiveIntegerField(default=60)
    is_active = models.BooleanField(default=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class DatabaseQuery(models.Model):
    """A recorded query to a life science database."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="db_queries"
    )
    database = models.ForeignKey(LifeScienceDatabase, on_delete=models.CASCADE)
    query_params = models.JSONField(default=dict)
    response_summary = models.TextField(blank=True)
    raw_response = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=20, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)


class ExperimentDatabaseLink(models.Model):
    """Links an experiment to relevant database entries."""

    experiment = models.ForeignKey(
        "experiment.TrainingRun", on_delete=models.CASCADE, related_name="db_links"
    )
    database = models.ForeignKey(LifeScienceDatabase, on_delete=models.CASCADE)
    entry_id = models.CharField(max_length=100)
    entry_label = models.CharField(max_length=300, blank=True)
    relevance_score = models.FloatField(default=0.5)
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


# ═══════════════════════════════════════════════════════════════
# 5. RESEARCH GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════
class ResearchGapAnalysis(models.Model):
    """An AI-generated research gap analysis."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="gap_analyses"
    )
    title = models.CharField(max_length=300)
    domain = models.CharField(max_length=200, blank=True)
    papers_compared = models.JSONField(
        default=list, help_text="List of paper IDs compared"
    )
    explicit_gaps = models.JSONField(
        default=list, help_text="Gaps explicitly stated in literature"
    )
    implicit_gaps = models.JSONField(
        default=list, help_text="Gaps inferred by AI analysis"
    )
    methodology_gaps = models.JSONField(default=list)
    suggested_future_work = models.JSONField(default=list)
    gap_map_data = models.JSONField(
        default=dict, help_text="2D visualization data for gap map"
    )
    status = models.CharField(max_length=20, default="queued")
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]


class SideBySideComparison(models.Model):
    """A side-by-side comparison of research outputs."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="comparisons"
    )
    title = models.CharField(max_length=300)
    items = models.JSONField(
        default=list, help_text="List of compared items with metadata"
    )
    comparison_dimensions = models.JSONField(default=list)
    similarity_matrix = models.JSONField(default=dict, blank=True)
    differences_highlighted = models.JSONField(default=list)
    ai_analysis = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


# ═══════════════════════════════════════════════════════════════
# 6. GRANT DISCOVERY & PROPOSAL MATCHING
# ═══════════════════════════════════════════════════════════════
class GrantOpportunity(models.Model):
    """A discovered grant opportunity."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    source = models.CharField(max_length=100, help_text="e.g., NIH, NSF, ERC, Wellcome")
    source_id = models.CharField(max_length=200, blank=True)
    title = models.CharField(max_length=500)
    description = models.TextField(blank=True)
    funding_amount = models.CharField(max_length=200, blank=True)
    deadline = models.DateField(null=True, blank=True)
    eligibility_criteria = models.TextField(blank=True)
    url = models.URLField(blank=True)
    keywords = models.JSONField(default=list, blank=True)
    is_active = models.BooleanField(default=True)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["deadline"]


class GrantMatch(models.Model):
    """A match between a researcher and a grant opportunity."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="grant_matches"
    )
    grant = models.ForeignKey(GrantOpportunity, on_delete=models.CASCADE)
    match_score = models.FloatField(default=0.0, help_text="0–1 match score")
    match_reasons = models.JSONField(
        default=list, help_text="Why this grant matches the researcher"
    )
    suggested_proposal_title = models.CharField(max_length=500, blank=True)
    suggested_proposal_outline = models.TextField(blank=True)
    is_saved = models.BooleanField(default=False)
    is_applied = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)


class ResearcherGrantProfile(models.Model):
    """A researcher's profile for grant matching."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="grant_profile"
    )
    research_keywords = models.JSONField(default=list)
    publications = models.JSONField(default=list, blank=True)
    past_grants = models.JSONField(default=list, blank=True)
    institution = models.CharField(max_length=300, blank=True)
    career_stage = models.CharField(
        max_length=50,
        blank=True,
        choices=[
            ("early_career", "Early Career"),
            ("mid_career", "Mid Career"),
            ("senior", "Senior"),
            ("student", "Student"),
        ],
    )
    funding_preferences = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
