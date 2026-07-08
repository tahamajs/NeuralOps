import uuid

from django.conf import settings
from django.db import models


# ═══════════════════════════════════════════════════════════════
# 1. LLM Providers & Models
# ═══════════════════════════════════════════════════════════════
class LLMProvider(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=50, unique=True)
    api_base_url = models.URLField(blank=True)
    api_key_env = models.CharField(
        max_length=100, help_text="Environment variable for API key"
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class LLMModel(models.Model):
    provider = models.ForeignKey(
        LLMProvider, on_delete=models.CASCADE, related_name="models"
    )
    model_id = models.CharField(max_length=100, unique=True, help_text="e.g. 'gpt-4o'")
    display_name = models.CharField(max_length=200)
    context_window = models.PositiveIntegerField(default=4096)
    max_output_tokens = models.PositiveIntegerField(default=1024)
    supports_vision = models.BooleanField(default=False)
    supports_tools = models.BooleanField(default=True)
    supports_json_mode = models.BooleanField(default=True)
    pricing_input_per_1k = models.DecimalField(
        max_digits=10, decimal_places=6, default=0
    )
    pricing_output_per_1k = models.DecimalField(
        max_digits=10, decimal_places=6, default=0
    )
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.display_name


# ═══════════════════════════════════════════════════════════════
# 2. Prompt Management & Versioning
# ═══════════════════════════════════════════════════════════════
class PromptTemplate(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="prompts"
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    current_version = models.PositiveIntegerField(default=1)
    tags = models.JSONField(default=list, blank=True)
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class PromptVersion(models.Model):
    prompt = models.ForeignKey(
        PromptTemplate, on_delete=models.CASCADE, related_name="versions"
    )
    version_number = models.PositiveIntegerField()
    content = models.TextField(help_text="Prompt text with {{variable}} placeholders")
    model_config = models.JSONField(
        default=dict, blank=True, help_text="Default temperature, top_p, etc."
    )
    change_notes = models.TextField(blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["prompt", "version_number"]]


# ═══════════════════════════════════════════════════════════════
# 3. Playground & Side‑by‑Side Comparison
# ═══════════════════════════════════════════════════════════════
class PlaygroundSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="playground_sessions",
    )
    name = models.CharField(max_length=200, blank=True)
    input_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class PlaygroundRun(models.Model):
    session = models.ForeignKey(
        PlaygroundSession, on_delete=models.CASCADE, related_name="runs"
    )
    model = models.ForeignKey(LLMModel, on_delete=models.CASCADE)
    prompt_content = models.TextField()
    temperature = models.FloatField(default=0.7)
    top_p = models.FloatField(default=1.0)
    max_tokens = models.PositiveIntegerField(default=1024)
    response_content = models.TextField(blank=True)
    response_json = models.JSONField(default=dict, blank=True)
    tool_calls = models.JSONField(default=list, blank=True)
    tokens_input = models.PositiveIntegerField(default=0)
    tokens_output = models.PositiveIntegerField(default=0)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    latency_ms = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)


# ═══════════════════════════════════════════════════════════════
# 4. Tool Calling & Structured Output
# ═══════════════════════════════════════════════════════════════
class ToolDefinition(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="tool_definitions",
    )
    name = models.CharField(max_length=100)
    description = models.TextField()
    parameters_schema = models.JSONField(default=dict)
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)


class StructuredOutputSchema(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="output_schemas",
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    json_schema = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)


# ═══════════════════════════════════════════════════════════════
# 5. Evaluation Framework
# ═══════════════════════════════════════════════════════════════
class EvalSuite(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="eval_suites"
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class EvalTestCase(models.Model):
    suite = models.ForeignKey(
        EvalSuite, on_delete=models.CASCADE, related_name="test_cases"
    )
    input_text = models.TextField()
    expected_output = models.TextField(blank=True)
    evaluation_type = models.CharField(
        max_length=30,
        default="exact_match",
        choices=[
            ("exact_match", "Exact Match"),
            ("contains", "Contains"),
            ("semantic_similarity", "Semantic Similarity"),
            ("llm_judge", "LLM Judge"),
            ("tool_call_correct", "Tool Call Correctness"),
        ],
    )
    metadata = models.JSONField(default=dict, blank=True)


class EvalRun(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    suite = models.ForeignKey(EvalSuite, on_delete=models.CASCADE, related_name="runs")
    model = models.ForeignKey(LLMModel, on_delete=models.CASCADE)
    prompt = models.ForeignKey(
        PromptVersion, on_delete=models.SET_NULL, null=True, blank=True
    )
    status = models.CharField(max_length=20, default="queued")
    overall_score = models.FloatField(null=True)
    started_at = models.DateTimeField(null=True)
    completed_at = models.DateTimeField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)


class EvalResult(models.Model):
    run = models.ForeignKey(EvalRun, on_delete=models.CASCADE, related_name="results")
    test_case = models.ForeignKey(EvalTestCase, on_delete=models.CASCADE)
    model_output = models.TextField(blank=True)
    score = models.FloatField(null=True)
    passed = models.BooleanField(null=True)
    details = models.JSONField(default=dict, blank=True)


# ═══════════════════════════════════════════════════════════════
# 6. Smart Memory System
# ═══════════════════════════════════════════════════════════════
class MemoryEntry(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="llm_memories"
    )
    project = models.ForeignKey(
        "experiment.Experiment", on_delete=models.CASCADE, null=True, blank=True
    )
    key = models.CharField(max_length=200)
    value = models.TextField()
    source = models.CharField(max_length=50, default="manual")
    importance = models.FloatField(default=0.5)
    last_accessed_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [["user", "project", "key"]]


# ═══════════════════════════════════════════════════════════════
# 7. Cost & Usage Tracking
# ═══════════════════════════════════════════════════════════════
class UsageRecord(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="llm_usage"
    )
    model = models.ForeignKey(LLMModel, on_delete=models.CASCADE)
    tokens_input = models.PositiveIntegerField(default=0)
    tokens_output = models.PositiveIntegerField(default=0)
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6, default=0)
    source = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class ArenaBattle(models.Model):
    """A single battle between two anonymous models."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model_a = models.ForeignKey(
        LLMModel, on_delete=models.CASCADE, related_name="battles_as_a"
    )
    model_b = models.ForeignKey(
        LLMModel, on_delete=models.CASCADE, related_name="battles_as_b"
    )
    prompt = models.TextField()
    response_a = models.TextField()
    response_b = models.TextField()
    # The models are presented anonymously; the user picks a winner.
    created_at = models.DateTimeField(auto_now_add=True)


class ArenaVote(models.Model):
    """A user's vote in a battle."""

    battle = models.ForeignKey(
        ArenaBattle, on_delete=models.CASCADE, related_name="votes"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True, blank=True
    )
    winner = models.CharField(
        max_length=1, choices=[("A", "Model A"), ("B", "Model B"), ("tie", "Tie")]
    )
    created_at = models.DateTimeField(auto_now_add=True)


class ModelElo(models.Model):
    """ELO rating for a model in the arena."""

    model = models.OneToOneField(LLMModel, on_delete=models.CASCADE, related_name="elo")
    rating = models.FloatField(default=1500.0)
    matches_played = models.PositiveIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)


class Guardrail(models.Model):
    """A pre-built guardrail rule."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    guardrail_type = models.CharField(
        max_length=50,
        choices=[
            ("toxicity_filter", "Toxicity Filter"),
            ("pii_redaction", "PII Redaction"),
            ("keyword_block", "Keyword Block"),
            ("custom_regex", "Custom Regex"),
        ],
    )
    config = models.JSONField(default=dict, help_text="Configuration for the guardrail")
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)


class GuardrailAttachment(models.Model):
    """Attaches one or more guardrails to a playground session or inference endpoint."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    guardrail = models.ForeignKey(Guardrail, on_delete=models.CASCADE)
    # Polymorphic: can attach to a session or model
    session = models.ForeignKey(
        PlaygroundSession,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="guardrails",
    )
    model = models.ForeignKey(
        LLMModel,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="guardrails",
    )
