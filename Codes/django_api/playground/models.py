import uuid

from django.conf import settings
from django.db import models


class PromptTemplate(models.Model):
    """
    A versioned prompt template with variable placeholders.
    Inspired by Langfuse prompt management.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    content = models.TextField(help_text="Prompt text with {{variable}} placeholders")
    variables_schema = models.JSONField(
        default=dict,
        blank=True,
        help_text='{"variable_name": {"type": "string", "description": "..."}}',
    )
    tags = models.JSONField(default=list, blank=True)
    is_public = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="prompt_templates",
    )
    version = models.PositiveIntegerField(default=1)
    parent = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="versions",
    )
    usage_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-usage_count", "-updated_at"]

    def __str__(self):
        return f"{self.name} (v{self.version})"

    def render(self, variables: dict) -> str:
        """Render the template with provided variables."""
        rendered = self.content
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered

    def create_new_version(self, content: str, variables_schema: dict = None):
        """Create a new version of this prompt (fork)."""
        new_version = PromptTemplate.objects.create(
            name=self.name,
            description=self.description,
            content=content,
            variables_schema=variables_schema or self.variables_schema,
            tags=self.tags,
            is_public=self.is_public,
            created_by=self.created_by,
            version=self.version + 1,
            parent=self,
        )
        return new_version


class ModelConfig(models.Model):
    """
    A saved model configuration (temperature, top-p, model name, etc.).
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="model_configs"
    )
    provider = models.CharField(
        max_length=50,
        default="openai",
        choices=[
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
            ("google", "Google"),
            ("local", "Local"),
            ("huggingface", "HuggingFace"),
        ],
    )
    model_name = models.CharField(max_length=200, default="gpt-4o")
    temperature = models.FloatField(default=0.7)
    top_p = models.FloatField(default=1.0)
    max_tokens = models.PositiveIntegerField(default=2048)
    frequency_penalty = models.FloatField(default=0.0)
    presence_penalty = models.FloatField(default=0.0)
    system_prompt = models.TextField(blank=True)
    extra_params = models.JSONField(default=dict, blank=True)
    is_default = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_default", "-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.provider}/{self.model_name})"


class PlaygroundSession(models.Model):
    """
    A single playground session – one conversation with the LLM.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="playground_sessions",
    )
    name = models.CharField(max_length=200, blank=True)
    model_config = models.ForeignKey(
        ModelConfig, on_delete=models.SET_NULL, null=True, blank=True
    )
    prompt_template = models.ForeignKey(
        PromptTemplate, on_delete=models.SET_NULL, null=True, blank=True
    )
    variables = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name or self.id.hex[:8]} ({self.user.username})"


class PlaygroundMessage(models.Model):
    """
    A single message in a playground session.
    """

    ROLE_CHOICES = [
        ("system", "System"),
        ("user", "User"),
        ("assistant", "Assistant"),
        ("tool", "Tool Output"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        PlaygroundSession, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField(blank=True)
    tool_calls = models.JSONField(
        default=list, blank=True, help_text="Array of tool call objects from the LLM"
    )
    token_count = models.PositiveIntegerField(default=0)
    latency_ms = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.role}] {self.content[:80]}"


class ComparisonSet(models.Model):
    """
    A set of side-by-side comparisons of multiple prompts/configs.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="comparison_sets",
    )
    name = models.CharField(max_length=200)
    input_text = models.TextField(help_text="The fixed input used for all comparisons")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name


class ComparisonResult(models.Model):
    """
    A single output in a comparison set.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    comparison_set = models.ForeignKey(
        ComparisonSet, on_delete=models.CASCADE, related_name="results"
    )
    model_config = models.ForeignKey(ModelConfig, on_delete=models.SET_NULL, null=True)
    prompt_template = models.ForeignKey(
        PromptTemplate, on_delete=models.SET_NULL, null=True
    )
    output_text = models.TextField(blank=True)
    latency_ms = models.PositiveIntegerField(default=0)
    token_count = models.PositiveIntegerField(default=0)
    cost_cents = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["comparison_set", "model_config__name"]

    def __str__(self):
        return f"Result: {self.model_config.name if self.model_config else 'Unknown'}"


class ToolDefinition(models.Model):
    """
    A tool definition that can be called by the LLM.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField()
    parameters_schema = models.JSONField(
        default=dict, help_text="JSON Schema for the tool's parameters"
    )
    implementation = models.TextField(
        blank=True, help_text="Python code or API endpoint to execute this tool"
    )
    is_active = models.BooleanField(default=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="tool_definitions",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

    def get_openai_format(self):
        """Return the tool definition in OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
