from django.contrib import admin
from .models import *

for model in [
    LLMProvider, LLMModel, PromptTemplate, PromptVersion,
    PlaygroundSession, PlaygroundRun, ToolDefinition,
    StructuredOutputSchema, EvalSuite, EvalTestCase,
    EvalRun, EvalResult, MemoryEntry, UsageRecord,
]:
    admin.site.register(model)