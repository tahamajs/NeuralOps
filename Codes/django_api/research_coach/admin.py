from django.contrib import admin

from .models import (
    AnalysisResult,
    ChatMessage,
    ChatSession,
    ExperimentMemory,
    Recommendation,
)

admin.site.register(AnalysisResult)
admin.site.register(Recommendation)
admin.site.register(ChatSession)
admin.site.register(ChatMessage)
admin.site.register(ExperimentMemory)
