from django.contrib import admin
from .models import AnalysisResult, Recommendation, ChatSession, ChatMessage, ExperimentMemory

admin.site.register(AnalysisResult)
admin.site.register(Recommendation)
admin.site.register(ChatSession)
admin.site.register(ChatMessage)
admin.site.register(ExperimentMemory)