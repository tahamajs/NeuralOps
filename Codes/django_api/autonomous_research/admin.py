from django.contrib import admin
from .models import ResearchPipeline, PipelineStep, AutonomousDecision, PipelineArtifact, PipelineTemplate

admin.site.register(ResearchPipeline)
admin.site.register(PipelineStep)
admin.site.register(AutonomousDecision)
admin.site.register(PipelineArtifact)
admin.site.register(PipelineTemplate)