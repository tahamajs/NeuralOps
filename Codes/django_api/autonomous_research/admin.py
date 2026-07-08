from django.contrib import admin

from .models import (AutonomousDecision, PipelineArtifact, PipelineStep,
                     PipelineTemplate, ResearchPipeline)

admin.site.register(ResearchPipeline)
admin.site.register(PipelineStep)
admin.site.register(AutonomousDecision)
admin.site.register(PipelineArtifact)
admin.site.register(PipelineTemplate)
