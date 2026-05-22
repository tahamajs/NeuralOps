from django.contrib import admin
from .models import ExperimentTemplate, OpportunityIndex, ExperimentRecommendation, CreativeVariant

admin.site.register(ExperimentTemplate)
admin.site.register(OpportunityIndex)
admin.site.register(ExperimentRecommendation)
admin.site.register(CreativeVariant)