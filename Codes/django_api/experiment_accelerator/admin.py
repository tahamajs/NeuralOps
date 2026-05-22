from django.contrib import admin
from .models import ExperimentInsight, ExperimentRecommendation, ExperimentSummary

admin.site.register(ExperimentInsight)
admin.site.register(ExperimentRecommendation)
admin.site.register(ExperimentSummary)