from django.contrib import admin

from .models import (CreativeVariant, ExperimentRecommendation,
                     ExperimentTemplate, OpportunityIndex)

admin.site.register(ExperimentTemplate)
admin.site.register(OpportunityIndex)
admin.site.register(ExperimentRecommendation)
admin.site.register(CreativeVariant)
