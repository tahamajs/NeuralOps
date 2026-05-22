from django.contrib import admin
from .models import ExperimentDataset, SavedVisualization

admin.site.register(ExperimentDataset)
admin.site.register(SavedVisualization)