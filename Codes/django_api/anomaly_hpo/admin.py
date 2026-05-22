# admin.py
from django.contrib import admin
from .models import ExperimentLogAnalysis, HPORun, HPOTrial
admin.site.register(ExperimentLogAnalysis)
admin.site.register(HPORun)
admin.site.register(HPOTrial)
