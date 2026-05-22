from django.contrib import admin
from .models import FailureAnalysis, FixAction, ErrorPattern

admin.site.register(FailureAnalysis)
admin.site.register(FixAction)
admin.site.register(ErrorPattern)