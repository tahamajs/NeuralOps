from django.contrib import admin

from .models import ErrorPattern, FailureAnalysis, FixAction

admin.site.register(FailureAnalysis)
admin.site.register(FixAction)
admin.site.register(ErrorPattern)
