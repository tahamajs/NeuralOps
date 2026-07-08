from django.contrib import admin

from .models import ExpertProfile, ExpertReview, Project, ProjectApplication

admin.site.register(ExpertProfile)
admin.site.register(Project)
admin.site.register(ProjectApplication)
admin.site.register(ExpertReview)
