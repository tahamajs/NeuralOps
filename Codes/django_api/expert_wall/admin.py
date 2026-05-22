from django.contrib import admin
from .models import ExpertProfile, Project, ProjectApplication, ExpertReview

admin.site.register(ExpertProfile)
admin.site.register(Project)
admin.site.register(ProjectApplication)
admin.site.register(ExpertReview)