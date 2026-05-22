# literature_review/admin.py
from django.contrib import admin
from .models import LiteratureProject, Topic, ResearchNotebookSection, DeepReview, PaperReviewNote, LiteratureSearch, ResearchGap
for m in [LiteratureProject, Topic, ResearchNotebookSection, DeepReview, PaperReviewNote, LiteratureSearch, ResearchGap]:
    admin.site.register(m)