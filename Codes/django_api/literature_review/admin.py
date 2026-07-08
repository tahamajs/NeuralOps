# literature_review/admin.py
from django.contrib import admin

from .models import (DeepReview, LiteratureProject, LiteratureSearch,
                     PaperReviewNote, ResearchGap, ResearchNotebookSection,
                     Topic)

for m in [
    LiteratureProject,
    Topic,
    ResearchNotebookSection,
    DeepReview,
    PaperReviewNote,
    LiteratureSearch,
    ResearchGap,
]:
    admin.site.register(m)
