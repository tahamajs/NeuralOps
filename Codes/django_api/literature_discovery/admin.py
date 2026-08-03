from django.contrib import admin

from .models import (
    CitationPath,
    PaperFeedItem,
    PaperGraphCache,
    PaperRecommendation,
    ResearchLandscape,
)

admin.site.register(PaperGraphCache)
admin.site.register(CitationPath)
admin.site.register(PaperRecommendation)
admin.site.register(ResearchLandscape)
admin.site.register(PaperFeedItem)
