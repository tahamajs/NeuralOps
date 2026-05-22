from django.contrib import admin
from .models import ResearchSession, Hypothesis, HypothesisReview, MetaReview, TournamentMatch

admin.site.register(ResearchSession)
admin.site.register(Hypothesis)
admin.site.register(HypothesisReview)
admin.site.register(MetaReview)
admin.site.register(TournamentMatch)